#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from IPython.core.debugger import set_trace
import time

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)


# Save the chains as separate files. 
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
tmp_dir= masif_opts['tmp_dir']
protonated_file = tmp_dir+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# Extract chains of interest.
out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

# Compute MSMS of surface w/hydrogens, 
try:
    t1 = time.time()
    t1_cpu = time.process_time()
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+".pdb",\
        protonate=True)
    t2 = time.time()
    t2_cpu = time.process_time()
    print('MSMS time: {}\n'.format(t2-t1))
    print('MSMS cpu time: {}\n'.format(t2_cpu-t1_cpu))
except:
    set_trace()

# Compute "charged" vertices
if masif_opts['use_hbond']:
    t1 = time.time()
    t1_cpu = time.process_time()
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)
    t2 = time.time()
    t2_cpu = time.process_time()
    print('HBond time: {}\n'.format(t2-t1))
    print('HBond cpu time: {}\n'.format(t2_cpu-t1_cpu))

# For each surface residue, assign the hydrophobicity of its amino acid. 
if masif_opts['use_hphob']:
    t1 = time.time()
    t1_cpu = time.process_time()
    vertex_hphobicity = computeHydrophobicity(names1)
    t2 = time.time()
    t2_cpu = time.process_time()
    print('HPhob time: {}\n'.format(t2-t1))
    print('HPhob cpu time: {}\n'.format(t2_cpu-t1_cpu))


# If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
vertices2 = vertices1
faces2 = faces1

# Fix the mesh.
t1 = time.time()
t1_cpu = time.process_time()
mesh = pymesh.form_mesh(vertices2, faces2)
regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
t2 = time.time()
t2_cpu = time.process_time()
print('Fix mesh time: {}\n'.format(t2-t1))
print('Fix mesh cpu time: {}\n'.format(t2_cpu-t1_cpu))

# Compute the normals
t1 = time.time()
t1_cpu = time.process_time()
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
t2 = time.time()
t2_cpu = time.process_time()
print('Compute normals time: {}\n'.format(t2-t1))
print('Compute normals cpu time: {}\n'.format(t2_cpu-t1_cpu))
# Assign charges on new vertices based on charges of old vertices (nearest
# neighbor)

if masif_opts['use_hbond']:
    t1 = time.time()
    t1_cpu = time.process_time()
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hbond, masif_opts)
    t2 = time.time()
    t2_cpu = time.process_time()
    print('Assign HBond time: {}\n'.format(t2-t1))
    print('Assign HBond cpu time: {}\n'.format(t2_cpu-t1_cpu))

if masif_opts['use_hphob']:
    t1 = time.time()
    t1_cpu = time.process_time()
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hphobicity, masif_opts)
    t2 = time.time()
    t2_cpu = time.process_time()
    print('Assign HPhob time: {}\n'.format(t2-t1))
    print('Assign HPhob cpu time: {}\n'.format(t2_cpu-t1_cpu))

if masif_opts['use_apbs']:
    t1 = time.time()
    t1_cpu = time.process_time()
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)
    t2 = time.time()
    t2_cpu = time.process_time()
    print('APBS time: {}\n'.format(t2-t1))
    print('APBS cpu time: {}\n'.format(t2_cpu-t1_cpu))

iface = np.zeros(len(regular_mesh.vertices))
if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    # Compute the surface of the entire complex and from that compute the interface.
    v3, f3, _, _, _ = computeMSMS(pdb_filename,\
        protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    full_regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d) # Square d, because this is how it was in the pyflann version.
    assert(len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                        iface=iface)

else:
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])
shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
shutil.copy(out_filename1+'.pdb', masif_opts['pdb_chain_dir']) 
