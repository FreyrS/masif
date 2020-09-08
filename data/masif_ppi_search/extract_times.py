import glob
import numpy as np

time_names = ['MSMS time','HBond time','HPhob time',
'Fix mesh time','Compute normals time','Assign HBond time','Assign HPhob time',
'APBS time','Polar coordinates time','Curvature computation time','DDC computation time']


for time_name in time_names:
    for f in glob.glob('exelogs/*.out'):
        with open(f) as fo:
            lines = fo.readlines()
        lines = [l for l in lines if time_name in l]

        times = [float(l.split()[-1]) for l in lines]

    print(time_name)
    print('Mean',np.mean(times))
    print('Median',np.median(times))
    print('Std',np.std(times))