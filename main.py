import os
import subprocess                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
import argparse

parser = argparse.ArgumentParser(description='Crop tif files')
parser.add_argument('--indir', dest='initdir', default=os.path.abspath('.'))
args = parser.parse_args()
bigdirs = os.listdir(args.initdir)
for bigdir in bigdirs:
    init_dir = os.path.join(args.initdir, bigdir)
    dirs = os.listdir(init_dir)
    for dir in dirs:
        prefix = os.path.split(init_dir)[1]
        curdir = os.path.join(init_dir, dir)
        len_dir = len(os.listdir(curdir))
        if len_dir > 25:
            fps_out = 20
            scaling = 3
        else: # this is generally a DSA
            fps_out = 5
            scaling = 3
        newname = '_'.join([prefix, dir])
        print(f'python movie_maker.py --indir {os.path.join(init_dir,dir)} ' +\
                        f'--outputpref {newname} --fps_in 3 --fps_out {fps_out} --scaling_factor {scaling}')
        subprocess.run( f'python movie_maker.py --indir {os.path.join(init_dir,dir)} ' +\
                        f'--outputpref {newname} --fps_in 3 --fps_out {fps_out} --scaling_factor {scaling}', shell=True)