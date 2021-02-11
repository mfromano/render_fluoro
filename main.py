import os
import subprocess                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
import argparse

parser = argparse.ArgumentParser(description='Crop tif files')
parser.add_argument('--indir', dest='initdir', default=os.path.abspath('.'))
args = parser.parse_args()
init_dir = args.initdir
dirs = os.listdir(init_dir)
for dir in dirs:
    prefix = os.path.split(init_dir)[1]
    curdir = os.path.join(init_dir, dir)
    len_dir = len(os.listdir(curdir))
    if len_dir > 1:
        fps_out = 5
        scaling = 1
        newname = '_'.join([prefix, dir])
        print(f'python movie_maker.py --indir {os.path.join(init_dir,dir)} ' +\
                        f'--outputpref {newname} --fps_in 5 --fps_out {fps_out} --scaling_factor {scaling}')
        subprocess.run( f'python movie_maker.py --indir {os.path.join(init_dir,dir)} ' +\
                        f'--outputpref {newname} --fps_in 5 --fps_out {fps_out} --scaling_factor {scaling}', shell=True)