# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:04:13 2020

@author: mfromano
"""

import cv2
import tkinter.filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
from PIL import Image
import os


class MakeCine(object):

    def __init__(self):
        parser = argparse.ArgumentParser(description='Crop tif files')
        parser.add_argument('--indir',dest='initdir',
                            default=os.path.abspath('.'))
        parser.add_argument('--outdir',dest='outdir',
                            default=os.path.join(
                                    os.path.abspath('.'),'cropped_files'))
        parser.add_argument('--outputname',dest='outputname',
                            default=os.path.join(os.path.abspath('.'),'cropped_files','video.avi'))
        parser.add_argument('--fps',dest='fps',default=10)
        args = parser.parse_args()
        self.initdir = args.initdir
        self.outdir = args.outdir
        self.outputname = args.outputname
        self.fps = args.fps
        self.filenames = np.sort(glob(os.path.join(args.initdir,'*.tif')))

    def _prompt_fnames(self):
        root = tk.Tk()
        filenames = tkinter.filedialog.askopenfilenames(parent=root,
                initialdir='.', title='Select files',
                filetypes=(
                        ('tiff files', '*.tif'),
                ))
        self.filenames = filenames
        root.destroy()

    def _prompt_output_name(self):
        root = tk.Tk()
        tk.Label(root, text='Please enter output filename: ').grid(row=0)
        fname = tk.Entry(root)
        fname.grid(row=0, column=1)
        self.outputname = ''
        def enter(event):
            self.outputname = fname.get()
            root.destroy()
        root.bind('<Return>', enter)
        tk.mainloop()

    def load_videos(self, filenames=None):
        if filenames is None:
            filenames = self.filenames
        img = cv2.imread(filenames[0])
        img_stack = np.expand_dims(img, axis=3)
        for filename in filenames[1:]:
            img = cv2.imread(filename)
            img_stack = np.append(img_stack, np.expand_dims(img, axis=3), axis=3)
        self.img_stack = img_stack

    def remove_white_regions(self):
        img = np.squeeze(self.img_stack[:,:,:,0])
        white_mask = np.ones(img.shape[0:2], dtype=np.uint8)
        x,y,w,h = 0, 0, 1000, 100
        white_mask[y:y + h, x:x + w] = 0
        tmp = cv2.bitwise_and(
                np.squeeze(self.img_stack[:,:,0,0]),
                np.squeeze(self.img_stack[:,:,0,0]),
                mask=white_mask.astype(np.uint8)
        )
        tmp = tmp > 0
        [x, y, w, h] = self.bounding_box(tmp.astype(np.uint8))
        self.img_stack = self.img_stack[y:y + h, x:x + w,:,:]
        plt.imshow(self.img_stack[:,:,0,0])

    def bounding_box(self, img):
        x, y, w, h = cv2.boundingRect(img)
        return [x,y,w,h]

    def write_frames(self):
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        for idx in range(self.img_stack.shape[-1]):
            img = Image.fromarray(np.squeeze(self.img_stack[:,:,:,idx]))
            img.save('{}/img{}.png'.format(self.outdir, str(idx).zfill(4)))

    def interpolate_frames(self):
        if not os.path.isdir(os.path.join(self.outdir, 'interpolated')):
            os.mkdir(os.path.join(self.outdir, 'interpolated'))
        arg = 'python3 unsupervised-video-interpolation/eval.py --model CycleHJSuperSloMo --num_interp 5 --flow_scale 2.0 --val_file {} --name superslomo --save {} --post_fix img_interp --resume unsupervised-video-interpolation/pretrained_models/baseline_superslomo_adobe+youtube.pth --write_images --val_sample_rate 0 --val_step_size 1'.format(self.outdir, os.path.join(self.outdir, 'interpolated'))
        print(arg)
        #os.system(arg)
        self.image_files_new = np.sort(glob('{}/{}/{}'.format(self.outdir,'interpolated/superslomo','*.png')))
        self.load_videos(self.image_files_new)

    def write_video(self):
        fps = self.fps
        codec = cv2.VideoWriter_fourcc(*'XVID')  # use the mp4v codec
        shape = self.img_stack.shape[0:2]
        output_writer = cv2.VideoWriter(os.path.join(os.path.abspath('.'),
                                                      self.outputname),
                                        codec,
                        float(fps), (shape[1],shape[0]))
        for im_index in range(self.img_stack.shape[-1]):
            output_writer.write(np.squeeze(self.img_stack[:, :, :, im_index]))
        output_writer.release()

if __name__ == '__main__':
    cine = MakeCine()
    cine.load_videos()
    cine.remove_white_regions()
    cine.write_frames()
    cine.interpolate_frames()
    cine.write_video()
