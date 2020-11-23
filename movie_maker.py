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
        # tk.Label(root, text='Please enter FPS: ').grid(row=1)
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

    def upsample_frames(self, upsample_ratio=2):
        y,x,c,z = self.img_stack.shape
        upsampled_stack = np.zeros((y*upsample_ratio, x*upsample_ratio, c, z))
        for idx in range(z):
            upsampled_stack[:,:,:,idx] = cv2.resize(np.squeeze(self.img_stack[:,:,:,idx]), (x*upsample_ratio, y*upsample_ratio), interpolation=cv2.INTER_AREA)
        self.img_stack = upsampled_stack.astype(np.uint8) 

    def bounding_box(self, img):
        x, y, w, h = cv2.boundingRect(img)
        return [x,y,w,h]

    def write_frames(self, img_stack=None, prefix=''):
        if img_stack is None:
            img_stack = self.img_stack
        if prefix == '':
            prefix = 'img'
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        for idx in range(img_stack.shape[-1]):
            img = Image.fromarray(np.squeeze(img_stack[:,:,:,idx]), 'RGB')
            img.save('{}/{}{}.png'.format(self.outdir, prefix, str(idx).zfill(4)))

    # adoped from https://github.com/ferreirafabio/video2tfrecord
    def get_flow(self):
        if not os.path.isdir(os.path.join(self.outdir, 'flow')):
            os.mkdir(os.path.join(self.outdir, 'flow'))
        flow=None
        output = np.zeros_like(self.img_stack)
        hsv = np.zeros(self.img_stack.shape[0:3])
        hsv[...,1] = 255
        for idx in range(self.img_stack.shape[-1]-1):
            curr_frame = np.squeeze(self.img_stack[:,:,:,idx])
            curr_frame = cv2.cvtColor(curr_frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            next_frame = np.squeeze(self.img_stack[:,:,:,idx+1])
            next_frame = cv2.cvtColor(next_frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(curr_frame, next_frame, flow=flow, pyr_scale=0.8, levels=15, winsize=5, iterations=10, poly_n=5, poly_sigma=0, flags=10)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            col= cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            output[:,:,:,idx] = col
        self.flow = output
        self.write_frames(self.flow, prefix='flow_')

    def write_video(self, img_stack=None, outputname=''):
        if img_stack is None:
            img_stack = self.img_stack
        if outputname == '':
            outputname = self.output_name
        fps = self.fps
        codec = cv2.VideoWriter_fourcc(*'XVID')  # use the mp4v codec
        shape = img_stack.shape[0:2]
        output_writer = cv2.VideoWriter(os.path.join(os.path.abspath('.'),
                                                      outputname),
                                        codec,
                        float(fps), (shape[1],shape[0]))
        for im_index in range(img_stack.shape[-1]):
            output_writer.write(np.squeeze(img_stack[:, :, :, im_index]))
        output_writer.release()

    def compute_flow(self):
       pass 


if __name__ == '__main__':
    cine = MakeCine()
    cine.load_videos()
    cine.remove_white_regions()
    cine.upsample_frames()
    cine.write_frames()
    cine.write_video()
    cine.get_flow()
    cine.write_video(cine.flow, 'flow_video.avi')