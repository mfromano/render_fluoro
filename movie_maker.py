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
import os


class MakeCine(object):

    def __init__(self):
        self._prompt_fnames()
        self._prompt_output_name()

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

    def load_videos(self):
        img = cv2.imread(self.filenames[0])
        img_stack = np.expand_dims(img, axis=3)
        for filename in self.filenames[1:]:
            img = cv2.imread(filename)
            img_stack = np.append(img_stack, np.expand_dims(img, axis=3), axis=3)
        self.img_stack = img_stack

    def remove_white_regions(self):
        img = np.squeeze(self.img_stack[:,:,:,0])
        white_mask = np.sum(img > 250, axis=2) == 3
        # upper mask (170-180)
        kernel = np.ones((10, 10), dtype=np.uint8)
        opening = cv2.dilate(white_mask.astype(np.uint8), kernel)
        [x,y,w,h] = self.bounding_box(opening)
        white_mask = 1-white_mask
        white_mask[y:y + h, x:x + w] = 0
        for i in range(self.img_stack.shape[-1]):
            for j in range(self.img_stack.shape[-2]):
                    self.img_stack[:,:,j,i] = cv2.bitwise_and(
                    np.squeeze(self.img_stack[:,:,j,i]),
                    np.squeeze(self.img_stack[:,:,j,i]),
                    mask=white_mask.astype(np.uint8))
        np_mask = np.squeeze(np.sum(self.img_stack[:,:,:,0] > 0, axis=2))
        [x, y, w, h] = self.bounding_box(np_mask.astype(np.uint8))
        self.img_stack = self.img_stack[y:y + h, x:x + w,:,:]

    def bounding_box(self, img):
        x, y, w, h = cv2.boundingRect(img)
        return [x,y,w,h]

    def interpolate_frames(self, multiplier=10):
        sz = self.img_stack.shape
        new_img_stack = np.empty((sz[0:3] + ((sz[3]-1)*multiplier,)))
        for i in range(sz[-1]-1):
            for j in range(multiplier):
                a = 1-(j/multiplier)
                b = j/multiplier
                new_img_stack[:,:,:,i*multiplier+j] = \
                np.round(self.img_stack[:,:,:,i]*a + \
                self.img_stack[:,:,:,i+1]*b)
        self.img_stack = new_img_stack.astype(np.uint8)

    def compute_and_write_gradient(self):
        self.gradient = np.gradient(self.img_stack, axis=3)

    def write_video(self, fps=30):
        codec = cv2.VideoWriter_fourcc(*'XVID')  # use the mp4v codec
        shape = self.img_stack.shape[0:2]
        output_writer = cv2.VideoWriter(os.path.join(os.path.abspath('.'),
                                                      self.outputname),
                                        codec,
                        float(fps), (shape[1],shape[0]))
        for im_index in range(self.img_stack.shape[-1]):
            output_writer.write(np.squeeze(self.img_stack[:, :, :, im_index]))
        output_writer.release()

    def write_gradient(self, fps=30):
        codec = cv2.VideoWriter_fourcc(*'XVID')  # use the mp4v codec
        shape = self.gradient.shape[0:2]
        output_writer = cv2.VideoWriter(os.path.join(os.path.abspath('.'),
                                                     'gradient_' +
        self.outputname),
                                        codec,
                                        float(fps), (shape[1], shape[0]))
        for im_index in range(self.gradient.shape[-1]):
            output_writer.write(
                np.squeeze(self.gradient[:, :, :, im_index]))
        output_writer.release()

if __name__ == '__main__':
    cine = MakeCine()
    cine.load_videos()
    cine.remove_white_regions()
    cine.interpolate_frames()
    cine.compute_and_write_gradient()
    cine.write_gradient()
    cine.write_video()

    #
    # # from https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv, user derricw
    # def identify_red_regions(self, img):
    #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    #     # lower mask (0-1, 0)
    #     lower_red = np.array([0, 30, 30])
    #     upper_red = np.array([10, 255, 255])
    #     mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    #
    #     # upper mask (170-180)
    #     lower_red = np.array([170, 30, 30])
    #     upper_red = np.array([180, 255, 255])
    #     mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    #
    #     # join my masks
    #     mask = mask0 + mask1
    #
    #     # set my output img to zero everywhere except my mask
    #     output_img = img.copy()
    #     output_img[np.where(mask == 0)] = 0
    #     output_img[np.where(mask != 0)] = 1
    #     output_img = np.sum(output_img,axis=2)
    #     output_img[output_img > 1] = 1
    #     output_img = output_img.astype('uint8')
    #     kernel = np.ones((30, 30), dtype=np.uint8)
    #     opening = cv2.dilate(output_img, kernel)
    #     [x,y,w,h] = self.bounding_box(opening)
    #     opening[y:y + h, x:x + w] = 1
    #     opening = cv2.bitwise_not(opening)
    #     return opening