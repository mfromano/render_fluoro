# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:04:13 2020

@author: mfromano
"""

import cv2
from tkinter import filedialog, Tk
import numpy as np


def prompt_fnames():
    filenames = filedialog.askopenfilenames(initialdir='.', title='Select files', filetypes=(('jpg files', '*.jpg'),
                                                                                             ('all files', '*.*'),
                                                                                             ('png files', '*.png')))
    return filenames


def auto_crop(filenames, ref_no=0):
    first_file = cv2.imread(filenames[ref_no])
    non_red = identify_red_regions(first_file)

    first_file = cv2.cvtColor(first_file, cv2.COLOR_BGR2GRAY)
    first_file = cv2.bitwise_and(first_file, non_red)
    # first apply morphological
    kernel = np.ones((5, 5), dtype=np.uint8)
    _, mask = cv2.threshold(first_file, thresh=0, maxval=1, type=cv2.THRESH_BINARY)
    mask[np.where(mask > 0)] = 1
    dilate = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return dilate


def bounding_box(img):
    x, y, w, h = cv2.boundingRect(img)
    return [x,y,w,h]


def apply_bounding_box(image_files, bb):
    img = cv2.imread(image_files[0])
    [x,y,w,h] = bb
    img_stack = np.expand_dims(img[y:y + h, x:x + w, :], axis=3)
    for filename in image_files[1:]:
        img = cv2.imread(filename)
        img_stack = np.append(img_stack,np.expand_dims(img[y:y+h, x:x+w, :], axis=3), axis=3)
    return img_stack

# from https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv, user derricw
def identify_red_regions(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-1, 0)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 1
    output_img[np.where(mask != 0)] = 0
    kernel = np.ones((20, 20), dtype=np.uint8)
    opening = cv2.morphologyEx(output_img, cv2.MORPH_ERODE, kernel)
    output_img = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)

    return output_img

def write_video(output_filename, image_stack, fps=5):
    codec = cv2.VideoWriter_fourcc(*'MP42')  # use the mp4v codec
    shape = image_stack.shape[0:2]
    output_writer = cv2.VideoWriter(output_filename, codec, float(fps), (shape[1], shape[0]))
    for im_index in range(image_stack.shape[-1]):
        output_writer.write(np.squeeze(image_stack[:, :, :, im_index]))
    output_writer.release()


if __name__ == '__main__':
    # filenames = prompt_fnames()
    dilate = auto_crop(filenames)
    bb = bounding_box(dilate)
    img_stack = apply_bounding_box(filenames, bb)
    write_video('hello_world.avi', img_stack)
