# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:04:13 2020

@author: mfromano
"""

import argparse
import os
from tqdm import tqdm
import tkinter as tk
from glob import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from eval_new import main
from tqdm import tqdm

class Cine(object):
    def __init__(self, initdir='.', outdir='./cropped_files', outputname='video.mp4', fps=5):

        self.initdir = initdir
        self.outdir = outdir
        self.outputname = outputname
        self.fps = fps
        self.checkpoint = './SuperSloMo/checkpoints/SuperSloMo.ckpt'
        self.filenames = np.sort(glob(os.path.join(initdir, '*.tif')))
        self._load_cine_frames()
        self._crop_cine()

    def _load_cine_frames(self):
        filenames = self.filenames
        img_stack = []
        for filename in filenames:
            img = cv2.imread(filename)
            if img is None:
                print('Skipping ' + filename)
                continue
            img_stack.append(img)
        self.img_stack = img_stack

    def _crop_cine(self):
        img_stack = self.img_stack
        img = np.max(img_stack[0],axis=2).astype(np.uint8)
        white_mask = np.ones(img.shape[:2], dtype=np.uint8)
        x, y, w, h = 0, 0, 1000, 100
        white_mask[y:y + h, x:x + w] = 0
        tmp = cv2.bitwise_and(
            img, img, mask=white_mask.astype(np.uint8)
        )
        tmp = tmp > 0
        [x, y, w, h] = self.bounding_box(tmp.astype(np.uint8))
        img_stack = [im[y:y + h, x:x + w, ...] for im in img_stack]
        self.img_stack = img_stack
        return img_stack

    def bounding_box(self, img):
        x, y, w, h = cv2.boundingRect(img)
        return [x, y, w, h]

    def write_cine_frames(self, img_stack=None, prefix=''):
        if img_stack is None:
            img_stack = self.img_stack
        if prefix == '':
            prefix = 'img_'
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        new_filenames = []
        for idx in range(len(img_stack)):
            fname = '{}/{}{}.png'.format(self.outdir,
                                          prefix, str(idx).zfill(4))
            img = img_stack[idx]
            cv2.imwrite(fname,img)
            new_filenames.append(fname)
        return new_filenames

    # adoped from https://github.com/ferreirafabio/video2tfrecord under MIT license
    def compute_flow(self):
        if not os.path.isdir(os.path.join(self.outdir, 'flow')):
            os.mkdir(os.path.join(self.outdir, 'flow'))
        flow = None
        output = []
        hsv = np.zeros(self.img_stack[0].shape)
        hsv[..., 1] = 255
        print('Computing flow...')
        for idx in tqdm(range(len(self.img_stack)-1)):
            curr_frame = self.img_stack[idx]
            curr_frame = cv2.cvtColor(
                curr_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            next_frame = self.img_stack[idx + 1]
            next_frame = cv2.cvtColor(
                next_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                curr_frame, next_frame, flow=None, pyr_scale=0.8, levels=15, winsize=20, iterations=10, poly_n=7, poly_sigma=1.5, flags=10)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            col = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            output.append(col)
        return output


    def write_cine(self, img_stack=None, outputname='', fps=-1):
        if img_stack is None:
            img_stack = self.img_stack
        if outputname == '':
            outputname = self.outputname
        if fps < 1:
            fps = self.fps
        codec = cv2.VideoWriter_fourcc(*'mp4v')  # use the mp4v codec
        shape = img_stack[0].shape
        output_writer = cv2.VideoWriter(
            os.path.join(os.path.abspath('.'), outputname),
            codec,
            float(fps), (shape[1], shape[0]))
        print('Writing video')
        for idx in tqdm(range(len(img_stack))):
            output_writer.write(img_stack[idx])
        output_writer.release()
        return outputname

    def interpolate_frames(self, video, factor=5, batch_size=2, fps=20, suffix=''):
        if int(batch_size) < 2:
            batch_size = len(self.img_stack)
        main(video, self.checkpoint, self.outdir + '/interpolated_video{}.mp4'.format(suffix), int(batch_size), factor, fps)
        interpolated_videoname = self.outdir + '/interpolated_video{}.mp4'.format(suffix)
        return interpolated_videoname

    def frames_from_video(self, videoname=None):
        if videoname is None:
            videoname = self.interpolated_videoname
        video = cv2.VideoCapture(videoname)
        image_array = []
        while True:
            success, im = video.read()
            if not success:
                break
            image_array.append(im)
        return image_array

    def upsample_frames(self, img_stack=None, upsample_ratio=-1):
        if img_stack is None:
            img_stack = self.img_stack
        y, x = img_stack[0].shape[:2]
        print('Initial dims: {}x{}'.format(x,y))
        if upsample_ratio < 0:
            max_dim = np.max([y,x])
            upsample_ratio = 3000//max_dim
        print('Final dims: {}x{}'.format(x*upsample_ratio,y*upsample_ratio))
        upsampled_stack = []
        print('upsampling video')
        for idx in tqdm(range(len(img_stack))):
            upsampled_stack.append(cv2.resize(img_stack[idx], (
                x * upsample_ratio, y * upsample_ratio), interpolation=cv2.INTER_AREA))
        return upsampled_stack

    def blend_flow_and_stack(self, img_stack, flow_stack, alpha=0.8):
        output_array = []
        for idx in range(len(flow_stack)):
            flow = cv2.cvtColor(flow_stack[idx], cv2.COLOR_BGR2GRAY)
            flow = cv2.normalize(flow, None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            flow = cv2.merge([np.zeros_like(flow), np.zeros_like(flow), flow])
            img = img_stack[idx]
            output_array.append(cv2.addWeighted(img,
                                    alpha, flow,1-alpha,0))
        return output_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop tif files')
    parser.add_argument('--indir', dest='initdir',
                        default=os.path.abspath('.'))
    parser.add_argument('--outdir', dest='outdir',
                        default=os.path.join(
                            os.path.abspath('.'), 'cropped_files'))
    parser.add_argument('--outputpref', dest='outputprefix',
                        default=os.path.join(os.path.abspath('.'), 'cropped_files', 'video'))
    parser.add_argument('--fps_in', dest='fps_in', default=5)
    parser.add_argument('--scaling_factor', dest='factor',
                        default=5)
    parser.add_argument('--fps_out', dest='fps_out', default=20)
    parser.add_argument('--alpha', dest='alpha', default=0.8)
    parser.add_argument('--get_flow', dest='get_flow', default=None)
    parser.add_argument('--upsample_ratio', dest='upsample',default=-1)
    parser.add_argument('--batch_size', dest='batch', default=2)

    args = parser.parse_args()
    cine = Cine(initdir=args.initdir,
                    outdir=args.outdir,
                    outputname=args.outputprefix,
                    fps=args.fps_in)
    cine.write_cine_frames()
    if args.get_flow:
        flow = cine.compute_flow()
    fluoro_video = cine.write_cine(cine.img_stack, 'cine_cropped.mp4', fps=int(args.fps_in))
    if int(args.factor) > 1:
        interpolated_fluoro = cine.interpolate_frames(
                                        fluoro_video,
                                        batch_size=args.batch,
                                        factor=args.factor,
                                        fps=args.fps_out,
                                        suffix='_fluoro')
    else:
        interpolated_fluoro=fluoro_video

    frames = cine.frames_from_video(interpolated_fluoro)
    frames_upsampled = cine.upsample_frames(frames, int(args.upsample))
    cine.write_cine(frames_upsampled, outputname = '{}_fluoro_cine.mp4'.format(args.outputprefix), fps=int(args.fps_out))

    if args.get_flow:
        flow_video = cine.write_cine(flow, 'flow_cropped.mp4', fps=int(args.fps_in))

        interpolated_flow = cine.interpolate_frames(
            flow_video,
            factor=args.factor,
            fps=args.fps_out,
            suffix='_flow')
        flow = cine.frames_from_video(interpolated_flow)
        flow_upsampled = cine.upsample_frames(flow, int(args.upsample))
        cine.write_cine(flow_upsampled, outputname = '{}_flow_cine.mp4'.format(args.outputprefix), fps=int(args.fps_out))
        stack = cine.blend_flow_and_stack(frames_upsampled,flow_upsampled,alpha=0.5)
        cine.write_cine(stack, '{}_cine_blend.mp4'.format(args.outputprefix), fps=int(args.fps_out))
