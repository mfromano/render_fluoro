"""
Converts a Video to SuperSloMo version
"""
from time import time
import click
import cv2
import torch
import os
from PIL import Image
import numpy as np
from SuperSloMo import model
from torchvision import transforms
from torch.functional import F
from torch.utils.data import Dataset, DataLoader
from glob import glob


class FileSystemData(Dataset):

    def __init__(self, files):
        self.files = files
        first_fi = Image.open(self.files[0])
        self.w, self.h = first_fi.size
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        return img

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()
if device != "cpu":
    mean = [0.429, 0.431, 0.397]
    mea0 = [-m for m in mean]
    std = [1] * 3
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std), trans_backward])

flow = model.UNet(6, 4).to(device)
interp = model.UNet(20, 5).to(device)
back_warp = None

def setup_back_warp(w, h):
    global back_warp
    with torch.set_grad_enabled(False):
        back_warp = model.backWarp(w, h, device).to(device)

def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')
    interp.load_state_dict(states['state_dictAT'])
    flow.load_state_dict(states['state_dictFC'])

def interpolate_batch(frames, factor):
    frame0 = torch.stack(frames[:-1])
    frame1 = torch.stack(frames[1:])

    i0 = frame0.to(device)
    i1 = frame1.to(device)
    ix = torch.cat([i0, i1], dim=1)

    flow_out = flow(ix)
    f01 = flow_out[:, :2, :, :]
    f10 = flow_out[:, 2:, :, :]

    frame_buffer = []
    for i in range(1, factor):
        t = i / factor
        temp = -t * (1 - t)
        co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

        ft0 = co_eff[0] * f01 + co_eff[1] * f10
        ft1 = co_eff[2] * f01 + co_eff[3] * f10

        gi0ft0 = back_warp(i0, ft0)
        gi1ft1 = back_warp(i1, ft1)

        iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
        io = interp(iy)
        torch.cuda.empty_cache()

        ft0f = io[:, :2, :, :] + ft0
        ft1f = io[:, 2:4, :, :] + ft1
        vt0 = F.sigmoid(io[:, 4:5, :, :])
        vt1 = 1 - vt0

        gi0ft0f = back_warp(i0, ft0f)
        gi1ft1f = back_warp(i1, ft1f)

        co_eff = [1 - t, t]

        ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
               (co_eff[0] * vt0 + co_eff[1] * vt1)

        frame_buffer.append(ft_p)

    return frame_buffer

def load_batch(dataset, batch_size, batch, w, h, done):
    if len(batch) > 0:
        batch = [batch[-1]]
    for i in range(batch_size):
        if done+i > len(dataset)-1:
            break
        frame = dataset[done+i]
        frame = frame.resize((w, h), Image.ANTIALIAS)
        frame = frame.convert('RGB')
        frame = trans_forward(frame)
        batch.append(frame)
    return batch


def denorm_frame(frame, w0, h0):
    frame = frame.cpu()
    frame = trans_backward(frame)
    frame = frame.resize((w0, h0), Image.BILINEAR)
    frame = frame.convert('RGB')
    return np.array(frame)[:, :, ::-1].copy()


def convert_video(filenames, outdir, factor, batch_size=2, output_fps=10):
    dataset = FileSystemData(filenames)
    count = len(dataset)
    if not os.path.isdir(os.path.join(outdir, 'interp')):
        os.mkdir(os.path.join(outdir, 'interp'))
    w0, h0 = dataset.w, dataset.h

    w, h = (w0 // 32) * 32, (h0 // 32) * 32
    setup_back_warp(w, h)

    done = 0
    output_idx = 0
    batch = []
    while True:
        batch = load_batch(dataset, batch_size, batch, w, h, done)
        if len(batch) == 1:
            break
        done += len(batch) - 1
        intermediate_frames = interpolate_batch(batch, factor)
        intermediate_frames = list(zip(*intermediate_frames))

        for fid, iframe in enumerate(intermediate_frames):
            img = transforms.ToPILImage()(batch[fid]).convert("RGB")
            img.save('{}/interp/interp_img_{}.png'.format(outdir,str(output_idx).zfill(4)))
            output_idx += 1
            for frm in iframe:
                img = transforms.ToPILImage()(batch[fid]).convert("RGB")
                img.save('{}/interp/interp_img_{}.png'.format(outdir,str(output_idx).zfill(4)))
                output_idx += 1
        try:
            yield len(batch), done, count
        except StopIteration:
            break

def main(input, checkpoint, output, batch, scale, fps):
    avg = lambda x, n, x0: (x * n/(n+1) + x0 / (n+1), n+1)
    load_models(checkpoint)
    t0 = time()
    n0 = 0
    fpx = 0
    for dl, fd, fc in convert_video(input, output, int(scale), int(batch), output_fps=int(fps)):
        fpx, n0 = avg(fpx, n0, dl / (time() - t0))
        prg = int(100*fd/fc)
        eta = (fc - fd) / fpx
        print('\rDone: {:03d}% FPS: {:05.2f} ETA: {:.2f}s'.format(prg, fpx, eta) + ' '*5, end='')
        t0 = time()
    new_files = glob('{}/interp/interp_img_*.png'.format(output))
    return new_files


if __name__ == '__main__':
    main()
