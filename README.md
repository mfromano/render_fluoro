# Render Fluoro
This repository can be used to automatically crop, concatenate, upscale, and upsample CINE videos or TIFF image stacks. main.py illustrates an example.
The arguments that can be passed are as follows:
```
--indir=${directory containing tiff files}
--outdir=${directory where output video(s) will be saved)}
--fps_in=${frames per second at which original video was recorded (generally, 5Hz for fluoro, 2Hz for DSA. Doesn't matter what this value is for image stacks). Default 5.}
--scaling_factor=${multiplier for frame rate. I.e. if original rate is 5Hz, enter "2" for 10 Hz, or to have twice as many frames. Should be 1 for an image stack. Default 5.}
--output_pref
--fps_out=${frame rate to save output. For same rate as input, enter fps_in*scaling_factor. Default 20.}
--upsample_ratio=${multiplier to upsample the pixels. For example, to increase pixel dims of a 1024x1024 by 2x to 2048x2048, enter 2. Default upsamples to keep largest dimension < 2000 pixels}
--batch_size=${number of frames to use for frame interpolation. default=2 seems to work well}
--get_flow=${boolean value, 1 or 0. Default None. optional, asks whether or not you want to estimate optical flow using farneback method (cv2 library). Still a work in progress}
--alpha=${blending value for overlaying flow on top of cine video/image stack. Default 0.8}

An example:
python movie_maker.py --indir example_dir --outputpref example_ --fps_in 5 --fps_out 5 --scaling_factor 1
```
This line of code will take tiff files from example_dir, automatically ordered, crop them, produce a video at 5Hz with no frame upscaling, and will upsample the pixel dimensions using the default method (argmax where max dimension < 2000). No flow video will be produced. output video will be called example_fluoro_cine.mp4.

To install dependencies for windows:
```
conda create -n render_fluoro python=3.6 anaconda
conda activate render_fluoro
conda install pytorch=0.4.1 cuda92 torchvision==0.2.1 -c pytorch
conda install opencv
```
