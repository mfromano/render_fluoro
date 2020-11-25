from movie_maker import Cine
a = Cine(initdir='/home/mfromano/data/fluoro/011819/series3')
filenames = a.write_cine_frames()
flow = a.compute_flow()
video = a.write_cine(a.img_stack)
flow_video = a.write_cine(flow, 'flow_cropped.mp4')
interpolated_video = a.interpolate_frames(video, suffix='_fluoro')
interpolated_video_flow = a.interpolate_frames(flow_video, suffix='_flow')
frames = a.frames_from_video(interpolated_video)
frames_upsampled = a.upsample_frames(frames)
a.write_cine(img_stack=frames_upsampled, outputname='fluoro_cine.mp4')

flow = a.frames_from_video(interpolated_video_flow)
flow_upsampled = a.upsample_frames(flow)
a.write_cine(flow_upsampled, 'flow_cine.mp4')

stack = a.blend_flow_and_stack(frames_upsampled,flow_upsampled,alpha=0.5)
a.write_cine(stack, 'cine_blend.mp4')
