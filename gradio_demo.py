import gradio as gr
from sklearn import calibration
import torch
import os
from src import sample_svd, focal2fov, rotation_matrix_to_euler
from src.calibrate_cameras import get_camera_params_from_frames
from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
from transformers import CLIPVisionModelWithProjection
from dataset.video_dataset import VideoDataset
from accelerate import Accelerator
import numpy as np
import argparse
from src import get_rotating_demo, seperate_into_patches, batch_interpolation, patch_to_equi, StableVideoDiffusionPipelineCustom
import math
import subprocess
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

def infer_model(video_path, num_frames_generation, guidance_scale, num_inference_steps, visualization_fov_x, visualization_fov_y,
				predict_camera_motion, super_resolution, fov_x, calibration_model,
				progress=gr.Progress(track_tqdm=True)):

	global args, accelerator, pipeline, weight_dtype

	if num_frames_generation != args.num_frames:
		args.num_frames_batch = args.num_frames
		args.num_frames = num_frames_generation
		args.blend_frames = 5

	# Load dataset
	yield 'Processing video...', None, None, None, None
	val_dataset = VideoDataset(video_path, sample_frames=args.num_frames,
							   fixed_start_frame=args.fixed_start_frame,
							   calibration_img_size=args.calibration_img_size,
							   frame_rate=args.frame_rate,
							   )
	
	val_save_dir = os.path.abspath(os.path.join(args.unet_path, args.val_save_folder))
	os.makedirs(val_save_dir, exist_ok=True)

	batch = val_dataset[0]
	video = batch["video"].to(accelerator.device, non_blocking=True)
	path = batch['path']
	frame_rate = batch['frame_rate']
	ext = '.'+path.split('.')[-1]

	out_file_dir = os.path.join(val_save_dir, os.path.basename(path).replace(ext, ''))
	os.makedirs(out_file_dir, exist_ok=True)

	if predict_camera_motion:

		if calibration_model == "MASt3R":
			from mast3r.model import AsymmetricMASt3R
			mast3r_model = AsymmetricMASt3R.from_pretrained(args.calibration_model_path).to(accelerator.device).eval()

			yield 'Predicting camera poses with MASt3R...', None, None, None, None
			poses, intrinsics, resized_width = get_camera_params_from_frames(video, shared_intrinsics=True, 
								img_size=args.calibration_img_size,
								scene_graph='swin-3-noncyclic', # swin-2-noncyclic, logwin-2-noncyclic, complete
								model=mast3r_model) # (T, 4, 4), (T, 3, 3), but intrinsics are the same for all frames

			focal_length = intrinsics[0, 0, 0].cpu().item() # focal length in pixels
			fov_x = torch.tensor(focal2fov(focal_length, resized_width), dtype=torch.float32).item()
			mast3r_model = mast3r_model.to('cpu')
			poses = poses.to(accelerator.device, non_blocking=True)

		elif calibration_model == "MegaSaM":
			yield 'Predicting camera poses with MegaSaM...', None, None, None, None
			pose_file_path = os.path.join(out_file_dir, os.path.basename(path).replace(ext, '.npz'))
			if not os.path.exists(pose_file_path):
				# go to subrepo mega-sam and run the script run_trajectory_predict.py
				os.chdir(os.path.join(os.path.dirname(__file__), 'mega-sam'))
				os.makedirs(os.path.join(out_file_dir, 'images'), exist_ok=True)
				subprocess.run(['python', 'run_trajectory_predict.py', video_path, out_file_dir, 
								'--temp_image_dir', os.path.join(out_file_dir, 'images'),
								'--camera_pose_env', 'mega_sam',
								'--mono_depth_env', 'mega_sam',
								'--fps', str(frame_rate), '--num_frames', str(num_frames_generation)])
				os.chdir(os.path.dirname(__file__))
				print(f"out_file_dir: {out_file_dir}", video_path)
			pose_data = np.load(pose_file_path)
			poses = torch.from_numpy(pose_data['cam_c2w']).to(accelerator.device, non_blocking=True)
			intrinsics = pose_data['intrinsic']
			focal = abs(intrinsics[0, 0])
			cx, cy = intrinsics[0, 2], intrinsics[1, 2]
			fov_x = torch.tensor(np.degrees(2 * np.arctan2(cx, focal)), dtype=torch.float32).item()

		'''
		The calibrated pose are camera-to-world, with z forward, x right, y up convention. (right-handed)
		We use z top, x forward, y right convention for camera poses. (also right-handed)
		'''

		convention_rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
		convention_inverse = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

		rolls, pitches, yaws = np.zeros(len(poses)), np.zeros(len(poses)), np.zeros(len(poses))
		R1 = poses[0, :3, :3].cpu().numpy()

		for i in range(1, len(poses)):
			R2 = poses[i, :3, :3].cpu().numpy()
			roll, pitch, yaw = rotation_matrix_to_euler(convention_inverse @ R2.T @ R1 @ convention_rotation, z_down=True) # rotation matrix are camera-to-world, cam1 --> cam2 is R2.T @ R1
			rolls[i] = -roll
			pitches[i] = pitch
			yaws[i] = yaw
	
	else:
		rolls, pitches, yaws = np.zeros(len(video)), np.zeros(len(video)), np.zeros(len(video))

	out_file_path = os.path.join(out_file_dir, '360video.mp4')

	# Sample the video with the SVD method
	yield 'Generating 360 video...', None, None, None, None
	pipeline = pipeline.to(accelerator.device)
	output_path = sample_svd(args, accelerator, pipeline, weight_dtype,
								out_file_path=out_file_path,
								fov_x=fov_x,
								roll=rolls, pitch=pitches, yaw=yaws,
								conditional_video=video,
								fps=frame_rate,
								num_inference_steps=num_inference_steps,
								width=1024, height=512,
								guidance_scale=guidance_scale,
								decode_chunk_size=10,
								inference_final_rotation=args.inference_final_rotation,
								blend_decoding_ratio=args.blend_decoding_ratio,
								extended_decoding=args.extended_decoding,
								rotation_during_inference=args.rotation_during_inference,
								equirectangular_input=False,
								)
	pipeline = pipeline.to('cpu')
	torch.cuda.empty_cache()

	print(f"output_path: {output_path}")

	if super_resolution:
		yield "Enhancing video...", None, None, None, None
		# cd to the VEnhancer directory, then run the script ./run_VEnhancer_one_video.sh
		venhancer_dir = os.path.join(os.path.dirname(__file__), 'venhancer')
		os.chdir(venhancer_dir)
		cmd = ['./run_VEnhancer_one_video.sh', output_path, out_file_dir]
		if args.venhancer_env is not None:
			cmd = ['conda', 'run', '-n', args.venhancer_env] + cmd
		subprocess.run(cmd, check=True, text=True, capture_output=False)
		os.chdir(os.path.dirname(__file__))
		output_path = output_path.replace('.mp4', '_enhanced.mp4')
	
	# Generate perspective videos
	yield "Generating perspective videos...", None, None, None, None
	width = 384 if not super_resolution else 768
	height = int(width / math.atan(math.radians(visualization_fov_x / 2)) * math.atan(math.radians(visualization_fov_y / 2)))

	out_clockwise = os.path.join(out_file_dir, '360video_clockwise.mp4')
	out_counter_clockwise = os.path.join(out_file_dir, '360video_counter_clockwise.mp4')
	out_front = os.path.join(out_file_dir, '360video_front.mp4')
	out_visualization_paths = [out_clockwise, out_counter_clockwise, out_front]
	rotation_angles =  [360, -360, 0] 

	get_rotating_demo(output_path, out_visualization_paths, visualization_fov_x, rotation_angles,
				   		device=accelerator.device, width=width, height=height)
	
	yield "Finished!", output_path, out_front, out_clockwise, out_counter_clockwise  # Return the path to the generated video

	# copy the video to the output folder
	os.system(f'cp {video_path} {os.path.join(out_file_dir, "input_persepective"+ext)}')

	torch.cuda.empty_cache()

	print(f"Finished! Output video: {output_path}")


def enable_button(video_input):
	if video_input is not None:
		return gr.Button(interactive=True)
	else:
		return gr.Button(interactive=False)

args = argparse.Namespace(
	val_save_folder='gradio_output',
	unet_path='./checkpoints/pretrained-weights',
	pretrained_model_name_or_path='stabilityai/stable-video-diffusion-img2vid',
	calibration_model_path='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
	calibration_img_size=512,
	frame_rate=5.0,
	fixed_start_frame=True,
	blend_decoding_ratio=16,
	rotation_during_inference=True,
	inference_final_rotation=0,
	extended_decoding=True,
	num_frames=25,
	camera_pose_env='mega_sam',
	mono_depth_env='mega_sam',
	venhancer_env='venhancer',
)

accelerator = Accelerator(mixed_precision='no')
weight_dtype = torch.float32

unet = UNetSpatioTemporalConditionModel.from_pretrained(args.unet_path, subfolder="unet")
feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder").to(weight_dtype)

pipeline = StableVideoDiffusionPipelineCustom.from_pretrained(
			args.pretrained_model_name_or_path,
			unet=accelerator.unwrap_model(unet),
			feature_extractor=feature_extractor,
			image_encoder=image_encoder,
			revision=None,
			torch_dtype=weight_dtype,
			)

with gr.Blocks() as demo:

	title = ("""<h1 style="font-size: 2.5em; text-align: center; color: #333;">Video to 360 Video Generation</h1>""")
	gr.HTML(title)

	with gr.Row():
		with gr.Column():
			video_input = gr.Video(label="Input Video")

			# Generation parameters
			with gr.Group():
				gr.HTML("<div style='font-size:18px; font-weight:bold; margin-bottom:0.5em;'>Generation Parameters</div>")
				num_frames = gr.Dropdown(choices=[25, 45, 65], value=25, label="Number of Frames, select among 25 (1 round), 45 (2 rounds), 65 (3 rounds)")
				guidance_scale = gr.Slider(2.5, 10.0, 3, step=0.1, label="Guidance Scale")
				num_inference_steps = gr.Slider(1, 100, 25, step=1, label="Number of Inference Steps")
				super_resolution = gr.Checkbox(label="Super Resolution", value=False)

			# Calibration parameters
			with gr.Group():
				gr.HTML("<div style='font-size:18px; font-weight:bold; margin-bottom:0.5em;'>Calibration Parameters</div>")
				predict_camera_motion = gr.Checkbox(label="Predict Camera Motion. If disabled, the input video will be centered in the output 360 video.", value=True)
				calibration_model = gr.Dropdown(choices=["MASt3R", "MegaSaM"], value="MegaSaM", label="Calibration Model")
				fov_x = gr.Slider(30, 120, 90, step=1, label="Field of view for the input video, only used when predicting camera motion is disabled")

			# Visualization parameters
			with gr.Group():
				gr.HTML("<div style='font-size:18px; font-weight:bold; margin-bottom:0.5em;'>Visualization Parameters</div>")
				visualization_fov_x = gr.Slider(30, 120, 90, step=1, label="Width field of view for the perspective video")
				visualization_fov_y = gr.Slider(30, 120, 90, step=1, label="Height field of view for the perspective video")

			# Submit button
			submit_button = gr.Button("Submit", interactive=False)

		with gr.Column():
			progress = gr.Textbox(label="Progress")
			output_video = gr.Video(label="Output Video")
			output_perspective_front = gr.Video(label="Output Video Frontview")
			output_perspective_clockwise = gr.Video(label="Output Video Clockwise Rotation")
			output_perspective_counter_clockwise = gr.Video(label="Output Video Counter Clockwise Rotation")

	video_input.change(enable_button, inputs=[video_input], outputs=[submit_button])

	submit_button.click(infer_model, 
						inputs = [video_input, num_frames, guidance_scale, num_inference_steps, visualization_fov_x, visualization_fov_y,
									predict_camera_motion, super_resolution, fov_x, calibration_model],
						outputs = [progress, output_video, output_perspective_front, output_perspective_clockwise, output_perspective_counter_clockwise])
	# Launch the demo
	demo.queue().launch(share=True)
