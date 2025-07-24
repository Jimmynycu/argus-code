"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
import accelerate
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, DDIMScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from dataset.video_dataset import VideoDataset
from accelerate.utils import DummyOptim, DummyScheduler

from src import rand_log_normal, _resize_with_antialiasing, tensor_to_vae_latent, get_rpy, StableVideoDiffusionPipelineCustom, generate_mask_batch, sample_svd
from equilib import equi2pers

from torch.utils.tensorboard import SummaryWriter
import time

def parse_args():
	parser = argparse.ArgumentParser(
		description="Script to train Stable Video Diffusion."
	)
	parser.add_argument(
		"--pretrained_model_name_or_path", type=str, default=None, required=True,
		help="Path to pretrained model or model identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--revision", type=str, default=None, required=False,
		help="Revision of pretrained model identifier from huggingface.co/models.",
	)

	# data arguments
	parser.add_argument('--train_base_folder', nargs='+',
					 	type=str, required=True, help='Path to the training dataset')
	parser.add_argument('--train_clip_file', type=str, default=None, help='Path to the training clip file')
	parser.add_argument('--wanted_name_in_dataset', nargs='+',
					 	type=str, default=None, help='Wanted names in the dataset')
	parser.add_argument("--val_base_folder", nargs='+',
					 	type=str, required=True, help="Path to the valiadtion dataset.")
	parser.add_argument("--val_clip_file", type=str, default=None, help="Path to the validation clip file.")
	parser.add_argument("--num_frames", type=int, default=25,)
	parser.add_argument("--width", type=int, default=1024,)
	parser.add_argument("--height", type=int, default=512,)
	parser.add_argument('--frame_rate', type=int, default=None, help='Frame rate for the video')
	parser.add_argument('--min_frame_rate', type=int, default=None, help='Minimum frame rate for the video')
	parser.add_argument('--max_frame_rate', type=int, default=None, help='Maximum frame rate for the video')
	parser.add_argument('--fixed_rpy', action='store_true', help='Use fixed rpy for the video')
	parser.add_argument('--center_yaw', action='store_true', help='Center yaw for the video')
	parser.add_argument('--fov_x_min', type=float, default=90., help='Minimum width fov')
	parser.add_argument('--fov_x_max', type=float, default=90., help='Maximum width fov')
	parser.add_argument('--fov_y_min', type=float, default=90., help='Minimum height fov')
	parser.add_argument('--fov_y_max', type=float, default=90., help='Maximum height fov')
	parser.add_argument('--fixed_start_frame', action='store_true', help='for each video, start from the first frame, for debugging')
	parser.add_argument('--train_dataset_size', type=int, default=None, help='Size of the training dataset')

	# inference arguments
	parser.add_argument('--rotation_during_inference', action='store_true', help='Rotate the video during inference')
	parser.add_argument('--inference_final_rotation', type=float, default=0., help='Final rotation during inference.')
	# parser.add_argument('--blend_rotation', type=int, default=None, help='Blend rotation.')
	parser.add_argument('--blend_decoding_ratio', type=int, default=None, help='Blend decoding ratio. typically 2 or 4')
	parser.add_argument('--extended_decoding', action='store_true', help='Use extended decoding.')
	# parser.add_argument('--outpaint', action='store_true', help='Whether to do outpaint')
	
	# training arguments
	parser.add_argument('--rotation', action='store_true', help='Whether to rotate the 360 images during training')
	parser.add_argument("--num_validation_each_step", type=int, default=1,
		help="Number of images that should be generated during validation with `validation_prompt`.",)
	parser.add_argument("--validation_steps", type=int, default=500,
						help=("Run fine-tuning validation every X steps. \
							The validation process consists of running the text/image prompt"),)
	parser.add_argument("--output_dir", type=str, default="./experiments-svd",
						help="The output directory where the model predictions and checkpoints will be written.",)
	parser.add_argument('--equirectangular_weighing', action='store_true', help='Use equirectangular weighing')
	parser.add_argument('--equirectangular_weighing_alpha', type=float, default=0.25, help='Equirectangular weighing alpha')
	parser.add_argument('--noise_mean', type=float, nargs='+', default=[0.,], help='Mean of the gaussian noise added to the masked region of the conditioning video')
	parser.add_argument('--noise_std', type=float, nargs='+', default=[1.,], help='Standard deviation of the gaussian noise added to the masked region of the conditioning video')
	parser.add_argument('--noise_schedule', type=int, nargs='+', default=[50000,], help='Schedule of the gaussian noise added to the masked region of the conditioning video')
	parser.add_argument('--noise_conditioning', action='store_true', help='add gaussian noise to the masked region of the conditioning video')
	parser.add_argument('--noise_conditioning_strength', type=float, default=0.25, help='Strength of the gaussian noise added to the masked region of the conditioning video')
	parser.add_argument('--full_perspective_input_prob', type=float, default=1, help='Probability of using full perspective input')
	parser.add_argument('--num_known_frames', type=int, default=0, help='Number of known frames')
	
	# model arguments
	parser.add_argument('--spatial_only', action='store_true', help='Tune only the spatial part of the model.')
	parser.add_argument('--temporal_only', action='store_true', help='Tune only the temporal part of the model.')

	# logging arguments
	parser.add_argument('--experiment_name', type=str, default='default', required=True, help='Name of the experiment')
	parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
	
	parser.add_argument(
		"--per_gpu_batch_size", type=int, default=1,
		help="Batch size (per device) for the training dataloader.",
	)
	parser.add_argument("--max_train_steps", type=int, required=True, help="Total number of training steps to perform",)
	parser.add_argument(
		"--gradient_accumulation_steps", type=int, default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
		"--gradient_checkpointing", action="store_true",
		help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
	)
	parser.add_argument(
		"--learning_rate", type=float, default=1e-4,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
		"--scale_lr", action="store_true", default=False,
		help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
	)
	parser.add_argument(
		"--lr_scheduler", type=str, default="constant",
		help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
			' "constant", "constant_with_warmup"]'),
	)
	parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
	parser.add_argument(
		"--conditioning_dropout_prob",
		type=float,
		default=0.2,
		help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
	)
	parser.add_argument(
		"--allow_tf32",
		action="store_true",
		help=(
			"Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
			" https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
		),
	)
	parser.add_argument(
		"--num_workers",
		type=int,
		default=8,
		help=(
			"Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
		),
	)
	parser.add_argument(
		"--adam_beta1",
		type=float,
		default=0.9,
		help="The beta1 parameter for the Adam optimizer.",
	)
	parser.add_argument(
		"--adam_beta2",
		type=float,
		default=0.999,
		help="The beta2 parameter for the Adam optimizer.",
	)
	parser.add_argument(
		"--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
	)
	parser.add_argument(
		"--adam_epsilon",
		type=float,
		default=1e-08,
		help="Epsilon value for the Adam optimizer",
	)
	parser.add_argument(
		"--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
	)
	parser.add_argument(
		"--push_to_hub",
		action="store_true",
		help="Whether or not to push the model to the Hub.",
	)
	parser.add_argument(
		"--hub_token",
		type=str,
		default=None,
		help="The token to use to push to the Model Hub.",
	)
	parser.add_argument(
		"--hub_model_id",
		type=str,
		default=None,
		help="The name of the repository to keep in sync with the local `output_dir`.",
	)
	parser.add_argument(
		"--logging_dir",
		type=str,
		default="logs",
		help=(
			"[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
			" *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
		),
	)
	parser.add_argument(
		"--mixed_precision",
		type=str,
		default=None,
		choices=["no", "fp16", "bf16"],
		help=(
			"Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
			" 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
			" flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
		),
	)
	parser.add_argument(
		"--report_to",
		type=str,
		default="tensorboard",
		help=(
			'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
			' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
		),
	)
	parser.add_argument(
		"--local_rank",
		type=int,
		default=-1,
		help="For distributed training: local_rank",
	)
	parser.add_argument("--checkpointing_steps", type=int, default=500,
						help=("Save a checkpoint of the training state every X updates. Resume using `--resume_from_checkpoint`."),)
	parser.add_argument("--latest_checkpointing_steps", type=int, default=250,
						help="Save the latest checkpoint of the training state every X updates.",)
	parser.add_argument(
		"--checkpoints_total_limit",
		type=int,
		default=2,
		help=("Max number of checkpoints to store."),
	)
	parser.add_argument(
		"--resume_from_checkpoint",
		type=str,
		default=None,
		help=(
			"Whether training should be resumed from a previous checkpoint. Use a path saved by"
			' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
		),
	)
	parser.add_argument(
		"--pretrain_unet",
		type=str,
		default=None,
		help="use weight for unet block",
	)

	args = parser.parse_args()
	args.output_dir = os.path.join(args.output_dir, args.experiment_name)

	env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
	if env_local_rank != -1 and env_local_rank != args.local_rank:
		args.local_rank = env_local_rank

	return args

def get_noise_schedule_index(global_step, noise_schedule):
	index = 0
	for index in range(len(noise_schedule)):
		if global_step <= noise_schedule[index]:
			return index
	return len(noise_schedule) - 1

def main():
	args = parse_args()

	logging_dir = os.path.join(args.output_dir, args.logging_dir)
	accelerator_project_config = ProjectConfiguration(
		project_dir=args.output_dir, logging_dir=logging_dir)
	accelerator = Accelerator(
		log_with=args.report_to,
		project_config=accelerator_project_config,
	)

	custom_logging_dir = os.path.join(args.output_dir, 'tensorboard')
	if accelerator.is_main_process:
		os.makedirs(custom_logging_dir, exist_ok=True)
	writer = SummaryWriter(custom_logging_dir) if accelerator.is_main_process else None

	@accelerator.on_main_process
	class Logger(object):
		def __init__(self, log_dir):
			self.log = open(log_dir, "a")

		def info(self, *message):
			stacked_message = " ".join([str(m) for m in message])
			print(stacked_message)
			self.log.write(stacked_message + "\n")
			self.log.flush()

		def info_silent(self, *message):
			stacked_message = " ".join([str(m) for m in message])
			self.log.write(stacked_message + "\n")
			self.log.flush()
			
		def flush(self):
			pass
	
	class DummyLogger(object):
		def info(self, *message):
			pass

		def info_silent(self, *message):
			pass

		def flush(self):
			pass

	logger = Logger(os.path.join(args.output_dir, "log.txt")) if accelerator.is_main_process else DummyLogger()

	if accelerator.is_local_main_process:
		transformers.utils.logging.set_verbosity_warning()
		diffusers.utils.logging.set_verbosity_info()
	else:
		transformers.utils.logging.set_verbosity_error()
		diffusers.utils.logging.set_verbosity_error()

	# set different random seed for each process
	# process_seed = args.seed + accelerator.process_index if args.seed is not None else int(time.time()) + accelerator.process_index
	# set_seed(process_seed)

	if accelerator.is_main_process:
		logger.info(str(args))

	# Load img encoder, tokenizer and models.
	feature_extractor = CLIPImageProcessor.from_pretrained(
		args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
	)
	image_encoder = CLIPVisionModelWithProjection.from_pretrained(
		args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
	)
	vae = AutoencoderKLTemporalDecoder.from_pretrained(
		args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
	kwargs = {'subfolder': 'unet', 'revision': args.revision}
	if args.pretrain_unet is None:
		kwargs['variant'] = 'fp16'
	unet = UNetSpatioTemporalConditionModel.from_pretrained(
		args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
		**kwargs
	)

	# Freeze vae and image_encoder
	vae.requires_grad_(False)
	image_encoder.requires_grad_(False)
	unet.requires_grad_(False)

	# For mixed precision training we cast the text_encoder and vae weights to half-precision
	# as these models are only used for inference, keeping weights in full precision is not required.
	weight_dtype = torch.float32
	if accelerator.mixed_precision == "fp16" or accelerator.state.deepspeed_plugin.deepspeed_config['fp16']['enabled']:
		weight_dtype = torch.float16
	elif accelerator.mixed_precision == "bf16" or accelerator.state.deepspeed_plugin.deepspeed_config['bf16']['enabled']:
		weight_dtype = torch.bfloat16

	# override the config with the one from the model
	args.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config.get("gradient_accumulation_steps", args.gradient_accumulation_steps)
	args.per_gpu_batch_size = accelerator.state.deepspeed_plugin.deepspeed_config.get("train_micro_batch_size_per_gpu", args.per_gpu_batch_size)

	# Move image_encoder and vae to gpu and cast to weight_dtype
	image_encoder.to(accelerator.device, dtype=weight_dtype)
	vae.to(accelerator.device, dtype=weight_dtype)

	if args.gradient_checkpointing:
		unet.enable_gradient_checkpointing()

	# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
	def save_model_hook(models, weights, output_dir):

		for i, model in enumerate(models):
			model.save_pretrained(os.path.join(output_dir, "unet"))

		if weights: # make sure to pop weight so that corresponding model is not saved again
			weights.pop()

	def load_model_hook(models, input_dir):

		for i in range(len(models)):
			# pop models so that they are not loaded again
			model = models.pop()

			# load diffusers style into model
			load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
			model.register_to_config(**load_model.config)

			model.load_state_dict(load_model.state_dict())
			del load_model

	accelerator.register_save_state_pre_hook(save_model_hook)
	accelerator.register_load_state_pre_hook(load_model_hook)

	# Enable TF32 for faster training on Ampere GPUs,
	# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
	if args.allow_tf32:
		torch.backends.cuda.matmul.allow_tf32 = True

	if args.scale_lr:
		args.learning_rate *= math.sqrt(args.gradient_accumulation_steps * args.per_gpu_batch_size * accelerator.num_processes)

	# Initialize the optimizer
	optimizer_cls = (
		torch.optim.AdamW
		if accelerator.state.deepspeed_plugin is None
		or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
		else DummyOptim
	)

	parameters_list = []

	'''Customize the parameters that need to be trained; Downcast the non-learnable parameters to the weight_dtype'''
	for name, param in unet.named_parameters():
		if ('temporal_transformer_block' in name) and (not args.spatial_only):
			param.requires_grad = True
			parameters_list.append(param)
		elif (not 'temporal_transformer_block' in name) and (not args.temporal_only):
			param.requires_grad = True
			parameters_list.append(param)
		else:
			param.requires_grad = False
			param.data = param.data.to(dtype=weight_dtype)


	optimizer = optimizer_cls(
		parameters_list,
		lr=args.learning_rate,
		betas=(args.adam_beta1, args.adam_beta2),
		weight_decay=args.adam_weight_decay,
		eps=args.adam_epsilon,
	)

	# DataLoaders creation:
	train_dataset = VideoDataset(args.train_base_folder, width=args.width, height=args.height, 
							  wanted_name_in_dataset=args.wanted_name_in_dataset,
							  sample_frames=args.num_frames, dataset_size=args.train_dataset_size,
							  rotation=args.rotation,
							  frame_rate=args.frame_rate,
							  fixed_start_frame=args.fixed_start_frame,
							  clip_info_path=args.train_clip_file)	
	val_dataset = VideoDataset(args.val_base_folder, width=args.width, height=args.height,
							   sample_frames=args.num_frames,
							   wanted_name_in_dataset=args.wanted_name_in_dataset,
							   frame_rate=args.frame_rate,
							   fixed_start_frame=args.fixed_start_frame,
							   clip_info_path=args.val_clip_file)

	sampler = RandomSampler(train_dataset)
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		sampler=sampler,
		batch_size=args.per_gpu_batch_size,
		num_workers=args.num_workers,
	)

	if accelerator.state.deepspeed_plugin is None or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config:
		lr_scheduler = get_scheduler(
			args.lr_scheduler,
			optimizer=optimizer,
			num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
			num_training_steps=args.max_train_steps * accelerator.num_processes,
		)
	else:
		lr_scheduler = DummyScheduler(optimizer, 
								total_num_steps=args.max_train_steps * accelerator.num_processes, 
								warmup_num_steps=args.lr_warmup_steps * accelerator.num_processes)

	# Prepare everything with our `accelerator`.
	unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
		unet, optimizer, lr_scheduler, train_dataloader
	)

	# Train!
	total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(
		f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
	logger.info(
		f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
	logger.info(
		f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	logger.info(f"  Total optimization steps = {args.max_train_steps}")
	global_step = 0

	def encode_image(pixel_values):
		# pixel: [-1, 1]
		pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
		# We unnormalize it after resizing.
		pixel_values = (pixel_values + 1.0) / 2.0 # [0, 1] for CLIP input

		# Normalize the image with for CLIP input
		pixel_values = feature_extractor(
			images=pixel_values.to(dtype=torch.float32),
			do_normalize=True,
			do_center_crop=False,
			do_resize=False,
			do_rescale=False,
			return_tensors="pt",
		).pixel_values.to(dtype=weight_dtype)

		pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)
		image_embeddings = image_encoder(pixel_values).image_embeds
		return image_embeddings

	def _get_add_time_ids(
		fps: float | torch.Tensor,
		motion_bucket_id: float | torch.Tensor,
		noise_aug_strength: float | torch.Tensor,
		dtype,
		batch_size=1
	):	
		if type(fps) != torch.Tensor:
			fps = torch.tensor([fps], dtype=dtype, device=accelerator.device)
		if type(motion_bucket_id) != torch.Tensor:
			motion_bucket_id = torch.tensor([motion_bucket_id], dtype=dtype, device=accelerator.device)
		if type(noise_aug_strength) != torch.Tensor:
			noise_aug_strength = torch.tensor([noise_aug_strength], dtype=dtype, device=accelerator.device)

		fps = fps.repeat(batch_size // len(fps))
		motion_bucket_id = motion_bucket_id.repeat(batch_size // len(motion_bucket_id))
		noise_aug_strength = noise_aug_strength.repeat(batch_size // len(noise_aug_strength))

		# stack them together
		add_time_ids = torch.stack([fps, motion_bucket_id, noise_aug_strength], dim=1).to(dtype)

		return add_time_ids

	# Potentially load in the weights and states from a previous save
	if args.resume_from_checkpoint:
		if args.resume_from_checkpoint != "latest":
			path = os.path.basename(args.resume_from_checkpoint)
		else:
			# Get the most recent checkpoint
			dirs = os.listdir(args.output_dir)
			dirs = [d for d in dirs if d.startswith("checkpoint")]
			dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
			path = dirs[-1] if len(dirs) > 0 else None

		if path is None:
			logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
			args.resume_from_checkpoint = None
		else:
			logger.info(f"Resuming from checkpoint {path}")
			accelerator.load_state(os.path.join(args.output_dir, path))
			global_step = int(path.split("-")[-1])
			# global_step = int(torch.load(os.path.join(args.resume_from_checkpoint, "scheduler.bin"))["last_epoch"]) // (accelerator.num_processes * args.gradient_accumulation_steps)

	# Only show the progress bar once on each machine.
	progress_bar = tqdm(range(0, args.max_train_steps), disable=not accelerator.is_main_process)
	progress_bar.set_description("Steps")
	# update for the resumed step
	for _ in range(global_step):
		progress_bar.update(1)

	val_save_dir = os.path.join(args.output_dir, "validation_videos")
	if accelerator.is_main_process:
		os.makedirs(val_save_dir, exist_ok=True)

	if args.equirectangular_weighing: # each row has weight sqrt(1 - h**2), where h is the abs(h/(2*height) - 0.5)
		vae_downsample_factor = 8
		alpha = args.equirectangular_weighing_alpha
		height = args.height // vae_downsample_factor
		abs_heights = [abs(h - (height / 2 - 0.5)) / (height / 2) for h in range(height)]
		weights = torch.tensor([(math.sqrt(1 - h**2) + alpha) for h in abs_heights], dtype=weight_dtype, device=accelerator.device)
		weights = (height * weights / weights.sum()).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # (1, 1, 1, H, 1)

	start_time = time.time()

	while True:
		unet.train()
		for step, batch in enumerate(train_dataloader):
			# first, convert images to latent space.
			video = batch["video"].to(weight_dtype).to(accelerator.device, non_blocking=True) # (B, T, C, H, W), T=number of frames, e.g., 25
			latents = tensor_to_vae_latent(video, vae) # takes input from range [-1, 1]
			frame_rate = batch['frame_rate'] # list of frame rates for each video

			# Sample noise that we'll add to the latents
			noise = torch.randn_like(latents)
			bsz = latents.shape[0]

			'''Add noise'''
			# Sample a random timestep for each image, P_mean=0.7 P_std=1.6
			# index the global step to get the noise schedule, find i such that global_step >= noise_schedule[i] and global_step < noise_schedule[i+1]
			index = get_noise_schedule_index(global_step, args.noise_schedule)
			sigmas = rand_log_normal(shape=[bsz,], loc=args.noise_mean[index], scale=args.noise_std[index]).to(accelerator.device).to(weight_dtype)
			# Add noise to the latents according to the noise magnitude at each timestep
			# (this is the forward diffusion process)
			sigmas = sigmas[:, None, None, None, None]
			noisy_latents = latents + noise * sigmas
			timesteps = torch.tensor([0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device).to(weight_dtype)
			inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

			'''Add conditional information
			concatenate a noise augmented version of the conditioning frame channel-wise to the input of the UNet'''

			'''Get the text embedding for conditioning'''
			# fov in degrees, yaw, pitch, roll in radians
			fov_x = random.uniform(args.fov_x_min, args.fov_x_max) # (1,)
			fov_y_range_min = math.degrees(2 * math.atan(math.tan(math.radians(fov_x) / 2) / 2))
			fov_y_range_max = math.degrees(2 * math.atan(math.tan(math.radians(fov_x) / 2) * 2))
			fov_y = random.uniform(max(fov_y_range_min, args.fov_y_min), min(fov_y_range_max, args.fov_y_max)) # (1,)
			hw_ratio = math.tan(math.radians(fov_y) / 2) / math.tan(math.radians(fov_x) / 2)

			rots = []

			pitch, yaw, roll = np.zeros(args.num_frames, dtype=np.float32), np.zeros(args.num_frames, dtype=np.float32), np.zeros(args.num_frames, dtype=np.float32)
			if not args.fixed_rpy:
				pitch, yaw, roll = get_rpy(fps=frame_rate[0].item(), timesteps=args.num_frames)
				
			yaw = random.uniform(-np.pi, np.pi) + yaw if not args.center_yaw else yaw

			mask = generate_mask_batch(fov_x, hw_ratio=hw_ratio,
									roll=roll, pitch=pitch, yaw=yaw, 
									height=args.height, width=args.width, device=accelerator.device) # (T, 1, H, W)
			rots.extend([{ 'yaw': yaw[i], 'pitch': pitch[i], 'roll': roll[i]} for i in range(args.num_frames)])

			rots = rots * bsz
			mask = mask.unsqueeze(0).repeat(bsz, 1, 1, 1, 1) # (B, T, 1, H, W)
			
			# generate conditional videos
			_, T, C, H, W = video.shape
			with torch.no_grad(): 
				conditional_image = equi2pers(video.view(bsz * T, C, H, W).to(torch.float32), fov_x=fov_x, 
								  				rots=rots, z_down=True, width=480, height=int(480*hw_ratio)).to(weight_dtype) # (B*T, C, H, W)

				cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(accelerator.device).to(weight_dtype)
				masked_conditional_video = video + torch.randn_like(video) * cond_sigmas.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
				
				masked_conditional_video = torch.where(mask == 1, masked_conditional_video, 
							torch.randn_like(video, device=accelerator.device) * args.noise_conditioning_strength \
							if args.noise_conditioning else -torch.ones_like(video, device=accelerator.device)) 
							# masked out regions are -1 for black, if noise_conditioning is True, add noise to the masked region

				if random.random() > args.full_perspective_input_prob:
					masked_conditional_video[:, :args.num_known_frames] = video[:, :args.num_known_frames]

				noisy_conditional_latents = tensor_to_vae_latent(masked_conditional_video, vae)
				noisy_conditional_latents = noisy_conditional_latents * (1 / vae.config.scaling_factor)

				encoder_hidden_states = encode_image(conditional_image)
				encoder_hidden_states = encoder_hidden_states.unsqueeze(1).view(bsz, T, -1)

			added_time_ids = _get_add_time_ids(
				frame_rate - 1, # fixed frames per second
				127, # fixed motion_bucket_id = 127 (high id means high motion)
				cond_sigmas,
				latents.dtype,
				bsz,
			)
			added_time_ids = added_time_ids.to(latents.device)

			# Conditioning dropout to support classifier-free guidance during inference. For more details
			# check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
			if args.conditioning_dropout_prob is not None:

				rand_val = torch.rand((bsz), device=accelerator.device)
				cfg_mask = rand_val > args.conditioning_dropout_prob
				
				# 1. Apply masks for the the cross-attention conditioning latents
				null_conditioning = torch.zeros_like(encoder_hidden_states)
				encoder_hidden_states = torch.where(cfg_mask.unsqueeze(1).unsqueeze(1), encoder_hidden_states, null_conditioning)

				# 2. Apply masks for the concatenated conditioning latents
				noisy_conditional_latents = noisy_conditional_latents * (cfg_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1))

			inp_noisy_latents = torch.cat([inp_noisy_latents, noisy_conditional_latents], dim=2)				

			# check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
			target = latents
			model_pred = unet(inp_noisy_latents, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample

			# Denoise the latents
			c_out = -sigmas / ((sigmas**2 + 1)**0.5)
			c_skip = 1 / (sigmas**2 + 1)
			denoised_latents = model_pred * c_out + c_skip * noisy_latents
			weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

			# MSE loss
			if args.equirectangular_weighing:
				loss = (weighing.float() * weights * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1).mean(dim=1)
			else:
				loss = (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1).mean(dim=1) # (B,)
			loss = loss.mean()

			# Gather the losses across all processes for logging (if we use distributed training).
			avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean() # gather for logging

			# Backpropagate
			accelerator.backward(loss)
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

			# Checks if the accelerator has performed an optimization step behind the scenes
			progress_bar.update(1)
			global_step += 1
			time_elapsed = time.time() - start_time
			logger.info_silent(f"Step {global_step} - Loss: {avg_loss} - Time: {time_elapsed}")

			if accelerator.is_main_process:
				writer.add_scalar("Loss/train", avg_loss, global_step)
				writer.add_scalar("Learning_rate", lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, '_last_lr')
					  						 									else args.learning_rate, global_step)
				writer.add_scalar("noise/mean", args.noise_mean[index], global_step)
				writer.add_scalar("noise/std", args.noise_std[index], global_step)
				writer.flush()

			# save checkpoints, no need to be in the main process for deepspeed
			if global_step % args.checkpointing_steps == 0:
				if accelerator.is_main_process: # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
					if args.checkpoints_total_limit is not None:
						checkpoints = os.listdir(args.output_dir)
						checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
						checkpoints = list(filter(lambda x: x != "checkpoint-latest" and x != "checkpoint-archived", checkpoints))
						checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

						# before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
						if len(checkpoints) >= args.checkpoints_total_limit:
							num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
							removing_checkpoints = checkpoints[0:num_to_remove]

							logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
							logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

							for removing_checkpoint in removing_checkpoints:
								removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
								shutil.rmtree(removing_checkpoint)

				save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
				accelerator.save_state(save_path)
				logger.info(f"Saved state to {save_path}")

			accelerator.wait_for_everyone()

			if accelerator.is_main_process and ((global_step // args.gradient_accumulation_steps) % args.validation_steps) == 1:
				logger.info("Validation step at step %d", global_step)

				# randomly take args.num_validation_images from the validation dataset and generate videos
				indices = random.sample(range(len(val_dataset)), args.num_validation_each_step)

				for idx in indices:
					batch = val_dataset[idx] # torch.Tensor, (C, H, W), in range [0, 1]
					frame_rate = batch['frame_rate'] # int
					video = batch["video"].to(weight_dtype).to(accelerator.device, non_blocking=True) # (T, C, H, W)
					path = batch['path']
					video_id = path.split('/')[-2]
					video_idx = path.split('/')[-1].split('.')[0]

					fov_x = random.uniform(args.fov_x_min, args.fov_x_max) # (1,)
					fov_y_range_min = math.degrees(2 * math.atan(math.tan(math.radians(fov_x) / 2) / 2))
					fov_y_range_max = math.degrees(2 * math.atan(math.tan(math.radians(fov_x) / 2) * 2))
					fov_y = random.uniform(max(fov_y_range_min, args.fov_y_min), min(fov_y_range_max, args.fov_y_max)) # (1,)
					hw_ratio = math.tan(math.radians(fov_y) / 2) / math.tan(math.radians(fov_x) / 2)
					
					# create pipeline
					pipeline = StableVideoDiffusionPipelineCustom.from_pretrained(
						args.pretrained_model_name_or_path,
						unet=accelerator.unwrap_model(unet),
						image_encoder=accelerator.unwrap_model(image_encoder),
						vae=accelerator.unwrap_model(vae),
						revision=args.revision,
						torch_dtype=weight_dtype,
					).to(accelerator.device)

					pitch, yaw, roll = get_rpy(fps=frame_rate, timesteps=args.num_frames) \
						if not args.fixed_rpy else \
						(np.zeros(args.num_frames, dtype=np.float32), np.zeros(args.num_frames, dtype=np.float32), np.zeros(args.num_frames, dtype=np.float32))

					sample_svd(args, accelerator, pipeline, weight_dtype, 
								out_file_path=os.path.join(val_save_dir, f"step_{global_step}_{video_id}_{video_idx}.mp4"),
								fov_x=fov_x,
								roll=roll, pitch=pitch, yaw=yaw,
								conditional_video=video,
								fps = frame_rate,
								hw_ratio=hw_ratio,
								inference_final_rotation=args.inference_final_rotation,
								blend_decoding_ratio=args.blend_decoding_ratio,
								extended_decoding=args.extended_decoding,
								noise_conditioning=args.noise_conditioning,
								rotation_during_inference=args.rotation_during_inference,
								equirectangular_input=True,
								num_known_frames=args.num_known_frames if random.random() > args.full_perspective_input_prob else 0,
								)

			start_time = time.time()

			logs = {"loss": loss.detach().item(), 
		   			'lr': lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, '_last_lr') else args.learning_rate}
		
			progress_bar.set_postfix(**logs)


			if global_step >= args.max_train_steps:
				break

		if global_step >= args.max_train_steps:
			break

	# Create the pipeline using the trained modules and save it.
	accelerator.wait_for_everyone()

	accelerator.end_training()


if __name__ == "__main__":
	main()
