#!/bin/bash

unet_path=$1
val_folder_or_video_path=$2
val_save_folder=$3
guidance_scale=${4:-3}
num_inference_steps=${5:-25}

video_list=()
# if val_folder_or_video_path is a folder, then process all videos in the folder
if [ -d "$val_folder_or_video_path" ]; then
    for video_path in "$val_folder_or_video_path"/*; do
        video_list+=("$video_path")
    done
else
    video_list+=("$val_folder_or_video_path")
fi

echo "Processing ${#video_list[@]} videos"
accelerate launch --num_processes 1 inference.py \
    --val_base_folder ${video_list} \
    --val_save_folder ${val_save_folder} \
    --unet_path $unet_path \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid \
    --decode_chunk_size 10 \
    --noise_aug_strength 0.02 \
    --guidance_scale $guidance_scale \
    --frame_rate 5 \
    --height 512 --width 1024 \
    --fixed_start_frame \
    --num_frames 25 \
    --num_inference_steps $num_inference_steps \
    --inference_final_rotation 0 \
    --rotation_during_inference \
    --extended_decoding \
    --predict_camera_motion \
    --blend_decoding_ratio 16 

    