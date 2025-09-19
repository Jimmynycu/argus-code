#!/bin/bash
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=1024G
#SBATCH --gres=gpu:a6000:8
#SBATCH --time=72:00:00

# setting the job name and output file
#SBATCH --job-name="YOUR_JOB_NAME"
#SBATCH --output=logs/%j.out

main_process_port=9905
main_process_ip=localhost
num_process_per_node=8
num_machines=1
machine_rank=0
num_processes=$(($num_process_per_node * $num_machines))
gradient_accumulation_steps=1
per_gpu_batch_size=1
output_dir='experiments'

experiment_name=${1:-stage1}
height=${2:-384}
num_train_steps=${3:-100000}
width=$(($height * 2))
pretrain_unet=${3:-None}
resume_from_checkpoint=${4:-None}

TRAIN_DATASET_PATH=PATH_TO_TRAIN_DATASET
TRAIN_CLIP_FILE_PATH=PATH_TO_TRAIN_CLIP_FILE
VAL_DATASET_PATH=PATH_TO_VAL_DATASET
VAL_CLIP_FILE_PATH=PATH_TO_VAL_CLIP_FILE

num_frames=25
learning_rate=1e-5


if [ $machine_rank -eq 0 ]; then
        mkdir -p ./$output_dir/$experiment_name
        # the zero2 base config is at './scripts/deepspeed_config/deepspeed_zero2.yaml', modify the graident_accumulation_steps and the output_path for tensorboard logs
        cp ./scripts/deepspeed_config/zero2_offload.json ./$output_dir/$experiment_name/deepspeed_config.json
        sed -i "s/\"gradient_accumulation_steps\": 1/\"gradient_accumulation_steps\": $gradient_accumulation_steps/g" ./$output_dir/$experiment_name/deepspeed_config.json
        sed -i "/\"output_path\"/c\\        \"output_path\": \".\/$output_dir\/$experiment_name\"," ./$output_dir/$experiment_name/deepspeed_config.json
        sed -i "/\"train_micro_batch_size_per_gpu\"/c\\    \"train_micro_batch_size_per_gpu\": $per_gpu_batch_size," ./$output_dir/$experiment_name/deepspeed_config.json
fi

# create an accelerator config file, the base config file is at './scripts/accelerate_config/accelerate_zero2.yaml', modify the num_processes and save it 
cp ./scripts/accelerate_config/accelerate_config.yaml ./$output_dir/$experiment_name/accelerate_config_$machine_rank.yaml
sed -i "/num_processes/c\num_processes: $num_processes" ./$output_dir/$experiment_name/accelerate_config_$machine_rank.yaml
sed -i "/num_machines/c\num_machines: $num_machines" ./$output_dir/$experiment_name/accelerate_config_$machine_rank.yaml
sed -i "/main_process_port/c\main_process_port: $main_process_port" ./$output_dir/$experiment_name/accelerate_config_$machine_rank.yaml
sed -i "/machine_rank/c\machine_rank: $machine_rank" ./$output_dir/$experiment_name/accelerate_config_$machine_rank.yaml
sed -i "/main_process_ip/c\main_process_ip: $main_process_ip" ./$output_dir/$experiment_name/accelerate_config_$machine_rank.yaml
sed -i "/deepspeed_config_file/c\\ deepspeed_config_file: .\/$output_dir\/$experiment_name\/deepspeed_config.json" ./$output_dir/$experiment_name/accelerate_config_$machine_rank.yaml

cmd="accelerate launch --config_file ./$output_dir/$experiment_name/accelerate_config_$machine_rank.yaml train.py \
    --pretrained_model_name_or_path='stabilityai/stable-video-diffusion-img2vid' \
    --train_base_folder $TRAIN_DATASET_PATH \
    --train_clip_file $TRAIN_CLIP_FILE_PATH \
    --val_base_folder $VAL_DATASET_PATH \
    --val_clip_file $VAL_CLIP_FILE_PATH \
    --frame_rate 5 \
    --fov_x_min 30 --fov_x_max 120 --fov_y_min 30 --fov_y_max 120 \
    --blend_decoding_ratio 4 \
    --noise_mean 0 0.5 1 --noise_std 1 1 1 --noise_schedule 5000 10000 15000 \
    --equirectangular_weighing --equirectangular_weighing_alpha 0.25 \
    --rotation --center_yaw \
    --gradient_checkpointing \
    --max_train_steps=$num_train_steps \
    --output_dir $output_dir --experiment_name=$experiment_name \
    --per_gpu_batch_size=$per_gpu_batch_size --gradient_accumulation_steps=$gradient_accumulation_steps \
    --height=$height --width=$width --num_frames $num_frames \
    --checkpointing_steps 500 --checkpoints_total_limit 4 \
    --learning_rate=$learning_rate --lr_warmup_steps=0 \
    --extended_decoding \
    --mixed_precision fp16 \
    --validation_steps 250"

if [ "$pretrain_unet" != "None" ]; then
    cmd="$cmd --pretrain_unet $pretrain_unet"
fi

if [ "$resume_from_checkpoint" != "None" ]; then
    cmd="$cmd --resume_from_checkpoint $resume_from_checkpoint"
fi

echo $cmd

eval $cmd
