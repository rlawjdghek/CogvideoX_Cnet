#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU_IDX="2,3"
NGPU=$(echo $GPU_IDX | tr -cd ',' | wc -c)
BS=4
HEIGHT=384
WIDTH=576
MAX_NUM_FRAMES=16
SAVE_NAME="cnet_ablation_F${MAX_NUM_FRAMES}_H${HEIGHT}_W${WIDTH}_BS${BS}_GPU${NGPU}"

CURRENT_TIMESTAMP=$(TZ='Asia/Seoul' date +%Y%m%d)
CUDA_VISIBLE_DEVICES=$GPU_IDX nohup accelerate launch --config_file ./configs/accelerate_config_machine_single.yaml --main_process_port 1315 --num_processes ${NGPU} \
  train_controlnet.py \
  --train_batch_size ${BS} \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --max_num_frames ${MAX_NUM_FRAMES} \
  --validation_steps 1000 \
  --checkpointing_steps 1000 \
  --data_cfg_path "./configs/kubric.yaml" \
  --tracker_name "cogvideox-controlnet" \
  --pretrained_model_name_or_path "THUDM/CogVideoX-2b" \
  --validation_prompt "car is going in the ocean, beautiful waves::: " \
  --validation_video "../resources/car.mp4:::../resources/ship.mp4" \
  --validation_prompt_separator ::: \
  --num_inference_steps 28 \
  --num_validation_videos 1 \
  --fps 8 \
  --stride_min 1 \
  --stride_max 3 \
  --hflip_p 0.5 \
  --controlnet_weights 0.5 \
  --init_from_transformer \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 3 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 250 \
  --save_name ${SAVE_NAME} \
  --allow_tf32 &> "./exp_nohup/${CURRENT_TIMESTAMP}_${SAVE_NAME}.txt" &
  # --enable_slicing \
  # --enable_tiling \
  # --gradient_accumulation_steps 1 \
  # --optimizer AdamW \
  # --adam_beta1 0.9 \
  # --adam_beta2 0.95 \
  # --gradient_checkpointing \
  # --controlnet_type "canny" \
  # --validation_prompt "car is going in the ocean, beautiful waves:::ship in the vulcano" \
  # --csv_path "set-path-to-csv-file" \
  # --dataloader_num_workers 0 \
  # --video_root_dir "set-path-to-video-directory" \
  # --report_to wandb
  # --pretrained_controlnet_path "cogvideox-controlnet-2b/checkpoint-2000.pt" \