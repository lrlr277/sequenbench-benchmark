export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=6
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /mnt/beegfs/xr/models/vl/Qwen2.5-VL-7B-Instruct \
    --template qwen2_vl \
    --finetuning_type lora \
    --dataset_dir /mnt/beegfs/xr/lm_multimodal/seq/data/split_712 \
    --dataset train \
    --eval_dataset val \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 20 \
    --output_dir /mnt/beegfs/xr/lm_multimodal/seq/finetune/lora/qwen_7b \
    --plot_loss True \
    --trust_remote_code True \
    --optim adamw_torch \
