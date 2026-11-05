MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
NCCL_ALGO=Tree \
swift rlhf \
    --rlhf_type grpo \
    --model ./models/Qwen2.5VL-CoT_Template/v0-20250720-132016/checkpoint-600 \
    --external_plugins ./scripts/GRPO/reward_function.py \
    --reward_funcs external_format external_bbox external_state external_indication external_countdown \
    --reward_weights 0.05 0.35 0.25 0.175 0.175\
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_vit True \
    --freeze_aligner True \
    --torch_dtype bfloat16 \
    --dataset ./data/instructions/grpo_datasets.json \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir ./models/Qwen2.5VL-CoT-Template-GRPO \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system ./scripts/GRPO/prompt.txt \
    --log_completions True \
    --save_only_model True \
    --beta 0.04 