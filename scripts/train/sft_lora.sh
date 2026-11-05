MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /root/autodl-fs/zhangyixin/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct \
    --dataset ./data/instructions/marked-smart_resize-none-42.json \
    --split_dataset_ratio 0.1 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 4 \
    --temperature 0.9 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir ./models/Qwen2.5VL-noCoT-Lora \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01