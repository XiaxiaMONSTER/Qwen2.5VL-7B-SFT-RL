MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
NCCL_ALGO=Tree \
swift sft \
    --model /mnt/workspace/xiayuxuan/Qwen_TF/Qwen2.5-VL-7B-Instruct \
    --dataset /mnt/workspace/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/data/instructions/marked-smart_resize-cot_template-42.json \
    --split_dataset_ratio 0.1 \
    --train_type full \
    --freeze_vit True \
    --freeze_aligner True \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --temperature 0.9 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir /mnt/workspace/xiayuxuan/Qwen_TF/Project-2_trace_sft_prompt_v2/models/Qwen2.5VL-CoT-Template-Full \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --save_only_model True