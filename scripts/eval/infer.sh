CUDA_VISIBLE_DEVICES=0,1 \
swift infer \  
    --model ./models/Qwen2.5VL-CoT-Template-GRPO/v1-20250811-172441/checkpoint-700-merged \
    --infer_backend pt \
    --temperature 0 \
    --stream true 
