# We may only be able to run on one GPU due to LLaMA's code (MP supposed to be 1 for 7B model).
export MP=8
export TARGET_FOLDER="/scratch/cnicholas/helm_llama/weights"
export MODEL_SIZE=65B

torchrun --rdzv-backend=c10d --nnodes=1 --rdzv-endpoint=localhost:0 --nproc_per_node $MP helm_llama.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model  --max_seq_len 2048 --max_batch_size $1 --temperature 0 --max_new_tokens $2 --data_id $6 --p_id $6 -k $5 --num_instances 1000 --num_examples 5
