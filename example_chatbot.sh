#BSUB -N
#BSUB -o logs/chatbot.%J
#BSUB -q gpu-compute
#BSUB -n 1
#BSUB -gpu "num=1:gmodel=NVIDIAA10080GBPCIe:gmem=75"
#BSUB -R "rusage[mem=25]"

export NUM_GPUS=1
export MP=1
export TARGET_FOLDER="/storage1/chenguangwang/Active/weights"
export MODEL_SIZE=7B #33B

torchrun --nproc_per_node $MP example_chatbot.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model --temperature 0.7

