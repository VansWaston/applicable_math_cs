INDX=2
STRIDE=8
PERCENTAGE=0.01
GPU_DEVICES=4

# conda activate cu90 | sh screen_run.sh 7



if [ "$#" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(($INDX % $GPU_DEVICES))
    python cs_3_1_repo.py --job_index $INDX --job_stride $STRIDE --percentage $PERCENTAGE
    exit 1
elif [ "$#" -eq 3 ]; then
    export CUDA_VISIBLE_DEVICES=$(($1 % $GPU_DEVICES))
    python cs_3_1_repo.py --job_index $1 --job_stride $2 --percentage $3
elif [ "$#" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=$(($1 % $GPU_DEVICES))
    python cs_3_1_repo.py --job_index $1 --job_stride $STRIDE --percentage $PERCENTAGE
else
    echo "Usage: $0 [job_index] [job_stride] [percentage]"
    exit 1
fi
