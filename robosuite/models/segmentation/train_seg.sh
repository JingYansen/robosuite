set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py \
    --total_epochs 100 \
    --batch_size 64