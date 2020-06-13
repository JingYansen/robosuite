set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py \
    --total_epochs 10 \
    --test_interval 2 \
    --batch_size 128