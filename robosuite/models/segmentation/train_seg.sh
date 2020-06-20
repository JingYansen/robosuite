set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py \
    --encoder 'resnet34' \
    --data_list_path '/home/yeweirui/data/8obj_smaller_bound' \
    --data_path '/home/yeweirui/' \
    --ckpt_path 'results/8obj_smaller_bound_1e-5' \
    --vis_path 'results/8obj_smaller_bound_1e-5' \
    --total_epochs 20 \
    --test_interval 2 \
    --batch_size 128 \
    --test_batch 128