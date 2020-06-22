set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py \
    --encoder 'resnet50' \
    --data_list_path '/home/yeweirui/data/random_take_8obj' \
    --data_path '/home/yeweirui/' \
    --ckpt_path 'results/random_take_8obj_1e-6_res50' \
    --vis_path 'results/random_take_8obj_1e-6_res50' \
    --total_epochs 20 \
    --test_interval 2 \
    --batch_size 128 \
    --test_batch 128