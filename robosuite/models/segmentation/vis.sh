set -ex
python vis.py \
    --type 1 \
    --total_num 20 \
    --model_path 'results/random_data/checkpoint_1.pth' \
    --data_list_path '/home/yeweirui/data/random' \
    --data_path '/home/yeweirui/' \
    --vis_path 'results/random_data/vis'