set -ex
python vis.py \
    --type 0 \
    --total_num 30 \
    --model_path 'results/random_take_8obj/checkpoint_10.pth' \
    --data_list_path '/home/yeweirui/data/random_take_8obj/' \
    --data_path '/home/yeweirui/' \
    --vis_path 'results/random_take_8obj/vis'