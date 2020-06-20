set -ex
python vis.py \
    --type 0 \
    --total_num 30 \
    --model_path 'results/8obj_smaller_bound_1e-4/checkpoint_20.pth' \
    --data_list_path '/home/yeweirui/data/8obj_smaller_bound/' \
    --data_path '/home/yeweirui/' \
    --vis_path 'results/8obj_smaller_bound_1e-4/vis'