set -ex
python vis.py \
    --type 0 \
    --total_num 30 \
    --model_path 'results/8obj_half_in_1m_1e-3_res34/checkpoint_2.pth' \
    --data_list_path '/home/yeweirui/data/8obj_half_in_1m/' \
    --data_path '/home/yeweirui/' \
    --vis_path 'results/8obj_half_in_1m_1e-3_res34/vis'