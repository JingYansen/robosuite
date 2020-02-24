## train with states

python bin_packing_baselines.py \
    --log False \
    --make_video True \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 0 \
    --network mlp \
    --num_layers 3 \
    --load_path 'results/baselines/states_ppo2_mlp_3layer_0.001lr_256stpes_8async_small_bin_not_norm/model.pth' \
    --has_offscreen_renderer True \
    --camera_height 320 \
    --camera_width 240 \
    --render_drop_freq 10 \
    --video_name 'not_norm.mp4' \
    --debug 'make_video'

