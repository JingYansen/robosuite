## train with states

python bin_packing_baselines.py \
    --log False \
    --make_video True \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 0 \
    --network mlp \
    --num_layers 2 \
    --load_path 'results/baselines/states_ppo2_mlp_2layer_0.001lr_256stpes_8async_0.01explore_small_bin_not_norm/checkpoints/00300' \
    --has_offscreen_renderer True \
    --camera_height 320 \
    --camera_width 240 \
    --render_drop_freq 10 \
    --video_name '2layer_0.01ent_78ksteps_not_norm.mp4' \
    --test True \
    --test_episode 30 \
    --debug 'make_video'
