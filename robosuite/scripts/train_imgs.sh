## train with states

python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1000000 \
    --nsteps 256 \
    --save_interval 50 \
    --lr 1e-3 \
    --network 'cnn' \
    --keys 'image' \
    --camera_name 'targetview' \
    --use_camera_obs True \
    --has_offscreen_renderer True \
    --ent_coef 0.01 \
    --debug 'targetview_only'
