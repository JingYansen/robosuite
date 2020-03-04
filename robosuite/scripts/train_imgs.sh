## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1000000 \
    --nsteps 256 \
    --save_interval 100 \
    --lr 1e-3 \
    --network 'cnn' \
    --keys 'image' \
    --use_camera_obs True \
    --has_offscreen_renderer True \
    --ent_coef 0.01 \
    --camera_height 64 \
    --camera_width 64 \
    --random_take True \
    --debug '2view'
