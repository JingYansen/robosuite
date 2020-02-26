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
    --camera_height 128 \
    --camera_width 128 \
    --make_video True \
    --render_drop_freq 10 \
    --video_name '2view_128x128.mp4' \
    --debug '2view'
