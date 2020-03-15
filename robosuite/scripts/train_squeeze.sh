## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 1000000 \
    --nsteps 256 \
    --noptepochs 40 \
    --nminibatches 2 \
    --save_interval 50 \
    --lr_type 'linear' \
    --max 1e-4 \
    --min 1e-4 \
    --ent_coef 0.005 \
    --network 'cnn' \
    --keys 'image' \
    --use_camera_obs True \
    --has_offscreen_renderer True \
    --camera_height 64 \
    --camera_width 64 \
    --test True \
    --log True \
    --random_take True \
    --debug 'squeeze'
