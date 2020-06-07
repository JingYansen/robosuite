## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 100000 \
    --nsteps 64 \
    --noptepochs 10 \
    --nminibatches 32 \
    --lr_type 'linear' \
    --max 1e-5 \
    --min 1e-5 \
    --camera_height 64 \
    --camera_width 64 \
    --log True \
    --debug 'debug'
