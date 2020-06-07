## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 1000000 \
    --nsteps 128 \
    --noptepochs 10 \
    --nminibatches 32 \
    --lr_type 'linear' \
    --max 1e-4 \
    --min 1e-4 \
    --camera_height 64 \
    --camera_width 64 \
    --log True \
    --test True \
    --debug 'version-1.0.0'
