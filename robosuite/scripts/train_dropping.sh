## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 1000000 \
    --nsteps 256 \
    --noptepochs 10 \
    --nminibatches 16 \
    --lr_type 'linear' \
    --max 1e-5 \
    --min 1e-5 \
    --camera_height 64 \
    --camera_width 64 \
    --log True \
    --debug 'version-2.0.0'
