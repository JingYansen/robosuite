## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1000000 \
    --nsteps 1024 \
    --noptepochs 20 \
    --nminibatches 4 \
    --lr_type 'linear' \
    --max 1e-3 \
    --min 1e-3 \
    --network 'cnn' \
    --log True \
    --debug 'zw_only'
