set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 1200000 \
    --nsteps 128 \
    --noptepochs 20 \
    --nminibatches 4 \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --network 'cnn' \
    --step_size 0.002 \
    --place_num 3 \
    --fix_rotation True \
    --log True \
    --debug 'debug'
