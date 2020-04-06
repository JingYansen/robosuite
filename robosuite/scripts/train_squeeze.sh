## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 1200000 \
    --nsteps 1024 \
    --noptepochs 20 \
    --nminibatches 4 \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --network 'cnn' \
    --step_size 0.002 \
    --place_num 3 \
    --camera_depth True \
    --log True \
    --debug 'version-0.9'
