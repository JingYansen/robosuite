## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 1000000 \
    --nsteps 512 \
    --noptepochs 20 \
    --nminibatches 4 \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --network 'cnn' \
    --log True \
    --debug '3view_6dim_easy_task'
