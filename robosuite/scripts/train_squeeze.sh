## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 1000000 \
    --nsteps 1024 \
    --noptepochs 10 \
    --nminibatches 64 \
    --lr_type 'linear' \
    --max 1e-5 \
    --min 1e-5 \
    --network 'cnn' \
    --ent_coef 0.2 \
    --total_steps 200 \
    --place_num 0 \
    --random_target True \
    --fix_rotation True \
    --log True \
    --debug 'version-0.9.9.5'
