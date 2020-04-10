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
    --max 1e-5 \
    --min 1e-5 \
    --network 'cnn' \
    --energy_tradeoff 0.8 \
    --ent_coef 0.05 \
    --place_num 3 \
    --no_delta True \
    --fix_rotation True \
    --log True \
    --debug 'version-0.9.5'
