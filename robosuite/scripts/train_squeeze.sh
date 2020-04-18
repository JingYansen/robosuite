## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 1000000 \
    --nsteps 512 \
    --noptepochs 10 \
    --nminibatches 32 \
    --lr_type 'linear' \
    --max 1e-5 \
    --min 1e-5 \
    --network 'cnn' \
    --energy_tradeoff 0.8 \
    --ent_coef 0.2 \
    --place_num 3 \
    --log True \
    --debug 'version-0.9.7'
