## train with states
set -ex
python bin_squeeze_baselines.py \
    --env_id 'BinSqueezeMulti-v0' \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 3000000 \
    --nsteps 1024 \
    --noptepochs 10 \
    --nminibatches 64 \
    --lr_type 'linear' \
    --max 1e-5 \
    --min 1e-5 \
    --network 'cnn' \
    --ent_coef 0.2 \
    --total_steps 1000 \
    --place_num 5 \
    --camera_type 'image+depth' \
    --fix_rotation True \
    --log True \
    --debug 'version-1.0.2'
