## train with states
set -ex
python bin_packing_baselines2.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 4000000 \
    --nsteps 256 \
    --noptepochs 10 \
    --nminibatches 32 \
    --lr_type 'linear' \
    --max 1e-4 \
    --min 1e-4 \
    --take_nums 8 \
    --ent_coef 0.005 \
    --camera_height 64 \
    --camera_width 64 \
    --log True \
    --test True \
    --use_typeVector True \
    --debug 'version-1.2.0'
