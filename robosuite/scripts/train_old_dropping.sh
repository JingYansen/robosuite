## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 1000000 \
    --nsteps 256 \
    --noptepochs 20 \
    --nminibatches 4 \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --ent_coef 0.005 \
    --camera_height 64 \
    --camera_width 64 \
    --log True \
    --test True \
    --use_typeVector True \
    --debug 'version-1.0.3'
