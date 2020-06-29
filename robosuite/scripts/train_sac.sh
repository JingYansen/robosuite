## train with states
set -ex
python bin_packing_baselines.py \
    --alg sac \
    --num_env 1 \
    --num_timesteps 100000 \
    --nsteps 1 \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --take_nums 6 \
    --camera_height 64 \
    --camera_width 64 \
    --log True \
    --debug 'version-1.0.0'
