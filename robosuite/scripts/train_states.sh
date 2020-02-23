## train with states

python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1000000 \
    --nsteps 64 \
    --save_interval 100 \
    --lr 1e-3 \
    --network mlp \
    --num_layers 3 \
    --debug 'small_bin'
