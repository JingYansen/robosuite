## train with states

python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1000000 \
    --nsteps 256 \
    --save_interval 100 \
    --lr 1e-3 \
    --network mlp \
    --num_layers 2 \
    --ent_coef 0.01 \
    --debug 'test_tensorboard'
