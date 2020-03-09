## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1200000 \
    --nsteps 256 \
    --noptepochs 10 \
    --nminibatches 8 \
    --save_interval 50 \
    --lr 3e-4 \
    --network mlp \
    --num_layers 2 \
    --cliprange 0.2 \
    --ent_coef 0.005 \
    --use_object_obs True \
    --test True \
    --log True \
    --debug 'type_obs'
