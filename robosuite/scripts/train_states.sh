## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1200000 \
    --nsteps 128 \
    --noptepochs 20 \
    --nminibatches 4 \
    --save_interval 50 \
    --lr_type 'linear' \
    --max 7e-4 \
    --min 7e-4 \
    --network mlp \
    --num_layers 2 \
    --cliprange 0.2 \
    --ent_coef 0.005 \
    --use_object_obs True \
    --test True \
    --log True \
    --debug 'baseline'
