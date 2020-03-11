## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1000000 \
    --nsteps 128 \
    --noptepochs 40 \
    --nminibatches 2 \
    --save_interval 50 \
    --keys 'state' \
    --lr_type 'linear' \
    --max 1e-3 \
    --min 3e-4 \
    --network mlp \
    --num_layers 2 \
    --cliprange 0.2 \
    --ent_coef 0.005 \
    --use_object_obs True \
    --test True \
    --log True \
    --obj_nums '1,1,2,2' \
    --random_take True \
    --debug '6objects'
