## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 16 \
    --num_timesteps 1000000 \
    --nsteps 256 \
    --save_interval 50 \
    --lr 1e-3 \
    --network mlp \
    --num_layers 2 \
    --ent_coef 0.01 \
    --use_object_obs True \
    --test True \
    --debug 'fine_desighed_no_z_rotation'
