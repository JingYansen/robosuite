## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 1200000 \
    --nsteps 256 \
    --noptepochs 20 \
    --nminibatches 4 \
    --save_interval 50 \
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
    --debug 'debug'

#python bin_packing_baselines.py \
#    --alg ppo2 \
#    --num_env 1 \
#    --num_timesteps 1200000 \
#    --nsteps 256 \
#    --noptepochs 20 \
#    --nminibatches 4 \
#    --save_interval 50 \
#    --lr_type 'linear' \
#    --max 1e-3 \
#    --min 3e-4 \
#    --network 'cnn2x' \
#    --keys 'image' \
#    --use_camera_obs True \
#    --has_offscreen_renderer True \
#    --ent_coef 0.005 \
#    --camera_height 64 \
#    --camera_width 64 \
#    --random True \
#    --debug 'debug'