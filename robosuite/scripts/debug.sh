## train with states

python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 3 \
    --num_timesteps 1000000 \
    --nsteps 256 \
    --save_interval 100 \
    --lr 1e-3 \
    --network mlp \
    --num_layers 2 \
    --ent_coef 0.01 \
    --use_object_obs True \
    --debug 'debug_log_lasttime'

#python bin_packing_baselines.py \
#    --alg ppo2 \
#    --num_env 1 \
#    --num_timesteps 1000000 \
#    --nsteps 256 \
#    --save_interval 50 \
#    --lr 1e-3 \
#    --network 'cnn' \
#    --keys 'image' \
#    --camera_name 'targetview' \
#    --use_camera_obs True \
#    --has_offscreen_renderer True \
#    --ent_coef 0.01 \
#    --debug 'test_2image'