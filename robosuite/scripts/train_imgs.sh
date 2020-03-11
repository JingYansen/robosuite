## train with states
set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 1000000 \
    --nsteps 128 \
    --noptepochs 20 \
    --nminibatches 4 \
    --save_interval 50 \
    --lr_type 'linear' \
    --max 1e-3 \
    --min 3e-4 \
    --ent_coef 0.005 \
    --network 'cnn2x' \
    --keys 'image' \
    --use_camera_obs True \
    --has_offscreen_renderer True \
    --camera_height 64 \
    --camera_width 64 \
    --test True \
    --log True \
    --obj_nums '1,1,2,2' \
    --random_take True \
    --debug '6objects'
