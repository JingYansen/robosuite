## train with states
set -ex
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
    --use_object_obs True \
    --random_take True \
    --make_video True \
    --has_offscreen_renderer True \
    --render_drop_freq 10 \
    --video_name 'fine_desighed_random.mp4' \
    --debug 'fine_desighed'
