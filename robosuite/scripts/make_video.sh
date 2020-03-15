## train with states
set -ex
#python bin_packing_baselines.py \
#    --log False \
#    --make_video True \
#    --alg ppo2 \
#    --num_env 8 \
#    --num_timesteps 0 \
#    --network mlp \
#    --num_layers 2 \
#    --keys 'state' \
#    --lr_type 'linear' \
#    --max 3e-4 \
#    --min 3e-4 \
#    --load_path 'results/baselines/object-state_ppo2_mlp_2layer_0.0003lr_1500000total_256nsteps_8env_0.2clip_0.005ent-coef_True_rand_10noptepochs_8batch_baseline/model.pth' \
#    --use_object_obs True \
#    --has_offscreen_renderer True \
#    --camera_height 320 \
#    --camera_width 240 \
#    --make_video True \
#    --render_drop_freq 10 \
#    --video_name 'random_3e-4_258nsteps_10_8.mp4' \
#    --random True \
#    --debug 'make_video'

python bin_packing_baselines.py \
    --log False \
    --make_video True \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 0 \
    --noptepochs 4 \
    --network 'cnn2x' \
    --keys 'image' \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --ent_coef 0.005 \
    --load_path 'results/6objects/random_image_ppo2_cnn2x_linear_0.0003_0.0003_1000000total_256nsteps_8env_0.2clip_0.005ent-coef_40noptepochs_2batch__64x64/model.pth' \
    --use_camera_obs True \
    --camera_height 64 \
    --camera_width 64 \
    --has_offscreen_renderer True \
    --render_drop_freq 10 \
    --obj_nums '1,1,2,2' \
    --video_name 'cnn2x_3e-4.mp4' \
    --random_take True \
    --debug 'make_video'