## train with states
set -ex
python bin_packing_baselines.py \
    --log False \
    --make_video True \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 0 \
    --network mlp \
    --num_layers 2 \
    --keys 'state' \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --load_path 'results/baselines/object-state_ppo2_mlp_2layer_0.0003lr_1500000total_256nsteps_8env_0.2clip_0.005ent-coef_True_rand_10noptepochs_8batch_baseline/model.pth' \
    --use_object_obs True \
    --has_offscreen_renderer True \
    --camera_height 320 \
    --camera_width 240 \
    --make_video True \
    --render_drop_freq 10 \
    --video_name 'random_3e-4_258nsteps_10_8.mp4' \
    --random True \
    --debug 'make_video'

#python bin_packing_baselines.py \
#    --log False \
#    --make_video True \
#    --alg ppo2 \
#    --num_env 8 \
#    --num_timesteps 0 \
#    --noptepochs 4 \
#    --network 'cnn' \
#    --keys 'image' \
#    --camera_name 'targetview' \
#    --load_path '/home/yeweirui/code/robosuite/robosuite/scripts/results/baselines/image_ppo2_cnn_2layer_0.001lr_256stpes_8async_0.01explore__128x128_2view/model.pth' \
#    --use_camera_obs True \
#    --camera_height 128 \
#    --camera_width 128 \
#    --has_offscreen_renderer True \
#    --render_drop_freq 10 \
#    --video_name '2view_128x128.mp4' \
#    --debug 'make_video'