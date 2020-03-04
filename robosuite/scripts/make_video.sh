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
#    --load_path '/home/yeweirui/code/robosuite/robosuite/scripts/results/baselines/object-state_ppo2_mlp_2layer_0.001lr_256stpes_8async_0.01explore_fine_desighed/model.pth' \
#    --use_object_obs True \
#    --has_offscreen_renderer True \
#    --camera_height 320 \
#    --camera_width 240 \
#    --make_video True \
#    --render_drop_freq 10 \
#    --video_name 'fine_designed.mp4' \
#    --debug 'make_video'

python bin_packing_baselines.py \
    --log False \
    --make_video True \
    --alg ppo2 \
    --num_env 8 \
    --num_timesteps 0 \
    --noptepochs 4 \
    --network 'cnn' \
    --keys 'image' \
    --camera_name 'targetview' \
    --load_path '/home/yeweirui/code/robosuite/robosuite/scripts/results/baselines/image_ppo2_cnn_2layer_0.001lr_256stpes_8async_0.01explore__random__64x64_2view/model.pth' \
    --use_camera_obs True \
    --camera_height 64 \
    --camera_width 64 \
    --has_offscreen_renderer True \
    --render_drop_freq 10 \
    --video_name '2view_64x64_random.mp4' \
    --debug 'make_video'