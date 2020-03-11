## train with states
set -ex
python test_hard_case_policy.py \
    --log False \
    --make_video True \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 0 \
    --network mlp \
    --num_layers 2 \
    --keys 'state' \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --ent_coef 0.005 \
    --load_path 'results/16objects/random_state_ppo2_mlp_linear_0.0005_0.0005_1000000total_128nsteps_16env_0.2clip_0.005ent-coef_40noptepochs_2batch_/model.pth' \
    --use_object_obs True \
    --has_offscreen_renderer True \
    --camera_height 320 \
    --camera_width 240 \
    --make_video True \
    --render_drop_freq 10 \
    --obj_nums '3,3,3,3' \
    --video_name 'hardcase.mp4' \
    --random_take True \
    --debug 'make_video'


#python bin_packing_baselines.py \
#    --log False \
#    --make_video True \
#    --alg ppo2 \
#    --num_env 8 \
#    --num_timesteps 0 \
#    --noptepochs 4 \
#    --network 'cnn2x' \
#    --keys 'image' \
#    --lr_type 'linear' \
#    --max 1e-3 \
#    --min 3e-4 \
#    --ent_coef 0.005 \
#    --load_path '/home/yeweirui/code/robosuite/robosuite/scripts/results/baselines/image_ppo2_cnn_2layer_0.001lr_256stpes_8async_0.01explore__128x128_2view/model.pth' \
#    --use_camera_obs True \
#    --camera_height 64 \
#    --camera_width 64 \
#    --has_offscreen_renderer True \
#    --render_drop_freq 10 \
#    --obj_nums '3,3,3,3' \
#    --video_name '2view_128x128.mp4' \
#    --random_take True \
#    --debug 'make_video'