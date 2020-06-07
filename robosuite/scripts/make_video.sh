set -ex
python bin_packing_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 0 \
    --nsteps 256 \
    --noptepochs 10 \
    --nminibatches 16 \
    --lr_type 'linear' \
    --max 1e-5 \
    --min 1e-5 \
    --camera_height 64 \
    --camera_width 64 \
    --log True \
    --test True \
    --load_path '/home/yeweirui/code/robosuite/robosuite/scripts/results/MultiStage/multi_test_dir/model_0.pth' \
    --video_name 'demo.mp4' \
    --debug 'temp_video'