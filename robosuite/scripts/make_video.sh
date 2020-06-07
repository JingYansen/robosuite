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
    --render_drop_freq 20 \
    --load_path '/home/yeweirui/code/robosuite/robosuite/scripts/results/BinPack-v0/version-1.0.0/image+depth/ppo2_cnn_linear_0.0001_0.0001_1000000total_256nsteps_10noptepochs_16batch_6take_64x64/model.pth' \
    --video_name 'demo.mp4' \
    --debug 'temp_video'