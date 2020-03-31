## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 0 \
    --nsteps 1024 \
    --noptepochs 20 \
    --nminibatches 4 \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --network 'cnn' \
    --reward_shaping True \
    --camera_depth True \
    --energy_tradeoff 0.01 \
    --load_path 'results_squeeze/version-0.2/ppo2_cnn_linear_0.0003_0.0003_1200000total_1024nsteps_16env_20noptepochs_4batch_TrueshapeRw_Truedepth_0.001energy_64x64/model.pth' \
    --video_name 'demo.mp4' \
    --log True \
    --debug 'debug'