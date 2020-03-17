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
    --max 1e-3 \
    --min 1e-3 \
    --network 'cnn' \
    --load_path 'results_squeeze/zw_only/ppo2_cnn_linear_0.0001_0.0001_1000000total_1024nsteps_8env_20noptepochs_4batch_64x64/model.pth' \
    --video_name 'demo.mp4' \
    --log True \
    --debug 'zw_only'