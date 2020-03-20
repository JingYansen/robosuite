## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 0 \
    --nsteps 2048 \
    --noptepochs 40 \
    --nminibatches 2 \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --network 'cnn' \
    --load_path 'results_squeeze/zw_only/ppo2_cnn_linear_0.0003_0.0003_2000000total_2048nsteps_16env_40noptepochs_2batch_64x64/model.pth' \
    --video_name 'demo.mp4' \
    --log True \
    --debug 'zw_only'