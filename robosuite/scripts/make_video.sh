## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 0 \
    --nsteps 512 \
    --noptepochs 20 \
    --nminibatches 4 \
    --lr_type 'linear' \
    --max 3e-4 \
    --min 3e-4 \
    --network 'cnn' \
    --load_path 'results_squeeze/3view_6dim_easy_task/ppo2_cnn_linear_0.0003_0.0003_1000000total_512nsteps_16env_20noptepochs_4batch_64x64/model.pth' \
    --video_name 'demo.mp4' \
    --log True \
    --debug '3view_6dim_easy_task'