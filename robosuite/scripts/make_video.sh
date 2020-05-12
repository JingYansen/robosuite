## train with states
set -ex
python bin_squeeze_baselines.py \
    --alg ppo2 \
    --num_env 1 \
    --num_timesteps 0 \
    --nsteps 1024 \
    --noptepochs 10 \
    --nminibatches 32 \
    --lr_type 'linear' \
    --max 1e-5 \
    --min 1e-5 \
    --network 'cnn' \
    --energy_tradeoff 0.8 \
    --ent_coef 0.1 \
    --place_num 0 \
    --env_id 'BinSqueeze-v0' \
    --camera_type 'image+depth' \
    --fix_rotation True \
    --random_quat True \
    --random_target True \
    --load_path '/home/yeweirui/code/robosuite/robosuite/scripts/results/MultiStage/multi_test_dir/model_0.pth' \
    --video_name 'demo.mp4' \
    --log True \
    --debug 'temp_video'