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
    --max 2e-5 \
    --min 1e-5 \
    --network 'cnn' \
    --energy_tradeoff 0.8 \
    --ent_coef 0.1 \
    --place_num 3 \
    --load_path 'results_squeeze/version-0.9.5/ppo2_cnn_linear_2e-05_1e-05_1500000total_1024nsteps_16env_10noptepochs_32batch_3init_0.1entropy_Falsefix_0.8energy_Falsenodelta_64x64/checkpoints/00001.zip' \
    --video_name 'demo.mp4' \
    --log True \
    --debug 'make_video'