# Specify architecture using --arch Options: unet, qna
# Specify dataset using --dataset Options: humanml, mixamo, bvh_general
# Use --device to define GPU id.
# Use --seed to specify seed.
# Add --train_platform_type {ClearmlPlatform, TensorboardPlatform} to track results with either ClearML or Tensorboard.
# Add --eval_during_training to run a short evaluation for each saved checkpoint.
# Add --gen_during_training to synthesize a motion and save its visualization for each saved checkpoint.
# Evaluation and generation during training will slow it down but will give you better monitoring.
# Please refer to file utils/parser_util.py for more arguments.

python -m train.train_sinmdm \
    --arch unet \
    --dataset mixamo \
    --save_dir <'path_to_save_models'> \
    --sin_path <'path to .bvh file for mixamo/bvh_general dataset or .npy file for humanml dataset'> \
    --lr_method ExponentialLR \
    --lr_gamma 0.99998 \
    --use_scale_shift_norm \
    --use_checkpoint

