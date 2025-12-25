#!/usr/bin/env bash

# https://github.com/SinMDM/SinMDM?tab=readme-ov-file
# --edit_mode     in_betweening, expansion, upper_body, lower_body, harmonization
# --ref_motion    is the path to the reference motion
# --prefix_end    is to specify the length of the prefix
# --suffix_start  is to specify the length of the suffix
# --num_samples   is the number of motions that will be generated
# --seed          is to specify a seed.

# スクリプトのディレクトリに移動
cd "$(dirname "$0")"

# デフォルト値の設定
REF_MOTION=${1:-"./dataset/mixamo/0000_Breakdance_Freezes.bvh"}
PREFIX_END=${2:-"0.25"}
SUFFIX_START=${3:-"0.75"}

args=(
    --ref_motion   "$REF_MOTION"
    --prefix_end   "$PREFIX_END"
    --suffix_start "$SUFFIX_START"
    --output_dir   "/mnt/g/3d/motion/generated/SinMDM"
    --edit_mode    harmonization
#    --model_path   ./save/humanml/0000/model000020000.pt
    --model_path   ./save/mixamo/0000/model000060000.pt
    --num_samples  1
    --seed         0
)

echo "Executing command:"
echo "python -m sample.edit ${args[@]}"
echo ""

python -m sample.edit "${args[@]}"



#usage: edit.py [-h] [--cuda CUDA] [--device DEVICE] [--seed SEED] [--batch_size BATCH_SIZE] --model_path
#               MODEL_PATH [--output_dir OUTPUT_DIR] [--num_samples NUM_SAMPLES]
#               [--edit_mode {in_betweening,expansion,upper_body,lower_body,harmonization}]
#               [--prefix_end PREFIX_END] [--suffix_start SUFFIX_START] [--prefix_length PREFIX_LENGTH]
#               [--suffix_length SUFFIX_LENGTH] [--ref_motion REF_MOTION] [--num_frames NUM_FRAMES]
#               [--dataset {humanml,mixamo,bvh_general}] [--repr {quat,6d}] [--data_dir DATA_DIR]
#               [--num_joints NUM_JOINTS] [--arch {trans_enc,trans_dec,gru,unet,qna}]
#               [--emb_trans_dec EMB_TRANS_DEC] [--layers LAYERS] [--latent_dim LATENT_DIM]
#               [--cond_mask_prob COND_MASK_PROB] [--unconstrained] [--image_size IMAGE_SIZE]
#               [--num_channels NUM_CHANNELS] [--num_res_blocks NUM_RES_BLOCKS] [--num_heads NUM_HEADS]
#               [--num_heads_upsample NUM_HEADS_UPSAMPLE] [--num_head_channels NUM_HEAD_CHANNELS]
#               [--attention_resolutions ATTENTION_RESOLUTIONS] [--channel_mult CHANNEL_MULT] [--learn_sigma]
#               [--dropout DROPOUT] [--class_cond] [--use_checkpoint] [--use_scale_shift_norm]
#               [--resblock_updown] [--use_fp16] [--use_new_attention_order] [--conv_1d CONV_1D]
#               [--padding_mode {zeros,reflect,replicate,circular}] [--padding PADDING]
#               [--lr_method LR_METHOD] [--lr_step LR_STEP] [--lr_gamma LR_GAMMA]
#               [--use_attention USE_ATTENTION] [--use_qna USE_QNA] [--head_dim HEAD_DIM]
#               [--num_downsample NUM_DOWNSAMPLE] [--drop_path DROP_PATH] [--use_diffusion_query]
#               [--kernel_size KERNEL_SIZE] [--use_global_pe] [--noise_schedule {linear,cosine}]
#               [--diffusion_steps DIFFUSION_STEPS] [--sigma_small SIGMA_SMALL] [--crop_ratio CROP_RATIO]
#               [--sin_path SIN_PATH]

