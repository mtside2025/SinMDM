import numpy as np
import torch
from data_utils.mixamo.motion import MotionData
from Motion.transforms import quat2repr6d
from Motion import BVH
import os

def load_sin_motion(args):
    motion_data = None
    suffix = args.sin_path.lower()[-4:]
    assert suffix in ['.npy', '.bvh']
    if args.dataset == 'humanml':
        # Support both .npy and .bvh for humanml
        if suffix == '.npy':
            try:
                motion = np.load(args.sin_path)  # only motion npy
                if len(motion.shape) == 2:
                    motion = np.transpose(motion)
                    motion = np.expand_dims(motion, axis=1)

            except:
                motion = np.array(np.load(args.sin_path, allow_pickle=True)[None][0]['motion_raw'][0])  # benchmark npy
        elif suffix == '.bvh':
            # Convert BVH to HumanML format on the fly
            print(f"Converting BVH to HumanML format: {args.sin_path}")
            from data_utils.humanml import humanml_utils
            motion = humanml_utils.bvh_to_humanml_format(args.sin_path, target_fps=20)
            # motion shape: (n_frames, 263) -> need to reshape to match expected format
            # Expected: n_feats x n_joints x n_frames but HumanML uses flat format
            # So we'll keep it as (n_frames, n_feats) and transpose
            motion = np.transpose(motion)  # (263, n_frames)
            motion = np.expand_dims(motion, axis=1)  # (263, 1, n_frames)
        else:
            raise ValueError(f"Unsupported file format for humanml: {suffix}")
        
        motion = torch.from_numpy(motion)
        motion = motion.permute(1, 0, 2)  # n_feats x n_joints x n_frames   ==> n_joints x n_feats x n_frames
        motion = motion.to(torch.float32)  # align with network dtype
    elif args.dataset == 'mixamo':  # bvh
        assert suffix == '.bvh'
        # 174 - 24 joint rotations (6d) + 3 root translation + 6*4 foot contact labels + 3 padding
        repr = 'repr6d' if args.repr == '6d' else 'quat'
        motion_data = MotionData(args.sin_path, padding=True, use_velo=True,
                                 repr=repr, contact=True, keep_y_pos=True,
                                 joint_reduction=True)
        _, raw_motion_joints, raw_motion_frames = motion_data.raw_motion.shape
        motion = motion_data.raw_motion.squeeze()
    else:
        assert args.dataset == 'bvh_general' and suffix == '.bvh'
        anim, _, _ = BVH.load(args.sin_path)
        if args.repr == '6d':
            repr_6d = quat2repr6d(torch.tensor(anim.rotations.qs))
            motion = np.concatenate([anim.positions, repr_6d], axis=2)
        else:
            motion = np.concatenate([anim.positions, anim.rotations.qs], axis=2)
        motion = torch.from_numpy(motion)
        motion = motion.permute(1, 2, 0)  # n_frames x n_joints x n_feats  ==> n_joints x n_feats x n_frames
        motion = motion.to(torch.float32)  # align with network dtype

    motion = motion.to(args.device)
    return motion, motion_data

