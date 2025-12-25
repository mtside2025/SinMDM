import numpy as np
import torch
from Motion import BVH
from Motion.Animation import Animation
from Motion.Quaternions import Quaternions
from Motion.transforms import repr6d2quat
import os

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK

# XY-plane velocities only
HML_TRAJ_MASK = np.concatenate(([True]*(1) + [True]*(2) + [False]*(1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False]*(4)))


def bvh_to_humanml_format(bvh_path, target_fps=20):
    """
    Convert a BVH file to HumanML3D format
    
    Args:
        bvh_path: Path to input BVH file
        target_fps: Target frame rate (default 20 for HumanML)
    
    Returns:
        numpy array of shape (n_frames, 263) in HumanML format
    """
    from data_utils.humanml.common.quaternion import quaternion_to_cont6d_np
    import data_utils.humanml.utils.paramUtil as paramUtil
    
    # Load BVH
    anim, joint_names, frametime = BVH.load(bvh_path)
    source_fps = int(round(1.0 / frametime))
    
    # Get positions and rotations
    positions = anim.positions  # (n_frames, n_joints, 3)
    rotations = anim.rotations  # Quaternions object
    
    n_frames_orig = positions.shape[0]
    n_joints = positions.shape[1]
    
    # Resample to target FPS if needed
    if source_fps != target_fps:
        ratio = source_fps / target_fps
        new_n_frames = int(n_frames_orig / ratio)
        indices = np.linspace(0, n_frames_orig-1, new_n_frames).astype(int)
        positions = positions[indices]
        rotations = Quaternions(rotations.qs[indices])
        n_frames = new_n_frames
    else:
        n_frames = n_frames_orig
    
    # Convert quaternions to 6D rotation representation
    rot_6d = quaternion_to_cont6d_np(rotations.qs)  # (n_frames, n_joints, 6)
    
    # Compute root velocity (angular and linear)
    root_rot_vel = np.zeros((n_frames, 1))
    root_rot_vel[1:] = np.arctan2(rot_6d[1:, 0, 1], rot_6d[1:, 0, 0]) - \
                       np.arctan2(rot_6d[:-1, 0, 1], rot_6d[:-1, 0, 0])
    root_rot_vel[0] = root_rot_vel[1]
    
    root_linear_vel = np.zeros((n_frames, 2))
    root_linear_vel[1:] = positions[1:, 0, [0, 2]] - positions[:-1, 0, [0, 2]]
    root_linear_vel[0] = root_linear_vel[1]
    
    root_y = positions[:, 0, 1:2]  # Y position of root
    
    # Compute RIC (rotation invariant coordinates)
    ric = np.zeros((n_frames, n_joints-1, 3))
    for i, chain in enumerate(paramUtil.t2m_kinematic_chain):
        for j in range(len(chain)):
            if chain[j] == 0:  # root
                continue
            parent_idx = chain[j-1] if j > 0 else 0
            child_idx = chain[j]
            ric[:, child_idx-1] = positions[:, child_idx] - positions[:, parent_idx]
    ric_flat = ric.reshape(n_frames, -1)
    
    # Get rotation data for non-root joints
    rot_data = rot_6d[:, 1:, :].reshape(n_frames, -1)
    
    # Compute local velocity
    local_vel = np.zeros((n_frames, n_joints, 3))
    local_vel[1:] = positions[1:] - positions[:-1]
    local_vel[0] = local_vel[1]
    local_vel_flat = local_vel.reshape(n_frames, -1)
    
    # Compute foot contact
    feet_indices = [10, 11, 7, 8]  # left_foot, right_foot, left_ankle, right_ankle
    feet_pos = positions[:, feet_indices]
    feet_vel = np.zeros_like(feet_pos)
    feet_vel[1:] = feet_pos[1:] - feet_pos[:-1]
    feet_vel[0] = feet_vel[1]
    vel_mag = np.linalg.norm(feet_vel, axis=-1)
    height = feet_pos[..., 1]
    foot_contact = ((vel_mag < 0.02) & (height < 0.05)).astype(np.float32)
    
    # Concatenate all features
    humanml_data = np.concatenate([
        root_rot_vel, root_linear_vel, root_y,
        ric_flat, rot_data, local_vel_flat, foot_contact
    ], axis=1)
    
    return humanml_data


def humanml_to_bvh(humanml_data, output_path, reference_bvh_path, fps=20):
    """
    Convert HumanML3D format back to BVH using a reference BVH skeleton
    
    Args:
        humanml_data: numpy array of shape (n_frames, 263) in HumanML format
        output_path: Path to output BVH file  
        reference_bvh_path: Path to reference BVH for skeleton structure
        fps: Frame rate (default 20)
    """
    from data_utils.humanml.common.quaternion import cont6d_to_matrix_np
    import data_utils.humanml.utils.paramUtil as paramUtil
    
    # Load reference BVH to get skeleton structure
    ref_anim, ref_joint_names, _ = BVH.load(reference_bvh_path)
    
    n_frames = humanml_data.shape[0]
    n_joints = NUM_HML_JOINTS  # 22
    
    # Parse HumanML format
    # Format: root_rot_vel(1) + root_linear_vel(2) + root_y(1) + ric(63) + rot(126) + vel(66) + contact(4)
    idx = 0
    root_rot_vel = humanml_data[:, idx:idx+1]; idx += 1
    root_linear_vel = humanml_data[:, idx:idx+2]; idx += 2
    root_y = humanml_data[:, idx:idx+1]; idx += 1
    ric_flat = humanml_data[:, idx:idx+(n_joints-1)*3]; idx += (n_joints-1)*3
    rot_data = humanml_data[:, idx:idx+(n_joints-1)*6]; idx += (n_joints-1)*6
    local_vel_flat = humanml_data[:, idx:idx+n_joints*3]; idx += n_joints*3
    foot_contact = humanml_data[:, idx:idx+4]; idx += 4
    
    # Reconstruct root position from velocity
    root_pos = np.zeros((n_frames, 3))
    root_pos[:, 1] = root_y[:, 0]  # Y position
    root_pos[1:, [0, 2]] = np.cumsum(root_linear_vel[1:], axis=0)
    root_pos[0, [0, 2]] = root_linear_vel[0]
    
    # Reconstruct 6D rotations
    rot_6d = rot_data.reshape(n_frames, n_joints-1, 6)
    
    # Add root rotation (reconstruct from velocity)
    root_angles = np.cumsum(root_rot_vel[:, 0])
    root_rot_6d = np.zeros((n_frames, 6))
    root_rot_6d[:, 0] = np.cos(root_angles)
    root_rot_6d[:, 1] = np.sin(root_angles)
    root_rot_6d[:, 3] = 1.0  # Second column of rotation matrix
    
    all_rot_6d = np.concatenate([root_rot_6d[:, None, :], rot_6d], axis=1)  # (n_frames, n_joints, 6)
    
    # Convert 6D to quaternions
    rot_matrices = cont6d_to_matrix_np(all_rot_6d.reshape(-1, 6)).reshape(n_frames, n_joints, 3, 3)
    
    # Convert rotation matrices to quaternions
    from scipy.spatial.transform import Rotation as R
    quats = np.zeros((n_frames, n_joints, 4))
    for i in range(n_frames):
        for j in range(n_joints):
            r = R.from_matrix(rot_matrices[i, j])
            quats[i, j] = r.as_quat()  # [x, y, z, w]
            # Convert to [w, x, y, z] format
            quats[i, j] = np.array([quats[i, j, 3], quats[i, j, 0], quats[i, j, 1], quats[i, j, 2]])
    
    # Create Animation object
    rotations = Quaternions(quats)
    
    # Reconstruct joint positions from RIC
    positions = np.zeros((n_frames, n_joints, 3))
    positions[:, 0] = root_pos
    
    ric = ric_flat.reshape(n_frames, n_joints-1, 3)
    for i, chain in enumerate(paramUtil.t2m_kinematic_chain):
        for j in range(len(chain)):
            if chain[j] == 0:
                continue
            parent_idx = chain[j-1] if j > 0 else 0
            child_idx = chain[j]
            positions[:, child_idx] = positions[:, parent_idx] + ric[:, child_idx-1]
    
    # Create Animation with reference skeleton structure
    anim = Animation(
        rotations=rotations,
        positions=positions,
        orients=ref_anim.orients,
        offsets=ref_anim.offsets,
        parents=ref_anim.parents
    )
    
    # Write to BVH file
    BVH.save(output_path, anim, names=ref_joint_names, frametime=1.0/fps)
    
    return anim
