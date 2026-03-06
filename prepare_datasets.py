import json
import zipfile
import scipy.io
import numpy as np
import os

def normalize_keypoints(joints_2d, mask, center_idx, scale_idx1, scale_idx2):
    # joints_2d: [N, J, 2]
    # mask: [N, J] bool, True if valid (not NaN)
    # This function operates on one example at a time, or broadly
    N, J, _ = joints_2d.shape
    out = np.full_like(joints_2d, np.nan)
    
    for i in range(N):
        if not mask[i, center_idx]:
            continue
            
        center = joints_2d[i, center_idx]
        
        if mask[i, scale_idx1] and mask[i, scale_idx2]:
            scale = np.linalg.norm(joints_2d[i, scale_idx1] - joints_2d[i, scale_idx2])
        else:
            # Fallback if scale joints are missing, just use scale 1 or skip?
            scale = 1.0
            
        if scale < 1e-6:
            scale = 1.0
            
        out[i] = (joints_2d[i] - center) / scale
        
    return out

def load_coco(json_path):
    print("Loading COCO WholeBody...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Standard COCO 17 joints
    coco_names = ['nose', 'leye', 'reye', 'lear', 'rear', 
                  'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 
                  'lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle']
                  
    # For normalization, we can use middle of hips as center, and shoulder distance as scale
    # lhip=11, rhip=12, lshoulder=5, rshoulder=6
                  
    poses = []
    for ann in data['annotations']:
        if 'keypoints' in ann:
            kpts = ann['keypoints']
            if len(kpts) >= 51:
                # [x, y, v]
                kpts = np.array(kpts[:51]).reshape(-1, 3)
                # Keep only those with v > 0
                joints = np.full((17, 2), np.nan)
                valid = kpts[:, 2] > 0
                joints[valid] = kpts[valid, :2]
                
                # Check if hips and shoulders exist to normalize
                if valid[11] and valid[12]:
                    center = (joints[11] + joints[12]) / 2.0
                    if valid[5] and valid[6]:
                        scale = np.linalg.norm(joints[5] - joints[6])
                    else:
                        scale = 1.0
                    if scale > 1e-6:
                        joints = (joints - center) / scale
                        # add dummy Z
                        joints_3d = np.concatenate([joints, np.full((17, 1), 1000.0)], axis=-1)
                        poses.append(joints_3d)
                        
    return np.array(poses), coco_names

def load_mpii(zip_path):
    print("Loading MPII...")
    # Read mat from zip
    with zipfile.ZipFile(zip_path) as z:
        with z.open('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat') as f:
            mat = scipy.io.loadmat(f)
            
    mpii_names = ['rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle', 
                  'pelvis', 'thorax', 'upperneck', 'headtop', 
                  'rwrist', 'relbow', 'rshoulder', 'lshoulder', 'lelbow', 'lwrist']
             
    # Extract annotations
    annolist = mat['RELEASE']['annolist'][0,0][0]
    poses = []
    
    for i in range(len(annolist)):
        annorect = annolist[i]['annorect']
        if annorect.size == 0:
            continue
        
        for j in range(annorect.shape[1]):
            rect = annorect[0, j]
            if rect is None or not hasattr(rect, 'dtype') or 'annopoints' not in rect.dtype.names or rect['annopoints'].size == 0:
                continue
                
            points = rect['annopoints'][0,0]['point'][0]
            
            joints = np.full((16, 2), np.nan)
            valid = np.zeros(16, dtype=bool)
            
            for p in points:
                ident = p['id'][0,0]
                if ident < 16:
                    x = p['x'][0,0]
                    y = p['y'][0,0]
                    joints[ident] = [x, y]
                    valid[ident] = True
                    
            if valid[6]: # pelvis is center
                center = joints[6]
                if valid[13] and valid[12]: # lshoulder and rshoulder for scale
                    scale = np.linalg.norm(joints[13] - joints[12])
                else:
                    scale = 1.0
                    
                if scale > 1e-6:
                    joints = (joints - center) / scale
                    joints_3d = np.concatenate([joints, np.full((16, 1), 1000.0)], axis=-1)
                    poses.append(joints_3d)
                    
    return np.array(poses), mpii_names


def main():
    coco_json = 'data/coco_wholebody_train_v1.0.json'
    mpii_zip = 'data/mpii_human_pose_v1_u12_2.zip'
    
    coco_poses, coco_names = load_coco(coco_json)
    mpii_poses, mpii_names = load_mpii(mpii_zip)
    
    print(f"COCO poses: {coco_poses.shape}")
    print(f"MPII poses: {mpii_poses.shape}")
    
    # Unified names
    all_names = list(set(coco_names + mpii_names))
    
    # Sort names so left/right are separated cleanly for ACAE?
    # Actually ACAE handles them by just finding 'l' and 'r' prefix
    
    N_coco = len(coco_poses)
    N_mpii = len(mpii_poses)
    J = len(all_names)
    
    # Create unified matrices
    poses_train = np.full((N_coco + N_mpii, J, 3), np.nan)
    
    for i, name in enumerate(coco_names):
        j_idx = all_names.index(name)
        poses_train[:N_coco, j_idx, :] = coco_poses[:, i, :]
        
    for i, name in enumerate(mpii_names):
        j_idx = all_names.index(name)
        poses_train[N_coco:, j_idx, :] = mpii_poses[:, i, :]
        
    print(f"Combined poses: {poses_train.shape}")
    
    # Optional split into train/test (e.g. 90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(poses_train))
    split = int(0.9 * len(poses_train))
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    poses_train_final = poses_train[train_idx]
    poses_test_final = poses_train[test_idx]
    
    np.save('poses_train.npy', poses_train_final)
    np.save('poses_test.npy', poses_test_final)
    np.save('joint_names.npy', np.array(all_names))
    
    print("Saved poses_train.npy, poses_test.npy, joint_names.npy")

if __name__ == '__main__':
    main()
