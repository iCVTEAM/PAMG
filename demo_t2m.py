import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

    converter = Joint2BVHConvertor()


    prompt_list = []
    with open('./dataset/HumanML3D/texts/000372.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            infos = line.split('#')
            prompt_list.append(infos[0])
            if len(infos) == 1 or (not infos[1].isdigit()):
                est_length = True
                length_list = []
            else:
                length_list.append(int(infos[-1]))

    data = np.load(r"E:\桌面整理\个人文件\投稿\CVPR2025\对比\LGTM\Man jumps twice in place..npy")
    captions = prompt_list
    m_length = [data.shape[0]]
    kinematic_chain = t2m_kinematic_chain
    data = np.expand_dims(data, axis=0)
    print(data.shape)
    for k, (caption, joint_data)  in enumerate(zip(captions, data)):
        print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
        r = 0
        animation_path = pjoin(animation_dir, str(k))
        joint_path = pjoin(joints_dir, str(k))

        os.makedirs(animation_path, exist_ok=True)
        os.makedirs(joint_path, exist_ok=True)

        joint_data = joint_data[:m_length[k]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

        bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
        _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

        bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
        _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

        save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
        ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))

        plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
        np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
        np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)