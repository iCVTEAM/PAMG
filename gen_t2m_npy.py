import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from models.mask_transformer_con.transformer import MaskTransformer, ResidualTransformer
from models.vq_seg_con.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np
clip_version = 'ViT-B/32'

def dim_sep(name="t2m"):
    if name == "t2m":
        left_arm = [16, 18, 20]
        right_arm = [17, 19, 21]
        split_boundary = [4, 4+63, 4+63+126]
        left_arm_dim = []
        right_arm_dim = []
        for boun in split_boundary:
            if boun == 4 or boun == 4+63:
                for t in left_arm:
                    for i in range(3):
                        left_arm_dim.append(boun+t*3+i)
                for t in right_arm:
                    for i in range(3):
                        right_arm_dim.append(boun+t*3+i)
        tmp_left = set(left_arm_dim)
        tmp_right = set(right_arm_dim)
        whole_body = list(range(0, 263))
        whole_body_dim = [x for x in whole_body if x not in tmp_left and x not in tmp_right]
        return left_arm_dim, right_arm_dim, whole_body_dim
    if name == "kit":
        left_arm = [8, 9, 10]
        right_arm = [5, 6, 7]
        split_boundary = [4, 4+60, 4+60+120]
        left_arm_dim = []
        right_arm_dim = []
        for boun in split_boundary:
            if boun == 4 or boun == 4+60:
                for t in left_arm:
                    for i in range(3):
                        left_arm_dim.append(boun+t*3+i)
                for t in right_arm:
                    for i in range(3):
                        right_arm_dim.append(boun+t*3+i)
        tmp_left = set(left_arm_dim)
        tmp_right = set(right_arm_dim)
        whole_body = list(range(0, 251))
        whole_body_dim = [x for x in whole_body if x not in tmp_left and x not in tmp_right]
        return left_arm_dim, right_arm_dim, whole_body_dim

def load_vq_model(vq_opt):

    left_arm_vq_model = RVQVAE(vq_opt,
                left_arm_exp,
                vq_opt.nb_code,
                vq_opt.left_arm_code_dim,
                vq_opt.left_arm_code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    
    right_arm_vq_model = RVQVAE(vq_opt,
            right_arm_exp,
            vq_opt.nb_code,
            vq_opt.right_arm_code_dim,
            vq_opt.right_arm_code_dim,
            vq_opt.down_t,
            vq_opt.stride_t,
            vq_opt.width,
            vq_opt.depth,
            vq_opt.dilation_growth_rate,
            vq_opt.vq_act,
            vq_opt.vq_norm)

    body_vq_model = RVQVAE(vq_opt,
                whole_body_exp,
                vq_opt.nb_code,
                vq_opt.body_code_dim,
                vq_opt.body_code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)

    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'latest.tar'),
                            map_location='cpu')
    left_arm_vq_model.load_state_dict(ckpt['left_arm_vq_model'])
    right_arm_vq_model.load_state_dict(ckpt['right_arm_vq_model'])
    body_vq_model.load_state_dict(ckpt['body_vq_model'])
    print(f'Loading VQ Model {opt.vq_name}')
    return left_arm_vq_model, right_arm_vq_model, body_vq_model, vq_opt

def load_trans_model(vq_opt, model_opt, opt, which_model):
    t2m_transformer = MaskTransformer(vq_opt = vq_opt,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(vq_opt=vq_opt,
                                        cond_mode='text',
                                        latent_dim=res_opt.latent_dim,
                                        ff_size=res_opt.ff_size,
                                        num_layers=res_opt.n_layers,
                                        num_heads=res_opt.n_heads,
                                        dropout=res_opt.dropout,
                                        clip_dim=512,
                                        shared_codebook=vq_opt.shared_codebook,
                                        cond_drop_prob=res_opt.cond_drop_prob,
                                        # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                        share_weight=res_opt.share_weight,
                                        clip_version=clip_version,
                                        opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)
    print(opt.gpu_id)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    caption = 'A man puts his left hand from left to right and then back to left.'
    joint_data = np.load(r"E:\桌面整理\个人文件\投稿\CVPR2025\对比\LGTM\This person swings both arms as though he is doing a workout, and moves quickly with his whole body..npy")
    # 162
    joint_data = joint_data

    # joint_data = np.load('motion.npy')
    m_length = joint_data.shape[0]
    k = 18
    print("---->Sample %d: %s %d"%(k, caption, m_length))
    animation_path = './generation/others/lgtm/'

    os.makedirs(animation_path, exist_ok=True)

    print(joint_data.shape)
    joint = joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
    print(joint.shape)
    # joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

    bvh_path = pjoin(animation_path, "%d_origin.bvh"%(k))
    _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

    bvh_path = pjoin(animation_path, "%d_origin.bvh" % (k))
    _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

    save_path = pjoin(animation_path, "%d_origin.mp4"%(k))
    ik_save_path = pjoin(animation_path, "%d_origin_ik.mp4"%(k))

    plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
    plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
