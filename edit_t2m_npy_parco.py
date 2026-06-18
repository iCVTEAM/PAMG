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

def index_sep(name="t2m"):
    if name == "t2m":
        up_body = [12, 15]
        left_arm = [13, 16, 18, 20]
        right_arm = [14, 17, 19, 21]
    whole_body = list(range(0, 22))
    down_body = [x for x in whole_body if x not in left_arm and x not in right_arm and x not in up_body]
    return up_body, left_arm, right_arm, down_body

def dim_sep(name="t2m"):
    if name == "t2m":
        up_body = [12, 15]
        left_arm = [16, 18, 20]
        right_arm = [17, 19, 21]
        split_boundary = [4, 4+63, 4+63+126]
        up_body_dim = []
        left_arm_dim = []
        right_arm_dim = []
        for boun in split_boundary:
            if boun == 4:
                for t in left_arm:
                    for i in range(3):
                        left_arm_dim.append(boun-3+t*3+i)
                for t in right_arm:
                    for i in range(3):
                        right_arm_dim.append(boun-3+t*3+i)
                for t in up_body:
                    for i in range(3):
                        up_body_dim.append(boun-3+t*3+i)
            elif boun == 4+63:
                for t in left_arm:
                    for i in range(6):
                        left_arm_dim.append(boun-6+t*6+i)
                for t in right_arm:
                    for i in range(6):
                        right_arm_dim.append(boun-6+t*6+i)
                for t in up_body:
                    for i in range(6):
                        up_body_dim.append(boun-6+t*6+i)
        tmp_left = set(left_arm_dim)
        tmp_right = set(right_arm_dim)
        tmp_upbody = set(up_body_dim)
        whole_body = list(range(0, 263))
        down_body_dim = [x for x in whole_body if x not in tmp_left and x not in tmp_right and x not in tmp_upbody]
        return left_arm_dim, right_arm_dim, up_body_dim, down_body_dim
    if name == "kit":
        up_body = [4]
        left_arm = [8, 9, 10]
        right_arm = [5, 6, 7]
        split_boundary = [4, 4+60, 4+60+120]
        up_body_dim = []
        left_arm_dim = []
        right_arm_dim = []
        for boun in split_boundary:
            if boun == 4:
                for t in left_arm:
                    for i in range(3):
                        left_arm_dim.append(boun-3+t*3+i)
                for t in right_arm:
                    for i in range(3):
                        right_arm_dim.append(boun-3+t*3+i)
                for t in up_body:
                    for i in range(3):
                        up_body_dim.append(boun-3+t*3+i)
            elif boun == 4+60:
                for t in left_arm:
                    for i in range(6):
                        left_arm_dim.append(boun-6+t*6+i)
                for t in right_arm:
                    for i in range(6):
                        right_arm_dim.append(boun-6+t*6+i)
                for t in up_body:
                    for i in range(6):
                        up_body_dim.append(boun-6+t*6+i)
        tmp_left = set(left_arm_dim)
        tmp_right = set(right_arm_dim)
        tmp_upbody = set(up_body_dim)
        whole_body = list(range(0, 251))
        down_body_dim = [x for x in whole_body if x not in tmp_left and x not in tmp_right and x not in tmp_upbody]
        return left_arm_dim, right_arm_dim, up_body_dim, down_body_dim

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
    joint_data = np.load('parco_result/A man puts his left hand from left to right and then back to left..npy')[0]
    source_data = np.load('./dataset/HumanML3D/new_joint_vecs/000000.npy')

    up_body_dim, left_arm_dim, right_arm_dim, down_body_dim = index_sep(opt.dataset_name)
    m_length = joint_data.shape[0]
    k = 10
    if source_data.shape[0] > joint_data.shape[0]:
        length = joint_data.shape[0]
    else:
        length = source_data.shape[0]
    print(source_data.shape, joint_data.shape, length)

    # source_data[:length, right_arm_dim] = joint_data[:length, right_arm_dim]
    # joint_data = source_data
    print("---->Sample %d: %s %d"%(k, caption, m_length))
    animation_path = './editing/others/parco/'

    os.makedirs(animation_path, exist_ok=True)

    joint = recover_from_ric(torch.from_numpy(source_data).float(), 22).numpy()
    joint[:, left_arm_dim, :] = 0
    joint[:length, left_arm_dim, :] = joint_data[:length, left_arm_dim, :]

    bvh_path = pjoin(animation_path, "%s_%d_md.bvh"%(caption, k))
    _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

    bvh_path = pjoin(animation_path, "%s_%d_md.bvh" % (caption, k))
    _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

    save_path = pjoin(animation_path, "%s_%d_md.mp4"%(caption, k))
    ik_save_path = pjoin(animation_path, "%s_%d_md_ik.mp4"%(caption, k))

    plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
    plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
