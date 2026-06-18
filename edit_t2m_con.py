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

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np

from gen_t2m_con_test import load_vq_model, load_res_model, load_trans_model, dim_sep

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./editing', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    left_arm_net, right_arm_net, up_body_net, down_body_net, vq_opt = load_vq_model(vq_opt)
    vq_opt.left_arm_dim, vq_opt.right_arm_dim, vq_opt.up_body_dim, vq_opt.down_body_dim = dim_sep(opt.dataset_name)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(vq_opt, model_opt, opt, 'latest.tar')

    t2m_transformer.eval()
    left_arm_net.eval()
    right_arm_net.eval()
    up_body_net.eval()
    down_body_net.eval()
    res_model.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    left_arm_net.to(opt.device)
    right_arm_net.to(opt.device)
    up_body_net.to(opt.device)
    down_body_net.to(opt.device)

    ##### ---- Data ---- #####
    max_motion_length = 196
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean
    ### We provided an example source motion (from 'new_joint_vecs') for editing. See './example_data/000612.mp4'###
    motion = np.load(opt.source_motion)
    m_length = len(motion)
    motion = (motion - mean) / std
    if max_motion_length > m_length:
        motion = np.concatenate([motion, np.zeros((max_motion_length - m_length, motion.shape[1])) ], axis=0)
    motion = torch.from_numpy(motion)[None].to(opt.device).to(dtype=torch.float32)
    motion = motion[:opt.motion_length]
    
    prompt_list = []
    length_list = []
    if opt.motion_length == 0:
        opt.motion_length = m_length
        print("Using default motion length.")
    
    prompt_list.append(opt.text_prompt)
    length_list.append(opt.motion_length)
    if opt.text_prompt == "":
        raise "Using an empty text prompt."

    token_lens = torch.LongTensor(length_list) // 4
    token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list
    print_captions = captions[0]

    _edit_slice = opt.mask_edit_section
    edit_slice = []
    for eds in _edit_slice:
        _start, _end = eds.split(',')
        _start = eval(_start)
        _end = eval(_end)
        edit_slice.append([_start, _end])

    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    with torch.no_grad():
        #c print(motion.shape)
        la_tokens, la_features = left_arm_net.encode(motion[:, :, vq_opt.left_arm_dim])
        ub_tokens, ub_features = up_body_net.encode(motion[:, :, vq_opt.up_body_dim])
        db_tokens, db_features = down_body_net.encode(motion[:, :, vq_opt.down_body_dim])
        ra_tokens, ra_features = right_arm_net.encode(motion[:, :, vq_opt.right_arm_dim])

        b, n, q = la_tokens.shape
        # 在新维度上堆叠它们
        tokens = torch.zeros((b, n*4, q), dtype=torch.long).cuda()
        tokens[:, 0::4] = la_tokens
        tokens[:, 1::4] = ub_tokens
        tokens[:, 2::4] = db_tokens
        tokens[:, 3::4] = ra_tokens
        
    ### build editing mask, TOEDIT marked as 1 ###
    edit_mask = torch.zeros_like(tokens[..., 0])
    seq_len = tokens.shape[1]
    # 创建一个字典来映射 limb 到 offset
    limb_to_offset = {
        'left_arm': 0,
        'up_body': 1,
        'down_body': 2,
        'right_arm': 3
    }
    
    # 遍历 opt.edit_limb 列表
    opt.edit_limb = ['up_body']

    for limb in opt.edit_limb:
        if limb in limb_to_offset:
            offset = limb_to_offset[limb]
            
            # 对于每个编辑区间，设置 edit_mask
            for _start, _end in edit_slice:
                if isinstance(_start, float):
                    _start = int(_start * seq_len)
                    _end = int(_end * seq_len)
                else:
                    _start //= 4
                    _end //= 4
                edit_mask[:, _start + offset: _end + offset: 4] = 1
        else:
            print(f"Warning: Unknown limb '{limb}'")
        print_captions = f'{print_captions} [{_start*4/20.}s - {_end*4/20.}s]'
    edit_mask = edit_mask.bool()
    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            mids = t2m_transformer.edit(
                                        captions, tokens[..., 0].clone(), m_length//4,
                                        timesteps=opt.time_steps,
                                        cond_scale=opt.cond_scale,
                                        temperature=opt.temperature,
                                        topk_filter_thres=opt.topkr,
                                        gsample=opt.gumbel_sample,
                                        force_mask=opt.force_mask,
                                        edit_mask=edit_mask.clone(),
                                        )
            if opt.use_res_model:
                mids = res_model.generate(mids, captions, m_length//4, temperature=1, cond_scale=2)
            else:
                mids.unsqueeze_(-1)

            pred_leftarm = left_arm_net.forward_decoder(mids[:, 0::4])
            pred_upbody = up_body_net.forward_decoder(mids[:, 1::4])
            pred_downbody = down_body_net.forward_decoder(mids[:, 2::4])
            pred_right_arm = right_arm_net.forward_decoder(mids[:, 3::4])
            pred_motions = motion.clone()
            for limb in opt.edit_limb:
                if limb == 'left_arm':
                    pred_motions[:, :pred_leftarm.shape[1], vq_opt.left_arm_dim] = pred_leftarm
                elif limb == 'up_body':
                    pred_motions[:, :pred_upbody.shape[1], vq_opt.up_body_dim] = pred_upbody
                elif limb == 'down_body':
                    pred_motions[:, :pred_downbody.shape[1], vq_opt.down_body_dim] = pred_downbody
                elif limb == 'right_arm':
                    pred_motions[:, :pred_right_arm.shape[1], vq_opt.right_arm_dim] = pred_right_arm

            pred_motions = pred_motions.detach().cpu().numpy()

            source_motions = motion.detach().cpu().numpy()

            data = inv_transform(pred_motions)
            source_data = inv_transform(source_motions)

        for k, (caption, joint_data, source_data)  in enumerate(zip(captions, data, source_data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            source_data = source_data[:m_length[k]]
            soucre_joint = recover_from_ric(torch.from_numpy(source_data).float(), 22).numpy()

            bvh_path = pjoin(animation_path, "%s_sample%d_repeat%d_len%d_ik.bvh"%(caption, k, r, m_length[k]))
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "%s_sample%d_repeat%d_len%d.bvh" % (caption, k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

            save_path = pjoin(animation_path, "%s_sample%d_repeat%d_len%d.mp4"%(caption, k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "%s_sample%d_repeat%d_len%d_ik.mp4"%(caption, k, r, m_length[k]))
            source_save_path = pjoin(animation_path, "%s_sample%d_source_len%d.mp4"%(caption, k, m_length[k]))

            plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=print_captions, fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title=print_captions, fps=20)
            plot_3d_motion(source_save_path, kinematic_chain, soucre_joint, title='None', fps=20)
            np.save(pjoin(joint_path, "%s_sample%d_repeat%d_len%d.npy"%(caption, k, r, m_length[k])), joint)
            np.save(pjoin(joint_path, "%s_sample%d_repeat%d_len%d_ik.npy"%(caption, k, r, m_length[k])), ik_joint)