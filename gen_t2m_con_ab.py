import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from models.mask_transformer_con_ab.transformer import MaskTransformer, ResidualTransformer
from models.vq_seg_con.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain
import clip
import numpy as np
bge_version = 'BAAI/bge-large-en-v1.5'

def load_and_freeze_clip(clip_version):
    clip_model, clip_preprocess = clip.load(clip_version, device='cuda:0',
                                            jit=False)  # Must set jit=False for training
    # Cannot run on cpu
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16
    # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encode_text(raw_text):
    clip_version = 'ViT-B/32'
    clip_model = load_and_freeze_clip(clip_version)
    device = 'cuda:0'
    text = clip.tokenize(raw_text, truncate=True).to(device)
    feat_clip_text = clip_model.encode_text(text).float()
    return feat_clip_text

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
            # else:
            #     for t in left_arm:
            #         for i in range(3):
            #             left_arm_dim.append(boun+t*3+i)
            #     for t in right_arm:
            #         for i in range(3):
            #             right_arm_dim.append(boun+t*3+i)
            #     for t in up_body:
            #         for i in range(3):
            #             up_body_dim.append(boun+t*3+i)
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
            # else:
            #     for t in left_arm:
            #         for i in range(3):
            #             left_arm_dim.append(boun+t*3+i)
            #     for t in right_arm:
            #         for i in range(3):
            #             right_arm_dim.append(boun+t*3+i)
            #     for t in up_body:
            #         for i in range(3):
            #             up_body_dim.append(boun+t*3+i)
        tmp_left = set(left_arm_dim)
        tmp_right = set(right_arm_dim)
        tmp_upbody = set(up_body_dim)
        whole_body = list(range(0, 251))
        down_body_dim = [x for x in whole_body if x not in tmp_left and x not in tmp_right and x not in tmp_upbody]
        return left_arm_dim, right_arm_dim, up_body_dim, down_body_dim

def load_vq_model(vq_opt):

    left_arm_exp, right_arm_exp, up_body_exp, down_body_exp = 27, 27, 18, 191   
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
    
    up_body_vq_model = RVQVAE(vq_opt,
                up_body_exp,
                vq_opt.nb_code,
                vq_opt.up_body_code_dim,
                vq_opt.up_body_code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)

    down_body_vq_model = RVQVAE(vq_opt,
                down_body_exp,
                vq_opt.nb_code,
                vq_opt.down_body_code_dim,
                vq_opt.down_body_code_dim,
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
    up_body_vq_model.load_state_dict(ckpt['up_body_vq_model'])
    down_body_vq_model.load_state_dict(ckpt['down_body_vq_model'])
    return left_arm_vq_model, right_arm_vq_model, up_body_vq_model, down_body_vq_model, vq_opt

def load_trans_model(vq_opt, model_opt, opt, which_model):
    t2m_transformer = MaskTransformer(vq_opt = vq_opt,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      bge_dim=1024,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      bge_version=bge_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    # print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

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

    dim_pose = 251 if opt.dataset_name == 'kit' else 263
    if opt.dataset_name == 'kit' :
        left_arm_exp, right_arm_exp, up_body_exp, down_body_exp = 27, 27, 9, 188
    else:
        left_arm_exp, right_arm_exp, up_body_exp, down_body_exp = 27, 27, 18, 191
    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
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
    vq_opt.left_arm_dim, vq_opt.right_arm_dim, vq_opt.up_body_dim, vq_opt.down_body_dim = dim_sep(opt.dataset_name)
    left_arm_net, right_arm_net, up_body_net, down_body_net, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(vq_opt, model_opt, opt, 'latest.tar')

    ##################################
    #####Loading Length Predictor#####
    ##################################
    length_estimator = load_len_estimator(model_opt)

    t2m_transformer.eval()
    left_arm_net.eval()
    right_arm_net.eval()
    up_body_net.eval()
    down_body_net.eval()
    length_estimator.eval()

    t2m_transformer.to(opt.device)
    left_arm_net.to(opt.device)
    right_arm_net.to(opt.device)
    up_body_net.to(opt.device)
    down_body_net.to(opt.device)
    length_estimator.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []

    est_length = False
    if opt.text_prompt != "":
        id_list = []
        prompt_list.append(opt.text_prompt)
        id_list.append('0')
        if opt.motion_length == 0:
            est_length = True
        else:
            length_list.append(opt.motion_length)
    elif opt.text_path != "":
        with open(opt.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    est_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[-1]))
    else:
        id_list = []
        with open('./dataset/HumanML3D/test.txt') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        for id in id_list[:100]:
            with open(pjoin('./dataset/HumanML3D/texts/', id + '.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    infos = line.split('#')
                    prompt_list.append(infos[0])
                    if len(infos) == 1 or (not infos[1].isdigit()):
                        est_length = True
                        length_list = []
                    else:
                        length_list.append(int(infos[-1]))  
        # raise "A text prompt, or a file a text prompts are required!!!"
    # print('loading checkpoint {}'.format(file))

    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        text_embedding = encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()  # (b, seqlen)
        # lengths = torch.multinomial()
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list

    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            mids = t2m_transformer.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample)

            mids = mids.unsqueeze(-1)
            pred_leftarm = left_arm_net.forward_decoder(mids[:, 0::4])
            pred_upbody = up_body_net.forward_decoder(mids[:, 1::4])
            pred_downbody = down_body_net.forward_decoder(mids[:, 2::4])
            pred_right_arm = right_arm_net.forward_decoder(mids[:, 3::4])
            b, t, n = pred_leftarm.shape
            pred_motions = torch.zeros((b, t, dim_pose))
            pred_motions = torch.tensor(pred_motions, dtype=torch.float32).cuda()
            pred_motions[:, :, vq_opt.left_arm_dim] = pred_leftarm
            pred_motions[:, :, vq_opt.up_body_dim] = pred_upbody
            pred_motions[:, :, vq_opt.down_body_dim] = pred_downbody
            pred_motions[:, :, vq_opt.right_arm_dim] = pred_right_arm

            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)

        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
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