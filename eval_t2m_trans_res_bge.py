import os
from os.path import join as pjoin

import torch

from models.mask_transformer_bge.transformer import MaskTransformer, ResidualTransformer
from models.vq_seg_con.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

import utils.eval_t2m as eval_t2m
from utils.fixseed import fixseed

import numpy as np

def load_vq_model(vq_opt, which_epoch):

    if vq_opt.dataset_name == 't2m':
        left_arm_exp, right_arm_exp, up_body_exp, down_body_exp = 48, 48, 60, 107
    else:
        left_arm_exp, right_arm_exp, up_body_exp, down_body_exp = 36, 36, 48, 131
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

    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch),
                            map_location='cpu')
    left_arm_vq_model.load_state_dict(ckpt['left_arm_vq_model'])
    right_arm_vq_model.load_state_dict(ckpt['right_arm_vq_model'])
    up_body_vq_model.load_state_dict(ckpt['up_body_vq_model'])
    down_body_vq_model.load_state_dict(ckpt['down_body_vq_model'])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.vq_name}')
    return left_arm_vq_model, right_arm_vq_model, up_body_vq_model, down_body_vq_model, vq_epoch

def dim_sep(name="t2m"):
    if name == "t2m":
        up_body = [3, 6, 9, 12, 15]
        left_arm = [13, 16, 18, 20]
        right_arm = [14, 17, 19, 21]
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
            else:
                for t in left_arm:
                    for i in range(3):
                        left_arm_dim.append(boun+t*3+i)
                for t in right_arm:
                    for i in range(3):
                        right_arm_dim.append(boun+t*3+i)
                for t in up_body:
                    for i in range(3):
                        up_body_dim.append(boun+t*3+i)
        tmp_left = set(left_arm_dim)
        tmp_right = set(right_arm_dim)
        tmp_upbody = set(up_body_dim)
        whole_body = list(range(0, 263))
        down_body_dim = [x for x in whole_body if x not in tmp_left and x not in tmp_right and x not in tmp_upbody]
        return left_arm_dim, right_arm_dim, up_body_dim, down_body_dim
    if name == "kit":
        up_body = [1, 2, 3, 4]
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
            else:
                for t in left_arm:
                    for i in range(3):
                        left_arm_dim.append(boun+t*3+i)
                for t in right_arm:
                    for i in range(3):
                        right_arm_dim.append(boun+t*3+i)
                for t in up_body:
                    for i in range(3):
                        up_body_dim.append(boun+t*3+i)
        tmp_left = set(left_arm_dim)
        tmp_right = set(right_arm_dim)
        tmp_upbody = set(up_body_dim)
        whole_body = list(range(0, 251))
        down_body_dim = [x for x in whole_body if x not in tmp_left and x not in tmp_right and x not in tmp_upbody]
        return left_arm_dim, right_arm_dim, up_body_dim, down_body_dim

def load_trans_model(vq_opt, model_opt, which_model):
    t2m_transformer = MaskTransformer(vq_opt=vq_opt,
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
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(vq_opt, res_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(vq_opt=vq_opt,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            bge_dim=1024,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            bge_version=bge_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

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
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "%s.log"%opt.ext)

    f = open(pjoin(out_path), 'w')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    bge_version = 'BAAI/bge-large-en-v1.5'
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    left_arm_net, right_arm_net, up_body_net, down_body_net, vq_epoch = load_vq_model(vq_opt, 'net_best_fid.tar')
    model_opt.left_arm_dim, model_opt.right_arm_dim, model_opt.up_body_dim, model_opt.down_body_dim = dim_sep(vq_opt.dataset_name)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.left_arm_code_dim = vq_opt.left_arm_code_dim
    model_opt.right_arm_code_dim = vq_opt.right_arm_code_dim
    model_opt.up_body_code_dim = vq_opt.up_body_code_dim
    model_opt.down_body_code_dim = vq_opt.down_body_code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(vq_opt, res_opt)

    assert res_opt.vq_name == model_opt.vq_name

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    # model_dir = pjoin(opt.)
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and file[:-4] not in opt.which_epoch:
            continue
        print('loading checkpoint {}'.format(file))
        t2m_transformer = load_trans_model(vq_opt, model_opt, file)
        t2m_transformer.to(opt.device)
        t2m_transformer.eval()

        left_arm_net.eval()
        left_arm_net.cuda()
        right_arm_net.eval()
        right_arm_net.cuda()
        up_body_net.eval()
        up_body_net.cuda()
        down_body_net.eval()
        down_body_net.cuda()
        res_model.to(opt.device)
        res_model.eval()

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mm = []

        repeat_time = 20
        for i in range(repeat_time):
            with torch.no_grad():
                best_fid, best_div, Rprecision, best_matching, best_mm = \
                    eval_t2m.evaluation_mask_transformer_test_plus_res_seg(model_opt, eval_val_loader, left_arm_net, up_body_net, down_body_net, right_arm_net, res_model, t2m_transformer,
                                                                       i, eval_wrapper=eval_wrapper,
                                                         time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                                                         temperature=opt.temperature, topkr=opt.topkr,
                                                                       force_mask=opt.force_mask, cal_mm=True)
                # eval_t2m.evaluation_mask_transformer_res_test(model_opt, eval_val_loader, t2m_transformer, left_arm_net, right_arm_net, body_net, eval_wrapper=eval_wrapper)
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mm.append(best_mm)

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mm = np.array(mm)

        print(f'{file} final result:')
        print(f'{file} final result:', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()
