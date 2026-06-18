import sys
import os
from os.path import join as pjoin

import torch
from models.vq_seg.model import RVQVAE
from options.vq_option import arg_parse
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
import utils.eval_t2m as eval_t2m
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from utils.word_vectorizer import WordVectorizer

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
                        up_body_dim.append(boun+t*3+i)
            elif boun == 4+60:
                for t in left_arm:
                    for i in range(6):
                        left_arm_dim.append(boun+t*6+i)
                for t in right_arm:
                    for i in range(6):
                        right_arm_dim.append(boun+t*6+i)
                for t in up_body:
                    for i in range(6):
                        up_body_dim.append(boun+t*6+i)
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
    print(f'Loading VQ Model {vq_opt.vq_name} epoch:{vq_epoch}')
    return left_arm_vq_model, right_arm_vq_model, up_body_vq_model, down_body_vq_model, vq_epoch

if __name__ == "__main__":
    ##### ---- Exp dirs ---- #####
    args = arg_parse(False)
    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))

    args.out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    os.makedirs(args.out_dir, exist_ok=True)

    f = open(pjoin(args.out_dir, '%s.log'%args.ext), 'w')

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataset_name == 'kit' \
                                                        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    args.nb_joints = 21 if args.dataset_name == 'kit' else 22
    dim_pose = 251 if args.dataset_name == 'kit' else 263

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=args.device)

    print(len(eval_val_loader))

    ##### ---- Network ---- #####
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=args.device)
    # net = load_vq_model()
    vq_opt.left_arm_dim, vq_opt.right_arm_dim, vq_opt.up_body_dim, vq_opt.down_body_dim = dim_sep(vq_opt.dataset_name)

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    for file in os.listdir(model_dir):
        # if not file.endswith('tar'):
        #     continue
        if not file.startswith('net_best_fid'):
            continue
        if args.which_epoch != "all" and args.which_epoch not in file:
            continue
        print(file)
        left_arm_net, right_arm_net, up_body_net, down_body_net, ep = load_vq_model(vq_opt, file)

        left_arm_net.eval()
        left_arm_net.cuda()
        right_arm_net.eval()
        right_arm_net.cuda()
        up_body_net.eval()
        up_body_net.cuda()
        down_body_net.eval()
        down_body_net.cuda()

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mae = []
        repeat_time = 20
        for i in range(repeat_time):
            best_fid, best_div, Rprecision, best_matching, l1_dist = \
                eval_t2m.evaluation_part_vqvae_plus_mpjpe(vq_opt, eval_val_loader, left_arm_net, right_arm_net, up_body_net, down_body_net, i, eval_wrapper=eval_wrapper, num_joint=args.nb_joints)
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mae.append(l1_dist)

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mae = np.array(mae)

        print(f'{file} final result, epoch {ep}')
        print(f'{file} final result, epoch {ep}', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMAE:{np.mean(mae):.3f}, conf.{np.std(mae)*1.96/np.sqrt(repeat_time):.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()

