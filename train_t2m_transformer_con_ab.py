import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer_con_ab.transformer import MaskTransformer
from models.mask_transformer_con_ab.transformer_trainer import MaskTransformerTrainer
from models.vq_seg_con.model import RVQVAE

from options.train_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper


def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20, radius=radius)

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
    
def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)

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
    
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    left_arm_vq_model.load_state_dict(ckpt['left_arm_vq_model'])
    right_arm_vq_model.load_state_dict(ckpt['right_arm_vq_model'])
    up_body_vq_model.load_state_dict(ckpt['up_body_vq_model'])
    down_body_vq_model.load_state_dict(ckpt['down_body_vq_model'])
    print(f'Loading VQ Model {opt.vq_name}')
    return left_arm_vq_model, right_arm_vq_model, up_body_vq_model, down_body_vq_model, vq_opt

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "t2m":
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        dim_pose = 263
        opt.left_arm_dim, opt.right_arm_dim, opt.up_body_dim, opt.down_body_dim = dim_sep(opt.dataset_name)
        left_arm_exp, right_arm_exp, up_body_exp, down_body_exp = 27, 27, 18, 191
        fps = 20
        radius = 4
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == "kit":
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.left_arm_dim, opt.right_arm_dim, opt.up_body_dim, opt.down_body_dim = dim_sep(opt.dataset_name)
        left_arm_exp, right_arm_exp, up_body_exp, down_body_exp = 27, 27, 9, 188
        opt.max_motion_length = 196
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'

    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')

    left_arm_net, right_arm_net, up_body_net, down_body_net, vq_opt = load_vq_model()

    bge_version = 'BAAI/bge-large-en-v1.5'

    opt.num_tokens = vq_opt.nb_code

    t2m_transformer = MaskTransformer(vq_opt=vq_opt,
                                      cond_mode='text',
                                      latent_dim=opt.latent_dim,
                                      ff_size=opt.ff_size,
                                      num_layers=opt.n_layers,
                                      num_heads=opt.n_heads,
                                      dropout=opt.dropout,
                                      bge_dim=1024,
                                      cond_drop_prob=opt.cond_drop_prob,
                                      bge_version=bge_version,
                                      opt=opt)

    if opt.fix_token_emb:
        t2m_transformer.load_and_freeze_token_emb(left_arm_net.quantizer.codebooks[0], right_arm_net.quantizer.codebooks[0],
                                                right_arm_net.quantizer.codebooks[0], down_body_net.quantizer.codebooks[0])

    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_transformer.parameters_wo_clip())

    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    trainer = MaskTransformerTrainer(opt, t2m_transformer, left_arm_net, right_arm_net, up_body_net, down_body_net)

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)