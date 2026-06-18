import random

import torch
import torch.nn as nn
from models.vq_com.encdec import Encoder, Decoder
from models.vq_com.residual_vq import ResidualVQ
    
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 left_arm,
                 right_arm,
                 up_body,
                 down_body,
                 input_width=263,
                 nb_code=1024,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        # assert output_emb_width == code_dim
        self.args = args
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.left_arm_encoder = Encoder(left_arm, self.args.left_arm_code_dim, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.right_arm_encoder = Encoder(right_arm, self.args.right_arm_code_dim, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.up_body_encoder = Encoder(up_body, self.args.up_body_code_dim, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.down_body_encoder = Encoder(down_body, self.args.down_body_code_dim, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.whole_encoder = Encoder(input_width, self.args.code_dim, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        cond_dim = self.args.left_arm_code_dim +  self.args.right_arm_code_dim + self.args.up_body_code_dim + self.args.down_body_code_dim         
        self.decoder = Decoder(input_width, cond_dim, self.args.code_dim, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        left_arm_rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':self.args.left_arm_code_dim, 
            'args': args,
        }
        self.left_arm_quantizer = ResidualVQ(**left_arm_rvqvae_config)

        right_arm_rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':self.args.right_arm_code_dim, 
            'args': args,
        }
        self.right_arm_quantizer = ResidualVQ(**right_arm_rvqvae_config)

        up_body_rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':self.args.up_body_code_dim, 
            'args': args,
        }
        self.up_body_quantizer = ResidualVQ(**up_body_rvqvae_config)

        down_body_rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':self.args.down_body_code_dim, 
            'args': args,
        }
        self.down_body_quantizer = ResidualVQ(**down_body_rvqvae_config)

        whole_rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':self.args.code_dim, 
            'args': args,
        }
        self.whole_quantizer = ResidualVQ(**whole_rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        left_arm_in = x_in[:, self.args.left_arm_dim, :]
        right_arm_in = x_in[:, self.args.right_arm_dim, :]
        up_body_in = x_in[:, self.args.up_body_dim, :]
        down_body_in = x_in[:, self.args.down_body_dim, :]

        x_left_arm_encoder = self.left_arm_encoder(left_arm_in)
        x_right_arm_encoder = self.right_arm_encoder(right_arm_in)
        x_up_body_encoder = self.up_body_encoder(up_body_in)
        x_down_body_encoder = self.down_body_encoder(down_body_in)
        x_encoder = self.whole_encoder(x_in)
 
        l_code_idx, l_all_codes = self.left_arm_quantizer.quantize(x_left_arm_encoder, return_latent=True)
        r_code_idx, r_all_codes = self.right_arm_quantizer.quantize(x_right_arm_encoder, return_latent=True)
        u_code_idx, u_all_codes = self.up_body_quantizer.quantize(x_up_body_encoder, return_latent=True)
        d_code_idx, d_all_codes = self.down_body_quantizer.quantize(x_down_body_encoder, return_latent=True)
        w_code_idx, w_all_codes = self.whole_quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        b, n, q = l_code_idx.shape
        # 在新维度上堆叠它们
        code_indices = torch.zeros((b, n*5, q)).cuda().to(dtype=l_code_idx.dtype)

        code_indices[:, 0::5] = l_code_idx
        code_indices[:, 1::5] = r_code_idx
        code_indices[:, 2::5] = u_code_idx
        code_indices[:, 3::5] = d_code_idx
        code_indices[:, 4::5] = w_code_idx
        all_codes = torch.cat((l_all_codes, r_all_codes, u_all_codes, d_all_codes, w_all_codes), dim=2)

        return code_indices, all_codes

    def encode2dif(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, x_quantized

    def forward(self, x):

        x_in = self.preprocess(x)
        # Encode
        left_arm_in = x_in[:, self.args.left_arm_dim, :]
        right_arm_in = x_in[:, self.args.right_arm_dim, :]
        up_body_in = x_in[:, self.args.up_body_dim, :]
        down_body_in = x_in[:, self.args.down_body_dim, :]

        x_left_arm_encoder = self.left_arm_encoder(left_arm_in)
        x_right_arm_encoder = self.right_arm_encoder(right_arm_in)
        x_up_body_encoder = self.up_body_encoder(up_body_in)
        x_down_body_encoder = self.down_body_encoder(down_body_in)
        x_encoder = self.whole_encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        left_arm_quantized, l_code_idx, l_commit_loss, l_perplexity = self.left_arm_quantizer(x_left_arm_encoder, sample_codebook_temp=0.5)
        right_arm_quantized, r_code_idx, r_commit_loss, r_perplexity = self.right_arm_quantizer(x_right_arm_encoder, sample_codebook_temp=0.5)
        up_body_quantized, u_code_idx, u_commit_loss, u_perplexity = self.up_body_quantizer(x_up_body_encoder, sample_codebook_temp=0.5)
        down_body_quantized, d_code_idx, d_commit_loss, d_perplexity = self.down_body_quantizer(x_down_body_encoder, sample_codebook_temp=0.5)
        whole_quantized, w_code_idx, w_commit_loss, w_perplexity = self.whole_quantizer(x_encoder, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_quantized = torch.cat((left_arm_quantized, right_arm_quantized, up_body_quantized, down_body_quantized, whole_quantized), dim=1)
        x_out = self.decoder(x_quantized)
        commit_loss = l_commit_loss + r_commit_loss + u_commit_loss + d_commit_loss + w_commit_loss
        perplexity = l_perplexity + r_perplexity + u_perplexity + d_perplexity + w_perplexity
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):

        x_l = self.left_arm_quantizer.get_codes_from_indices(x[:, 0::5])
        x_r = self.right_arm_quantizer.get_codes_from_indices(x[:, 1::5])
        x_u = self.up_body_quantizer.get_codes_from_indices(x[:, 2::5])
        x_d = self.down_body_quantizer.get_codes_from_indices(x[:, 3::5])
        x_w = self.whole_quantizer.get_codes_from_indices(x[:, 4::5])

        x_t = torch.cat((x_l, x_r, x_u, x_d, x_w), dim=3)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_t.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)