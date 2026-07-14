from options.base_option import BaseOptions


class EvalT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default="net_best_fid, latest, net_best_acc, net_best_top1", help='Checkpoint you want to use, {latest, net_best_fid, etc}')
        self.parser.add_argument('--ext', type=str, default='text2motion', help='Extension of the result file or folder')
        self.parser.add_argument("--cond_scale", default=4, type=float,
                                 help="Classifier-free guidance scale.")
        self.parser.add_argument("--temperature", default=1., type=float,
                                 help="Sampling temperature.")
        self.parser.add_argument("--topkr", default=0.9, type=float,
                                 help="Filter out low-probability token entries.")
        self.parser.add_argument("--time_steps", default=36, type=int,
                                 help="Mask generation steps.")
        self.parser.add_argument("--seed", default=10107, type=int)
        self.parser.add_argument('--res_name', type=str, default='rtrans_bge_b64dp0.2wog', help='Model name of residual transformer')
        self.is_train = False
