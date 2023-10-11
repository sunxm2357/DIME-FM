import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')

    parser.add_argument('--amp', action='store_true',
                        help='specify if using amp in training')

    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='the local rank for distributed training. ')

    parser.add_argument('--student_config_file', type=str, default='configs/models/vitb32_large_language.yaml',
                        help='the base path of datasets')

    # ####################  dataset #########################
    parser.add_argument('--dataroot', type=str, default='/fsx/sunxm/datasets/',
                        help='the base path of datasets')
    parser.add_argument('--tsv_file_list', type=str, nargs='+',  default=[],
                        help='the large scale dataset')
    parser.add_argument('--batch_size', type=int, default=640,
                        help='batch size used for the  dataset')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='the number of data workers')


    # ########################### optimizers and schedulers ###########################
    parser.add_argument('--base_lr', type=float, default=8e-4, help='the base learning rate')
    parser.add_argument('--warmup_lr', type=float, default=4e-6, help='the warmup learning rate')
    parser.add_argument('--min_lr', type=float, default=4e-5, help='the min learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='the number of epochs to train image encoder')
    parser.add_argument('--warmup_epochs', type=float, default=4, help='the base learning rate')

    # ###########################  Loss ###################################
    parser.add_argument('--t_vl', type=float, default=1.0,
                        help='the temperature of vl loss distillation (usually < 1)')
    parser.add_argument('--t_pvl', type=float, default=3.0,
                        help='the temperature of pseudo vl distillation (usually < 1)')
    parser.add_argument('--t_udist', type=float, default=7.0,
                        help='the temperature of udist loss distillation (usually < 1)')

    parser.add_argument('--w_vl_loss', type=float, default=0.7,
                        help='the weight of vl distillation loss')
    parser.add_argument('--w_pvl_loss', type=float, default=0.3,
                        help='the weight of pseudo vl distillation loss')
    parser.add_argument('--w_udist_loss', type=float, default=0,
                        help='the weight of udist  distillation loss')

    parser.add_argument('--use_udist_loss', action='store_true',
                        help='use the udist loss')
    parser.add_argument('--use_pvl_loss', action='store_true',
                        help='use the pseudo vl loss')

    # ########################### Training pipeline #######################
    parser.add_argument('--print_freq', type=int, default=100,
                        help='the frequency to print the intermediate losses during training')

    # ########################### Logging #######################
    parser.add_argument('--output_dir', type=str, default='./snapshots',
                        help='the directory to save logs')
    parser.add_argument('--prefix', type=str,
                        help='the prefix of the log folder name')

    return parser