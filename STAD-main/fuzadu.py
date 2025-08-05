import argparse
import sys

import torch
import torchvision
from packaging import version
from torchvision import models as models
from AFSD.thumos14.C3DBDNet import BDNet

from ptflops import get_model_complexity_info

pt_models = {'CNN':BDNet}

# if version.parse(torchvision.__version__) > version.parse('0.15'):
#     pt_models['vit_b_16'] = models.vit_b_16


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='CNN')
    parser.add_argument('--backend', choices=list(['pytorch', 'aten']),
                        type=str, default='pytorch')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    net = pt_models[args.model]()

    if torch.cuda.is_available():
        net.cuda(device=args.device)

    if args.model == 'inception':
        input_res = (3, 299, 299)
    elif args.model == 'CNN':
        input_res = (3, 8500, 1, 30)
    else:
        input_res = (3, 224, 224)


    macs, params = get_model_complexity_info(net, input_res,
                                             as_strings=True,
                                             backend=args.backend,
                                             print_per_layer_stat=True,
                                             ost=ost)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
