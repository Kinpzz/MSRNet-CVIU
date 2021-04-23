import os
import yaml
import argparse
import mxnet as mx

from utils import predict
from model import get_network
from EdgeSalDataset import EdgeSalDataset

parser = argparse.ArgumentParser()
parser.add_argument('--test_type', default=0, type=int,
                    help='0: saliency map, 1: contour (default: 0)')
parser.add_argument('--config', default='', type=str)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    for k,v in config.items():
        setattr(args, k, v)

ctx = [mx.gpu(int(args.SETTING['GPU']))]

batch_size = args.TRAIN['batch_size']*len(ctx)

type_names = ['sal', 'contour']

result_path = os.path.join(args.SETTING['root'], 'result', args.INFERENCE['dataset'], type_names[args.test_type])

# load dataset
test_root = os.path.join(args.SETTING['data_root'], args.INFERENCE['dataset'])
test_dataset = EdgeSalDataset(root=test_root,
                          subset=args.INFERENCE['subset'],
                          input_size=args.INFERENCE['input_size'],
                          image_dir=args.SETTING['image_dir'],
                          label_dir=args.SETTING['label_dir'],
                          image_set_dir=args.SETTING['image_set_dir'],
                          image_suffix=args.SETTING['image_suffix'],
                          label_suffix=args.SETTING['label_suffix'],
                          transform=False,
                          inference=True)

test_loader = mx.gluon.data.DataLoader(test_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       last_batch='keep',
                                       num_workers=1)

# load network architecture
net = get_network(args.NETWORK, input_size=args.INFERENCE['input_size'], pretrained=False, with_aspp=True)
# load trained params
if args.test_type == 0:
    trained_model = args.INFERENCE['region_model']
else:
    trained_model = args.INFERENCE['contour_model']
net.collect_params().load(trained_model)
net.hybridize()
net.collect_params().reset_ctx(ctx)

predict(test_loader, test_dataset.img_info, net, ctx, result_path, pad_zeros=False)
