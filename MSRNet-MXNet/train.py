import os
import yaml
import argparse
import mxnet as mx
from mxnet import gluon

from utils import train
from model import get_network
from EdgeSalDataset import EdgeSalDataset
from loss import SigmoidBinaryCrossEntropyLoss_Edge

parser = argparse.ArgumentParser()
parser.add_argument('--train_type', default=0, type=int,
                    help='0: saliency map, 1: contour (default: 0)')
parser.add_argument('--begin_epoch', default=0, type=int,
                    help='Resume training from this epoch, 0 means training from initilization (defalut: 0)')
parser.add_argument('--config', default='', type=str)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    for k,v in config.items():
        setattr(args, k, v)

ctx = []
gpus = [int(gpu) for gpu in args.SETTING['GPU'].split(',')]
for gpu in gpus:
    ctx.append(mx.gpu(gpu))
batch_size = args.TRAIN['batch_size']*len(ctx)

type_names = ['sal', 'contour']

save_prefix = os.path.join(args.SETTING['root'], args.SETTING['model_dir'], args.TRAIN['dataset'], type_names[args.train_type])
visual_path = os.path.join(args.SETTING['root'], args.SETTING['visual_dir'], args.TRAIN['dataset'], type_names[args.train_type])
logdir = os.path.join(args.SETTING['root'], args.SETTING['log_dir'], args.TRAIN['dataset'], type_names[args.train_type])
if not os.path.exists(save_prefix):
    os.makedirs(save_prefix)

train_root = os.path.join(args.SETTING['data_root'], args.TRAIN['dataset'])
val_root = os.path.join(args.SETTING['data_root'], args.VAL['dataset'])
sal_train = EdgeSalDataset(root=train_root,
                           subset=args.TRAIN['subset'],
                           input_size=args.TRAIN['input_size'],
                           image_dir=args.SETTING['image_dir'],
                           label_dir=args.SETTING['label_dir'],
                           image_set_dir=args.SETTING['image_set_dir'],
                           image_suffix=args.SETTING['image_suffix'],
                           label_suffix=args.SETTING['label_suffix'],
                           transform=True,
                           inference=False)

train_data = mx.gluon.data.DataLoader(sal_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      last_batch='rollover',
                                      num_workers=1)

sal_test = EdgeSalDataset(root=val_root,
                          subset='test_id',
                          input_size=args.TRAIN['input_size'],
                          image_dir=args.SETTING['image_dir'],
                          label_dir=args.SETTING['label_dir'],
                          image_suffix=args.SETTING['image_suffix'],
                          label_suffix=args.SETTING['label_suffix'],
                          transform=False,
                          inference=False)

test_data = mx.gluon.data.DataLoader(sal_test,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     last_batch='keep',
                                     num_workers=2)

if args.begin_epoch > 0:
    net = get_network(args.NETWORK, input_size=args.TRAIN['input_size'], pretrained=False, with_aspp=True)
    net.collect_params().load(args.TRAIN['pretrained'], allow_missing=False, ignore_extra=False)
else:
    # load network with pretrained vgg on bottom-up net
    net = get_network(args.NETWORK, input_size=args.TRAIN['input_size'], pretrained=True, with_aspp=True)
    # init params of the rest layers
    net.aspp.collect_params().initialize(init=mx.init.MSRAPrelu())
    net.refinement1.collect_params().initialize(init=mx.init.MSRAPrelu())
    net.refinement2.collect_params().initialize(init=mx.init.MSRAPrelu())
    net.refinement3.collect_params().initialize(init=mx.init.MSRAPrelu())
    net.mask_conv.collect_params().initialize(init=mx.init.MSRAPrelu())
net.hybridize()
# warm up
net.collect_params().reset_ctx(ctx)

# trainer
if args.train_type == 0:
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
else:
    loss = SigmoidBinaryCrossEntropyLoss_Edge(from_sigmoid=True)

trainer = gluon.Trainer(net.collect_params(),
                        'adam', {'learning_rate': args.TRAIN['lr']})

train(train_data, test_data, net, loss,
    trainer, ctx, save_prefix=save_prefix, logdir=logdir, visual_path=visual_path,
    begin_epoch=args.begin_epoch, num_epochs=args.TRAIN['num_epochs'],
    val_epoch=args.VAL['val_epoch'], num_outputs=1)
