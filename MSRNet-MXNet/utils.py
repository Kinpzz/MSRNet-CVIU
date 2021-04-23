
import os
import cv2
import random
import numpy as np
from math import exp
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet.gluon import nn
from time import time
from tqdm import tqdm

from metric import EdgeSal_Visual, EdgeSalMetric_MAE, EdgeSalMetric_Acc, EdgeSalMetric_IoU
from mxboard import SummaryWriter

import sys
sys.setrecursionlimit(10000)

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx, even_split=False),
            gluon.utils.split_and_load(label, ctx, even_split=False),
            data.shape[0])

def train(train_data, test_data, net, loss_func, trainer, ctx, num_epochs, save_prefix,
            num_outputs, logdir, begin_epoch=0, print_batches=None, visual_path=None, val_epoch=1):
    """Train a network"""
    print("Start training on ", ctx)
    # log
    sw = SummaryWriter(logdir=logdir, flush_secs=5)

    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    # metric
    eval_metrics = [] # list of compositeEvalMetric
    for i in range(num_outputs):
        eval_metric = mx.metric.CompositeEvalMetric()
        eval_metric.add(EdgeSalMetric_MAE())
        eval_metric.add(EdgeSalMetric_Acc())
        eval_metric.add(EdgeSalMetric_IoU())
        if visual_path is not None:
            eval_metric.add(EdgeSal_Visual(visual_path)) # visual
        eval_metrics.append(eval_metric)
    min_loss = 1000
    min_loss_epoch = 0
    for epoch in range(num_epochs):
        n, m = 0.0, 0.0
        train_losses = np.zeros(num_outputs)
        test_losses = np.zeros(num_outputs)
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        if isinstance(test_data, mx.io.MXDataIter):
            test_data.reset()
        start = time()
        # train
        if True:
            for iter, batch in enumerate(tqdm(train_data, unit='mini-batches')):
                data, label, batch_size = _get_batch(batch, ctx)
                losses = []
                losses_sum = []
                for X, y in zip(data, label): # for each mx device
                    with autograd.record():
                        masks = net(X)
                        loss = [loss_func(masks[0], y)]
                        losses.append(loss)
                        losses_sum.append(loss)
                for loss_sum in losses_sum:
                    #loss_sum.backward() # shape (batch_size, )
                    autograd.backward(loss_sum)
                nd.waitall()
                for loss in losses:
                    for j, l in enumerate(loss):
                        train_losses[j] += l.sum().asscalar() # 等待后端多gpu并行计算结束，再进行同步
                trainer.step(batch_size) # parameter update
                n += batch_size
                if print_batches and (iter+1) % print_batches == 0:
                    print("Batch %d." % (n))
                    for j, l in enumerate(train_losses):
                        print("Loss %d: %f\n" % (j, l / n))

            print("Epoch %d. Training Time %.1f sec" % (epoch+begin_epoch+1, time() - start))
            for j, l in enumerate(train_losses):
                loss = l / n
                print("Train-Loss %d: %f" % (j, loss))
                sw.add_scalar(tag='Train_loss-Mask%d'%(j+1), value=loss, global_step=epoch+begin_epoch+1)
        if epoch % val_epoch == 0:
            start = time()
            # val test
            for iter, batch in enumerate(tqdm(test_data, unit='mini-batches')):
                data, label, batch_size = _get_batch(batch, ctx)
                losses = []
                masks_all = []
                labels = []
                for X, y in zip(data, label): # multi devices
                    masks = net(X)
                    loss = [loss_func(masks[0], y)]
                    masks_all.append(masks)
                    labels.append(y)
                    losses.append(loss)
                nd.waitall()                                  # 等待后端多gpu并行计算结束，再进行同步
                # 同步 loss
                for loss in losses:
                    for j, l in enumerate(loss):
                        test_losses[j] += l.sum().asscalar()
                # 同步 metric
                for masks, label in zip(masks_all, labels):           # 多gpu
                    #for j, mask in enumerate(masks):                  # 多输出
                    eval_metrics[0].update(labels=label, preds=masks[0])   # 单batch
                m += batch_size
            print("Epoch %d. Validation Time %.1f sec" % (epoch+begin_epoch+1, time() - start))
            for j, l in enumerate(test_losses):
                c_loss = l / m
                if j == num_outputs - 1 and c_loss < min_loss:
                    min_loss = c_loss
                    min_loss_epoch = epoch + begin_epoch + 1
                print("Test-Loss %d %f" % (j, c_loss))
                sw.add_scalar(tag='Test_loss-Mask%d'%(j+1), value=c_loss, global_step=epoch+begin_epoch+1)

            for j, eval_metric in enumerate(eval_metrics):
                name_val = eval_metric.get()
                for name, val in zip(name_val[0], name_val[1]):
                    if name != 'Visual':
                        sw.add_scalar(tag='Test_%s-Mask%d'%(name, j+1), value=val, global_step=epoch+begin_epoch+1)
                        print("Mask %d %s: %f" % (j+1, name, val))

            # reset metric
            for eval_metric in eval_metrics:
                eval_metric.reset()
            # save params
            net.collect_params().save('%s-%04d.params'%(save_prefix, epoch+begin_epoch+1))
    print("min loss:%f at epoch %d"%(min_loss, min_loss_epoch))
    sw.close()


def predict(test_data, img_info, net, ctx, result_path, pad_zeros=False):
    """Prediction"""
    print("Start Predicting on ", ctx)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    start = time()
    iter = 0;
    for batch in tqdm(test_data, unit='mini-batches'):
        batch_size = batch.shape[0]
        data = gluon.utils.split_and_load(batch, ctx, even_split=False)
        masks = [net(X) for X in data] # multi devices
        nd.waitall()                   # 等待后端多gpu并行计算结束，再进行同步
        for mask_each in masks:        # 多gpu
            for mask_ch in mask_each:      # 多输出
                device_batch = mask_ch.shape[0]
                for n, mask in enumerate(mask_ch):       # 多batch [batch_size, 1, height, width]
                    img_width = img_info[iter+n]['size'][0]
                    img_height = img_info[iter+n]['size'][1]
                    img_name = img_info[iter+n]['name']
                    save_array2image(result_path, img_name, mask[0,:,:], img_width, img_height, pad_zeros)
            iter += device_batch
    print("Predition time %f s" % (time() - start))

def save_array2image(save_path, name, array, height, width, pad_zeros):
    x = (array * 255).asnumpy().astype('uint8')
    if pad_zeros:
        x = x[0:height, 0:width]
    else:
        x = cv2.resize(x, (width, height))
    cv2.imwrite(os.path.join(save_path, name+'.png'), x)
