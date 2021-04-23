import mxnet as mx
import numpy as np
import cv2
import os

def save_array2image(save_path, name, array):
    x = (array * 255).asnumpy().astype('uint8')
    cv2.imwrite(os.path.join(save_path, name+'.png'), x)
       
class EdgeSalMetric_MAE(mx.metric.EvalMetric):
    def __init__(self, **kwargs):
        super(EdgeSalMetric_MAE, self).__init__('MAE', **kwargs)
    def update(self, labels, preds):
        labels, preds = mx.metric.check_label_shapes(labels, preds, False, True)
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()
            
            pred_idx = pred[0,:,:]
            label_idx = label[0,:,:]
            if len(label_idx.shape) == 1:
                label_idx = label_idx.reshape(label_idx.shape[0], 1)
            if len(pred_idx.shape) == 1:
                pred_idx = pred_idx.reshape(pred_idx.shape[0], 1)
            self.sum_metric += np.abs(label_idx - pred_idx).mean()
            self.num_inst += 1
                
class EdgeSalMetric_Acc(mx.metric.EvalMetric):
    def __init__(self, **kwargs):
        super(EdgeSalMetric_Acc, self).__init__('Acc', **kwargs)
    def update(self, labels, preds):
        labels, preds = mx.metric.check_label_shapes(labels, preds, False, True)
        for label, pred in zip(labels, preds):
            
            pred_idx = pred[0,:,:] >= 0.5
            label_idx = label[0,:,:] > 0
            # float -> int
            pred_idx = pred_idx.asnumpy().astype('int32')
            label_idx = label_idx.asnumpy().astype('int32')
            pred_idx= pred_idx.flat
            label_idx = label_idx.flat
            self.sum_metric+= (pred_idx == label_idx).sum()
            self.num_inst+= len(pred_idx)

class EdgeSalMetric_IoU(mx.metric.EvalMetric):
    def __init__(self, **kwargs):
        super(EdgeSalMetric_IoU, self).__init__('IoU', **kwargs)
    def update(self, labels, preds):
        labels, preds = mx.metric.check_label_shapes(labels, preds, False, True)

        for label, pred in zip(labels, preds):
            pred_idx = pred[0,:,:] >= 0.5
            label_idx = label[0,:,:] > 0
            pred_idx = pred_idx.asnumpy().astype('int32')
            label_idx = label_idx.asnumpy().astype('int32')
            a_and_b = np.sum((pred_idx == label_idx) * pred_idx)
            a_or_b = np.sum((pred_idx + label_idx)>0)
            iou = a_and_b / a_or_b if a_or_b > 0 else 0
            self.sum_metric += iou
            self.num_inst += 1
                
class EdgeSal_Visual(mx.metric.EvalMetric):
    def __init__(self, visual_path, **kwargs):
        super(EdgeSal_Visual, self).__init__('Visual', **kwargs)
        self.iter = 0
        self.visual_path = visual_path
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)
        self.num_inst = 0
    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            save_array2image(self.visual_path, '%d_gt'%(self.iter), label[0])
            save_array2image(self.visual_path, '%d_Mask'%(self.iter), pred[0])
            self.iter += 1
    def reset(self):
        self.iter = 0
