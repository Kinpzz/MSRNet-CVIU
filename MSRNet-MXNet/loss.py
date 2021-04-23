from mxnet.gluon.block import HybridBlock
from mxnet.gluon.loss import Loss

class SigmoidBinaryCrossEntropyLoss_Edge(Loss):
    
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss_Edge, self).__init__(weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid
        
    def hybrid_forward(self, F, pred, label, sample_weight=None):

        if not self._from_sigmoid:
            # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
            loss = F.relu(pred) - pred * label + F.Activation(-F.abs(pred), act_type='softrelu')
        else:
            loss = -(F.log(pred+1e-12)*label + F.log(1.-pred+1e-12)*(1.-label))
        
        weight = F.zeros(label.shape)
        for i, label_each in enumerate(label):
            # beta = count_neg / (count_neg + count_pos)
            beta = 1 - F.mean(label_each)
            # label * beta + (1. - label)*(1 - beta) # pos weight + neg weight
            weight[i] = 1 - beta + (2 * beta - 1) * label_each

        loss = F.broadcast_mul(loss, weight)
        
        return F.mean(loss, axis=self._batch_axis, exclude=True)