from mxnet.gluon import nn
from mxnet import gluon
import mxnet as mx
import numpy as np

# residual block
class Bottleneck(nn.HybridBlock):

    def __init__(self, channels, strides, params, downsample=False, dilation=1):
        super(Bottleneck, self).__init__()
        self.body = nn.HybridSequential(prefix="")
        self.body.add(nn.Conv2D(channels=channels // 4, kernel_size=1, strides=strides, params=params.body[0].params))
        self.body.add(nn.BatchNorm(params=params.body[1].params))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels=channels // 4, kernel_size=3, strides=1, padding=dilation, dilation=dilation, use_bias=False, in_channels=channels // 4, params=params.body[3].params))
        self.body.add(nn.BatchNorm(params=params.body[4].params))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1, params=params.body[6].params))
        self.body.add(nn.BatchNorm(params=params.body[7].params))
        self.is_downsample = downsample
        if self.is_downsample:
            self.downsample = nn.HybridSequential()
            self.downsample.add(nn.Conv2D(channels=channels, kernel_size=1, strides=strides, use_bias=False, params=params.downsample[0].params))
            self.downsample.add(nn.BatchNorm(params=params.downsample[1].params))

    def hybrid_forward(self, F, x):
        if self.is_downsample:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.body(x)
        x = F.Activation(residual + x, act_type="relu")
        return x


class Bottleneck_skip(nn.HybridBlock):

    def __init__(self, channels, strides=1, downsample=False, dilation=1):
        super(Bottleneck_skip, self).__init__()
        self.body = nn.HybridSequential(prefix="")
        self.body.add(nn.Conv2D(channels=channels // 2, kernel_size=1, strides=strides))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels=channels // 2, kernel_size=3, strides=1, padding=dilation, dilation=dilation, use_bias=False, in_channels=channels // 2))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        self.is_downsample = downsample
        if self.is_downsample:
            self.downsample = nn.HybridSequential()
            self.downsample.add(nn.Conv2D(channels=channels, kernel_size=1, strides=strides, use_bias=False))
            self.downsample.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):
        if self.is_downsample:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.body(x)
        x = F.Activation(residual + x, act_type="relu")
        return x

class ASPP(nn.HybridBlock):

    def __init__(self, input_size=320, OS=16, attention=True):
        super(ASPP, self).__init__()
        self.attention = attention
        self.aspp0 = nn.HybridSequential()
        self.aspp0.add(nn.Conv2D(channels=256, kernel_size=1, strides=1, padding=0))
        self.aspp0.add(nn.BatchNorm())
        if attention:
            self.att_conv0 = nn.HybridSequential()
            self.att_conv0.add(nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
                            nn.BatchNorm(),
                            nn.Activation('relu'),
                            nn.Dropout(0.5))
            if OS == 16:
                self.aspp1, self.att_conv1 = self._make_aspp(6)
                self.aspp2, self.att_conv2 = self._make_aspp(12)
                self.aspp3, self.att_conv3 = self._make_aspp(18)
            elif OS == 8:
                self.aspp1, self.att_conv1 = self._make_aspp(12)
                self.aspp2, self.att_conv2 = self._make_aspp(24)
                self.aspp3, self.att_conv3 = self._make_aspp(36)
        else:
            if OS == 16:
                self.aspp1 = self._make_aspp(6)
                self.aspp2 = self._make_aspp(12)
                self.aspp3 = self._make_aspp(18)
            elif OS == 8:
                self.aspp1 = self._make_aspp(12)
                self.aspp2 = self._make_aspp(24)
                self.aspp3 = self._make_aspp(36)
        self.gap = nn.HybridSequential()
        self.gap_kernel = int(np.ceil(input_size / OS))
        self.gap.add(nn.AvgPool2D(pool_size=self.gap_kernel, strides=1),
                    nn.Conv2D(channels=256, kernel_size=1),
                    nn.BatchNorm())
        self.fire = nn.HybridSequential()
        self.fire.add(nn.Conv2D(channels=256, kernel_size=1),
                    nn.BatchNorm(),
                    nn.Dropout(0.1))

    def hybrid_forward(self, F, x):
        b4 = self.gap(x)
        b4 = F.contrib.BilinearResize2D(data=b4, height=self.gap_kernel, width=self.gap_kernel)
        b0 = self.aspp0(x)
        b1 = self.aspp1(x)
        b2 = self.aspp2(x)
        b3 = self.aspp3(x)
        if self.attention:
            att0 = self.att_conv0(x)
            att1 = self.att_conv1(x)
            att2 = self.att_conv2(x)
            att3 = self.att_conv3(x)
            b0 = F.elemwise_mul(b0, att0)
            b1 = F.elemwise_mul(b1, att1)
            b2 = F.elemwise_mul(b2, att2)
            b3 = F.elemwise_mul(b3, att3)
            #b4 = F.elemwise_mul(b4, att)
        x = F.concat(*[b4, b0, b1, b2, b3], dim=1)
        return self.fire(x)

    def _make_aspp(self, dilation):
        aspp = nn.HybridSequential()
        aspp.add(nn.Conv2D(channels=256, kernel_size=3, strides=1, dilation=dilation, padding=dilation),
                nn.BatchNorm())
        if self.attention:
            att_conv = nn.HybridSequential()
            att_conv.add(nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
                            nn.BatchNorm(),
                            nn.Activation('relu'),
                            nn.Dropout(0.5))
            return aspp, att_conv
        else:
            return aspp

class resnet50FCN(nn.HybridBlock):

    def __init__(self, pretrained, input_size=320, OS=16, with_aspp=True):
        super(resnet50FCN, self).__init__()
        resnet = gluon.model_zoo.vision.resnet50_v1(pretrained=pretrained)
        _resnet_block3 = resnet.features[6]
        _resnet_block4 = resnet.features[7]
        self.features = nn.HybridSequential()
        self.input_size = input_size
        if OS == 8:
            dilation = [2, 4]
            block3_stride = 1
        elif OS == 16:
            dilation = [1, 2]
            block3_stride = 2

        for layer in resnet.features[:6]:
            self.features.add(layer)
        with self.name_scope():
            _block_3 = nn.HybridSequential()
            _block_3.add(Bottleneck(channels=1024, strides=block3_stride, dilation=dilation[0], params=_resnet_block3[0], downsample=True),
                        Bottleneck(channels=1024, strides=1, dilation=dilation[0], params=_resnet_block3[1]),
                        Bottleneck(channels=1024, strides=1, dilation=dilation[0], params=_resnet_block3[2]),
                        Bottleneck(channels=1024, strides=1, dilation=dilation[0], params=_resnet_block3[3]),
                        Bottleneck(channels=1024, strides=1, dilation=dilation[0], params=_resnet_block3[4]),
                        Bottleneck(channels=1024, strides=1, dilation=dilation[0], params=_resnet_block3[5]))
            _block_4 = nn.HybridSequential()
            _block_4.add(Bottleneck(channels=2048, strides=1, dilation=dilation[1], params=_resnet_block4[0], downsample=True),
                        Bottleneck(channels=2048, strides=1, dilation=dilation[1], params=_resnet_block4[1]),
                        Bottleneck(channels=2048, strides=1, dilation=dilation[1], params=_resnet_block4[2]))
            self.features.add(_block_3, _block_4)
            if with_aspp:
                self.features.add(ASPP(input_size=input_size, OS=OS))
    #     upsampling = nn.Conv2DTranspose(channels=1, kernel_size=32, strides=16, padding=8, weight_initializer=mx.init.Bilinear(), use_bias=False)
    #     upsampling.collect_params().setattr("lr_mult", 0.0)
    #     self.mask_conv = nn.HybridSequential()
    #     self.mask_conv.add(upsampling,
    #                         nn.BatchNorm(),
    #                         nn.Activation('sigmoid'))
    # def hybrid_forward(self, F, x):
    #     x = self.features(x)
    #     x = self.mask_conv(x)
    #     return [x]

class deeplabv3plus(nn.HybridBlock):
    def __init__(self, pretrained, input_size=320, OS=16):
        super(deeplabv3plus, self).__init__()
        self.input_size=input_size
        self.backbone = resnet50FCN(pretrained=pretrained, input_size=input_size, OS=OS)
        self.block1_conv = self.backbone.features[:5]
        self.encoder = self.backbone.features[5:]
        self.aspp = self.backbone.aspp
        with self.name_scope():
            self.skip_conv = nn.HybridSequential()
            self.skip_conv.add(nn.Conv2D(channels=48, kernel_size=1, strides=1, padding=0, use_bias=False),
                                nn.BatchNorm(),
                                nn.Activation('relu'))
            self.decoder_conv = nn.HybridSequential()
            self.decoder_conv.add(nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, use_bias=False, in_channels=304),
                                nn.BatchNorm(),
                                nn.Activation('relu'),
                                nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, use_bias=False, in_channels=256),
                                nn.BatchNorm(),
                                nn.Activation('relu'),
                                nn.Conv2D(channels=1, kernel_size=1, strides=1, padding=0, in_channels=256))
    def hybrid_forward(self, F, x):
        block1 = self.block1_conv(x)
        x = self.encoder(block1)
        aspp = self.aspp(x)
        aspp = F.contrib.BilinearResize2D(data=aspp, height=int(self.input_size/4), width=int(self.input_size/4))
        skip = self.skip_conv(block1)
        x = F.concat(*[aspp, skip], dim=1)
        x = self.decoder_conv(x)
        x = F.contrib.BilinearResize2D(data=x, height=self.input_size, width=self.input_size)
        x = F.Activation(x, act_type='sigmoid')
        return x

class refinement_block(nn.HybridBlock):
    def __init__(self):
        super(refinement_block, self).__init__()
        with self.name_scope():
            self.skip_conv = Bottleneck_skip(96, downsample=True)
            self.refine_conv = nn.HybridSequential()
            self.refine_conv.add(nn.Conv2D(channels=128, kernel_size=3, strides=1, padding=1, use_bias=False),
                                nn.BatchNorm(),
                                nn.Activation('relu'))
    def hybrid_forward(self, F, td, bu):
        td = self.skip_conv(td)
        x = F.concat(*[bu, td], dim=1)
        x = self.refine_conv(x)
        return x

class ssrn_resnet50(nn.HybridBlock):
    def __init__(self, pretrained, input_size=320, OS=16, with_aspp=True):
        super(ssrn_resnet50, self).__init__()
        self.input_size=input_size
        self.backbone = resnet50FCN(pretrained=pretrained, input_size=input_size, OS=OS, with_aspp=with_aspp).features
        self.block1_conv = self.backbone[:5]
        self.block2_conv = self.backbone[5]
        self.block3_conv = self.backbone[6]
        self.block4_conv = self.backbone[7]
        self.with_aspp = with_aspp
        if with_aspp:
            self.aspp = self.backbone[8]
        with self.name_scope():
            self.refinement1 = refinement_block()
            self.refinement2 = refinement_block()
            self.refinement3 = refinement_block()
            self.mask_conv = nn.HybridSequential()
            self.mask_conv.add(nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, use_bias=False),
                                nn.BatchNorm(),
                                nn.Activation('relu'),
                                nn.Conv2D(channels=1, kernel_size=1, strides=1, padding=0),
                                nn.Activation('sigmoid'))
    def hybrid_forward(self, F, x):
        # encoder
        block1 = self.block1_conv(x)
        block2 = self.block2_conv(block1)
        block3 = self.block3_conv(block2)
        block4 = self.block4_conv(block3)
        if self.with_aspp:
            bu0 = self.aspp(block4)
        else:
            bu0 = block4
        # decoder
        bu1 = self.refinement1(block3, bu0)
        bu1 = F.contrib.BilinearResize2D(data=bu1, height=int(self.input_size/8), width=int(self.input_size/8))
        bu2 = self.refinement2(block2, bu1)
        bu2 = F.contrib.BilinearResize2D(data=bu2, height=int(self.input_size/4), width=int(self.input_size/4))
        bu3 = self.refinement3(block1, bu2)

        bu3 = F.contrib.BilinearResize2D(data=bu3, height=self.input_size, width=self.input_size)
        x = self.mask_conv(bu3)
        return [x]

def get_network(name, **args):
    if name == 'msrn_resnet50':
        return ssrn_resnet50(**args)
    elif name == 'resnet50FCN':
        return resnet50FCN(**args)
    else:
        raise Exception('No network named %s' % (name))
