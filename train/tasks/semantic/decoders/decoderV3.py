import torch
import torch.nn as nn
from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(self, feature_input, feature_output, kernel_size=1, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(feature_input, feature_output, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(feature_output)
        self.silu = nn.SiLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.silu(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, feature_input, feature_output):
        super(BottleNeck, self).__init__()

        hidden_output = feature_output
        self.block1 = ConvBlock(feature_input, hidden_output, 1, 1, 0)
        # deoarece kernel_size==3, trebuie sa setam padding=1 pentru a adăuga zerouri
        self.block2 = ConvBlock(hidden_output, feature_output, 3, 1, 1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class C3Block(nn.Module):
    def __init__(self, feature_input, feature_output):
        super(C3Block, self).__init__()

        hidden_output = int(feature_input * 0.5)


        self.block1 = ConvBlock(feature_input, hidden_output, 1, 1, 0)
        self.block2 = ConvBlock(feature_input, hidden_output, 1, 1, 0)
        self.block3 = ConvBlock(2 * hidden_output, feature_output, 1, 1, 0)

        self.model = BottleNeck(hidden_output, hidden_output)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)

        out1 = self.model(out1)

        out = torch.cat((out1, out2), 1)

        out = self.block3(out)

        return out


# ******************************************************************************

class Decoder(nn.Module):


    def __init__(self, params, stub_skips, OS=32, feature_depth=1024):
        super(Decoder, self).__init__()

        self.drop_prob = params["dropout"]
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth



        self.strides = [2, 2, 2, 2, 2]

        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))
   
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)


        self.dec5 = self._make_dec_layer(C3Block,
                                         [self.backbone_feature_depth, 1024],
                                         just_upsample=False)
        self.dec4 = self._make_dec_layer(C3Block, [1024, 512],
                                         just_upsample=False)
        self.dec3 = self._make_dec_layer(C3Block, [512, 256],
                                         just_upsample=False)
        self.dec2 = self._make_dec_layer(C3Block, [256, 128],
                                         just_upsample=False)
        self.dec1 = self._make_dec_layer(C3Block, [128, 64],
                                         just_upsample=True)


        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]


        self.dropout = nn.Dropout2d(self.drop_prob)

        self.last_channels = 64

    def _make_dec_layer(self, block, planes, just_upsample):
        layers = []

        # upsample
        layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                    kernel_size=[1, 4], stride=[1, 2],
                                                    padding=[0, 1])))
        layers.append(("bn", nn.BatchNorm2d(planes[1])))
        layers.append(("silu", nn.SiLU()))

        if not just_upsample:
            #  blocks
            layers.append(("C3", block(planes[1], planes[1])))

        return nn.Sequential(OrderedDict(layers))


    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up

        if feats.shape[3] > x.shape[3]:
            # pentru layerele la care facem upsample, adunam la tensorii obtinuti in urma
            # aplicarii layerului, tensorii din backbone care au acelasi numar de
            # caracteristici
            if os > 1:
                os //= 2
                # concat with features from encoder
                feats = feats + skips[os].detach()

        x = feats
        return x, skips, os


    def forward(self, x, skips):
        os = 32

        # run layers
        #print("Entry", x.size())
        x, skips, os = self.run_layer(x, self.dec5, skips, os)
        #print("D5",x.size())
        x, skips, os = self.run_layer(x, self.dec4, skips, os)
        #print("D4",x.size())
        x, skips, os = self.run_layer(x, self.dec3, skips, os)
        #print("D3",x.size())
        x, skips, os = self.run_layer(x, self.dec2, skips, os)
        #print("D2",x.size())
        x, skips, os = self.run_layer(x, self.dec1, skips, os)
        #print("D1",x.size())

        x = self.dropout(x)

        return x


    def get_last_depth(self):
        return self.last_channels
