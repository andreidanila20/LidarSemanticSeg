import torch
import torch.nn as nn
from collections import OrderedDict



# conv2d, batchNorm, Mish
class CBMBlock(nn.Module):
    def __init__(self, input_features, output_features, kernel_size=[1], stride=[1], padding=[0]):
        super(CBMBlock, self).__init__()

        self.conv = nn.Conv2d(input_features, output_features, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(output_features)
        self.mish = nn.Mish()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.mish(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, input_features, output_features):
        super(BottleNeck, self).__init__()

        hidden_features = output_features

        self.block1 = CBMBlock(input_features, hidden_features, 1, 1, 0)
        # deoarece kernel_size==3, trebuie sa setam padding=1 pentru a adăuga zerouri
        self.block2 = CBMBlock(hidden_features, output_features, 3, 1, 1)

    def forward(self, x):
        bottleNeck = x

        out = self.block1(x)
        out = self.block2(out)

        out += bottleNeck
        return out


# n - numarul bottleneck-urilor pentru un layer CSP
class CSPBlock(nn.Module):
    def __init__(self, input_features, output_features, n):
        super(CSPBlock, self).__init__()

        # micsoram numarul caracteristicilor cu un factor de 2, deoarece in bottleNeck vom utiliza o convolutie 3x3
        # utilizand caracteristicile/2, obtinem un timp computational mai bun si un model cu mai putini parametri
        hidden_features = int(output_features * 0.5)

        self.block1 = CBMBlock(input_features, hidden_features, 1, 1, 0)
        self.block2 = CBMBlock(input_features, hidden_features, 1, 1, 0)
        self.block3 = CBMBlock(2 * hidden_features, output_features, 1, 1, 0)

        self.model = nn.Sequential(*(BottleNeck(hidden_features, hidden_features) for _ in range(n)))

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)

        out1 = self.model(out1)

        out = torch.cat((out1, out2), 1)

        out = self.block3(out)

        return out


class SPPF(nn.Module):
    # Extragem caracteristicile cele mai importante din imagine
    def __init__(self, input_features, output_features, k=5):  
        super().__init__()
        hidden_features = int(input_features * 0.5)  # hidden channels
        self.cv1 = CBMBlock(input_features, hidden_features, 1, 1, 0)

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=int(k * 0.5))

        self.cv2 = CBMBlock(hidden_features * 4, output_features, 3, 1, 1)

    def forward(self, x):
        # convolutie 1x1 folosita pentru subesantionarea hartii de caracteristici
        x = self.cv1(x)

        # max pooling
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # concatenare
        out = self.cv2(torch.cat((x, y1, y2, y3), 1))

        return out


#----------------------------------------------------------------------------------------


class Backbone(nn.Module):
	#Clasa principala
    def __init__(self, params):
        super(Backbone, self).__init__()
        
        self.drop_prob = params["dropout"]
        self.OS = params["OS"]
        self.blocks = [1, 2, 8, 8, 4]
        
        #adancime, x, y, z, indice reflectanta 
        self.input_depth = 5
        self.input_idxs = [0, 1, 2, 3, 4]
        
        print("Depth of backbone input = ", self.input_depth)

        # la fiecare nivel din coloana latimea se imparte la 2 
        self.strides = [2, 2, 2, 2, 2]
        
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)

   
        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)
        else:
            #refacere stride, in cazul in care nu este egal cu cel introdus in fisierul de configuratii
            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)

        # strat intrare
        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.mish1 = nn.Mish()

        # straturi de codificare 
        self.enc1 = self._make_enc_layer(CSPBlock, [32, 64], self.blocks[0],
                                         stride=self.strides[0])
        self.enc2 = self._make_enc_layer(CSPBlock, [64, 128], self.blocks[1],
                                         stride=self.strides[1])
        self.enc3 = self._make_enc_layer(CSPBlock, [128, 256], self.blocks[2],
                                         stride=self.strides[2])
        self.enc4 = self._make_enc_layer(CSPBlock, [256, 512], self.blocks[3],
                                         stride=self.strides[3])
        self.enc5 = self._make_enc_layer(CSPBlock, [512, 1024], self.blocks[4],
                                         stride=self.strides[4])

        # aplicare strat SPPF
        self.sppf = SPPF(1024, 1024, k=5)

        # declarare strat de pierderi pentru reducerea riscului de supra-antrenare
        self.dropout = nn.Dropout2d(self.drop_prob)

        # numarul final de caracteristici
        self.last_channels = 1024

    # construim un strat de codificare 
    def _make_enc_layer(self, block, features, blocks, stride):
        layers = []

        #  subesantionare
        hidden_feature = int(features[0] * 2)
        layers.append(("conv", nn.Conv2d(features[0], hidden_feature,
                                         kernel_size=3,
                                         stride=[1, stride], dilation=1,
                                         padding=1, bias=False)))
        layers.append(("bn", nn.BatchNorm2d(hidden_feature)))
        layers.append(("mish", nn.Mish()))

        layers.append(("CSP", block(hidden_feature, features[1], blocks)))

        return nn.Sequential(OrderedDict(layers))

	#rulare strat
    def run_layer(self, x, layer, skips, os):
        y = layer(x)

        #print("X shape: ", x.shape)
        #print("Y shape: ", y.shape)

        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            #print("OS ", os)
            #print(x.detach().size())
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x):
        # filter input
        x = x[:, self.input_idxs]

        
        # aici stocam tensorii pentru conexiunle laterale 
        skips = {}
        os = 1
        #print("X before first", x.size())
       
        x, skips, os = self.run_layer(x, self.conv1, skips, os)
        x, skips, os = self.run_layer(x, self.bn1, skips, os)
        x, skips, os = self.run_layer(x, self.mish1, skips, os)

        #print("X after first", x.size())
		
        x, skips, os = self.run_layer(x, self.enc1, skips, os)
        #print("X after enc1", x.size())
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc2, skips, os)
        #print("X after enc2", x.size())
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc3, skips, os)
        #print("X after enc3", x.size())
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc4, skips, os)
        #print("X after enc4", x.size())
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc5, skips, os)
        #print("X after enc5", x.size())
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.sppf, skips, os)
        #print("X after SPPF", x.size())
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        return x, skips

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth