import torch
import torch.nn as nn
from collections import OrderedDict



class ConvBlock(nn.Module):
  def __init__(self, feature_input, feature_output, kernel_size=1, stride=1, padding=0):
    super(ConvBlock, self).__init__()
	
    self.conv = nn.Conv2d(feature_input, feature_output, kernel_size = kernel_size,
                           stride = stride, padding = padding, bias = False)						   
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
	
    hidden_output=feature_output
    self.block1 = ConvBlock(feature_input, hidden_output, 1, 1, 0)
    self.block2 = ConvBlock(hidden_output, feature_output, 3, 1, 1)


  def forward(self, x):
    bottleNeck = x

    out = self.block1(x)
    out = self.block2(out)

    out += bottleNeck
    return out


class C3Block(nn.Module):
  def __init__(self, feature_input, feature_output, n):
    super(C3Block, self).__init__()
 
    hidden_output= int(feature_output*0.5)
    
    self.block1 = ConvBlock(feature_input, hidden_output, 1, 1, 0)
    self.block2 = ConvBlock(feature_input, hidden_output, 1, 1, 0)
   
    self.block3 = ConvBlock(2*hidden_output, feature_output, 1, 1, 0)

    self.model  = nn.Sequential(*(BottleNeck(hidden_output, hidden_output) for _ in range(n)))

  def forward(self, x):
    
    out1 = self.block1(x)
    out2 = self.block2(x)

    out1 = self.model(out1)

    out  = torch.cat((out1, out2), 1)

    out  = self.block3(out)

    return out


class SPPF(nn.Module):
    def __init__(self, feature_input, feature_output, k=5):  
        super().__init__()
        hidden_output = int(feature_input * 0.5)  # hidden channels
        self.block1 = ConvBlock(feature_input, hidden_output, 1, 1)
        self.block2 = ConvBlock(hidden_output * 4, feature_output, 3, [1, 2], 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x  = self.block1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        out= self.block2(torch.cat((x,y1,y2,y3),1))
		
        return out
# ******************************************************************************

class Backbone(nn.Module):


  def __init__(self, params):
    super(Backbone, self).__init__()

    self.drop_prob = params["dropout"]
    self.OS = params["OS"]
    self.blocks = [3, 6, 9, 3, 5]

    # adancime, x, y, z, indice reflectanta
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
      # refacere stride, in cazul in care nu este egal cu cel introdus in fisierul de configuratii
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_os) != self.OS:
          if stride == 2:
            current_os /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == self.OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)

    # input layer
    self.conv1 = nn.Conv2d(self.input_depth, 64, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.silu1 = nn.SiLU()

    # encoder
	#am ramas la modificarea encoder-ului
    self.enc1 = self._make_enc_layer(C3Block, [128, 128], self.blocks[0],
                                     stride=self.strides[0], downsample=True)
    self.enc2 = self._make_enc_layer(C3Block, [256, 256], self.blocks[1],
                                     stride=self.strides[1], downsample=True)
    self.enc3 = self._make_enc_layer(C3Block, [512, 512], self.blocks[2],
                                     stride=self.strides[2], downsample=True)
    self.enc4 = self._make_enc_layer(C3Block, [1024, 1024], self.blocks[3],
                                     stride=self.strides[3], downsample=True)
    self.enc5 = self._make_enc_layer(SPPF, [1024, 1024], self.blocks[4],
                                     stride=self.strides[4], downsample=False)

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 1024

  # make layer useful function
  def _make_enc_layer(self, block, planes, blocks, stride, downsample):
    layers = []

    #  downsample
    if downsample:
      inline = int(planes[0]*0.5)
      layers.append(("conv", nn.Conv2d(inline, planes[1],
                                      kernel_size=3,
                                      stride=[1, stride], dilation=1,
                                      padding=1, bias=False)))
      layers.append(("bn", nn.BatchNorm2d(planes[1])))
      layers.append(("silu", nn.SiLU()))

    #  blocks
    layers.append(("C3",block(planes[0], planes[1], blocks)))

    return nn.Sequential(OrderedDict(layers))

  def run_layer(self, x, layer, skips, os):
    y = layer(x)

    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      #print("OS ",os)
      #print(x.detach().size())
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os

  def forward(self, x):
    # filter input
    x = x[:, self.input_idxs]

    # run cnn
    # store for skip connections
    skips = {}
    os = 1

    # first layer
    x, skips, os = self.run_layer(x, self.conv1, skips, os)
    x, skips, os = self.run_layer(x, self.bn1, skips, os)
    x, skips, os = self.run_layer(x, self.silu1, skips, os)



    # all encoder blocks with intermediate dropouts
    x, skips, os = self.run_layer(x, self.enc1, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc2, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc4, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc5, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)

    return x, skips

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth




