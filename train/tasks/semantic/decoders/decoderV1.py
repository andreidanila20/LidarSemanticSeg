import torch
import torch.nn as nn
from collections import OrderedDict



class ConvBlock(nn.Module):
  def __init__(self, c1, c2, kernel_size=1, stride=1, padding=0):
    super(ConvBlock, self).__init__()
	
    self.conv = nn.Conv2d(c1, c2, kernel_size = kernel_size,
                           stride = stride, padding = padding, bias = False)						   
    self.bn = nn.BatchNorm2d(c2)
    self.silu = nn.SiLU()
  

  def forward(self, x):

    out = self.conv(x)
    out = self.bn(out)
    out = self.silu(out)

    return out

class BottleNeck(nn.Module):
  def __init__(self, c1, c2):
    super(BottleNeck, self).__init__()
	
    c_=c2
    self.block1 = ConvBlock(c1, c_, 1, 1, 0)
    self.block2 = ConvBlock(c_, c2, 3, 1, 1)


  def forward(self, x):

    out=self.block1(x)
    out=self.block2(out)
    return out
	
class C3Block(nn.Module):
  def __init__(self, c1, c2):
    super(C3Block, self).__init__()

    c_= int(c1*0.5)

    
    self.block1 = ConvBlock(c1, c_, 1, 1, 0)
    self.block2 = ConvBlock(c1, c_, 1, 1, 0)

    self.bottleNeck=BottleNeck(c_, c_)
	
    self.block3 = ConvBlock(2*c_, c2, 1, 1, 0)
    


  def forward(self, x):
    
    out1 = self.block1(x)
    out2 = self.block2(x)

    out1 = self.bottleNeck(out1)

    out  = torch.cat((out1,out2),1)

    out  = self.block3(out)

    return out



# ******************************************************************************

class Decoder(nn.Module):


  def __init__(self, params, stub_skips, OS=32, feature_depth=1024):
    super(Decoder, self).__init__()

    self.drop_prob = params["dropout"]
    self.backbone_OS = OS
    self.backbone_feature_depth = feature_depth


    # stride play
    #aici e posibil sa putem face cateva modificari
    self.strides = [2, 2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Decoder original OS: ", int(current_os))
    # redo strides according to needed stride
    for i, stride in enumerate(self.strides):
      if int(current_os) != self.backbone_OS:
        if stride == 2:
          current_os /= 2
          self.strides[i] = 1
        if int(current_os) == self.backbone_OS:
          break
    print("Decoder new OS: ", int(current_os))
    print("Decoder strides: ", self.strides)

    # decoder
    self.dec5 = self._make_dec_layer(C3Block,
                                     [self.backbone_feature_depth, 1024, 0],
                                      just_upsample=True)
    self.dec4 = self._make_dec_layer(C3Block, [1024, 512, 0], 
									  just_upsample=True)
    self.dec3 = self._make_dec_layer(C3Block, [1024, 512, 256], 
									  just_upsample=False)
    self.dec2 = self._make_dec_layer(C3Block, [512,  256, 128],  
									  just_upsample=False)
    self.dec1 = self._make_dec_layer(C3Block, [256,  128, 64],  
									  just_upsample=False)


    self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]


    self.dropout = nn.Dropout2d(self.drop_prob)


    self.last_channels = 64

  def _make_dec_layer(self, block, planes, just_upsample):
    layers = []


    if just_upsample or planes[2]==0:
      #upsample
      layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                              kernel_size=[1, 4], stride=[1, 2],
                              padding=[0, 1])))
      layers.append(("bn", nn.BatchNorm2d(planes[1])))
      layers.append(("silu", nn.SiLU()))
      
    
    else:
    
      #  blocks
      layers.append(("C3", block(planes[0], planes[1])))
        
      #  upsample  
      layers.append(("upconv", nn.ConvTranspose2d(planes[1], planes[2],
                              kernel_size=[1, 4], stride=[1, 2],
                              padding=[0, 1])))
      layers.append(("bn", nn.BatchNorm2d(planes[2])))
      layers.append(("silu", nn.SiLU()))

    

    return nn.Sequential(OrderedDict(layers))

  def run_layer(self, x, layer, skips, os):
    feats = layer(x)  # up


    #print("X ",x.size())
    #print("Feats ", feats.size())
    
    #pentru layerele la care facem upsample, adunam la tensorii obtinuti in urma
    #aplicarii layerului, tensorii din backbone care au acelasi numar de 
    #caracteristici 
    if os>1:
      if feats.shape[1] in [512, 256, 128]:
        os //= 2   
        #concat with features from encoder 
        feats =torch.cat((feats,skips[os].detach()),1) 
        
    x = feats
    return x, skips, os

  def forward(self, x, skips):
    os = 16
    #print("Decoder")
    # run layers
    x, skips, os = self.run_layer(x, self.dec5, skips, os)
    #print("Decoder5")
    x, skips, os = self.run_layer(x, self.dec4, skips, os)
    #print("Decoder4")
    x, skips, os = self.run_layer(x, self.dec3, skips, os)
    #print("Decoder3")
    x, skips, os = self.run_layer(x, self.dec2, skips, os)
    #print("Decoder2")
    x, skips, os = self.run_layer(x, self.dec1, skips, os)
    #print("Decoder1")

    x = self.dropout(x)

    #print(x.size())

    return x

  def get_last_depth(self):
    return self.last_channels
