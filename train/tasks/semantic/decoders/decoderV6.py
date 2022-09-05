import torch
import torch.nn as nn




# conv2D, batchNorm, mish
class CBRBlock(nn.Module):
    def __init__(self, input_features, output_features, kernel_size=1, stride=1, padding=0):
        super(CBRBlock, self).__init__()

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
	#bloc rezidual 
    def __init__(self, input_features, output_features):
        super(BottleNeck, self).__init__()

        hidden_features = output_features * 2

        self.block1 = CBRBlock(input_features, hidden_features, 1, 1, 0)
        self.block2 = CBRBlock(hidden_features, output_features, 3, 1, 1)

    def forward(self, x):
        bottleNeck = x

        out = self.block1(x)
        out = self.block2(out)

        out += bottleNeck
        return out


class UpsampleBlock(nn.Module):
	#bloc de supraesantionare 
    def __init__(self, input_features, output_features, kernel_size=[1, 4], stride=[1, 2], padding=[0, 1]):
        super(UpsampleBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(input_features, output_features,
                                          kernel_size=kernel_size, stride=stride,
                                          padding=padding)
        self.bn = nn.BatchNorm2d(output_features)
        self.mish = nn.Mish()

    def forward(self, x):
        out = self.up_conv(x)
        out = self.bn(out)
        out = self.mish(out)

        return out


class DecoderBlock(nn.Module):
	#blocul principal al gatului 
  def __init__(self, input_features, output_features, n):
    super(DecoderBlock, self).__init__()
	
    hidden_features=int(input_features*0.5)
    self.upsample = UpsampleBlock(input_features, hidden_features)
    self.model = nn.Sequential(*(BottleNeck(hidden_features, output_features) for _ in range(n)))
    


  def forward(self, x):

    out = self.upsample(x)
    out = self.model(out)

    return out


# ******************************************************************************

class Decoder(nn.Module):
    

    def __init__(self, params, stub_skips, OS=32, feature_depth=1024):
        super(Decoder, self).__init__()

        self.drop_prob = params["dropout"]
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth

        # stride play
        # aici e posibil sa putem face cateva modificari
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

       
        # straturile principale ale gatului 
        self.dec5 = DecoderBlock(self.backbone_feature_depth, 512, 2)
        self.dec4 = DecoderBlock(512, 256, 2)
        self.dec3 = DecoderBlock(256, 128, 2)
        self.dec2 = DecoderBlock(128, 64, 2)
        self.dec1 = DecoderBlock(64, 32, 2)

        
        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]

        # strat de pierdere care va fi pus la final 
        self.dropout = nn.Dropout2d(self.drop_prob)

        # dimensiunea hartii de caracteristici de la finalul gatului 
        self.last_channels = 32

    def run_layer(self, x, layer, skips, os):
        feats = layer(x) 
        if os > 1:
            if x.shape[3] < feats.shape[3]:
                os //= 2

                encoder_value = skips[os].detach()

                feats = feats + encoder_value

        x = feats
        return x, skips, os

    def forward(self, x, skips):
        os = 32
        #print("Decoder")
        # run layers

        x, skips, os = self.run_layer(x, self.dec5, skips, os)
       # print("Decoder5")
        x, skips, os = self.run_layer(x, self.dec4, skips, os)
        #print("Decoder4")
        x, skips, os = self.run_layer(x, self.dec3, skips, os)
       # print("Decoder3")
        x, skips, os = self.run_layer(x, self.dec2, skips, os)
        #print("Decoder2")
        x, skips, os = self.run_layer(x, self.dec1, skips, os)
        #print("Decoder1")

        x = self.dropout(x)

        # print(x.size())

        return x

    def get_last_depth(self):
        return self.last_channels
