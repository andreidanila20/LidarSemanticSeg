import torch
import torch.nn as nn



# conv2D, batchNorm, relu
class CBRBlock(nn.Module):
    def __init__(self, input_features, output_features, kernel_size=1, stride=1, padding=0):
        super(CBRBlock, self).__init__()

        self.conv = nn.Conv2d(input_features, output_features, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, input_features, output_features, kernel_size=[1, 4], stride=[1, 2], padding=[0, 1]):
        super(UpsampleBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(input_features, output_features,
                                          kernel_size=kernel_size, stride=stride,
                                          padding=padding)
        self.bn = nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.up_conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
  def __init__(self, input_feature, output_feature):
    super(DecoderBlock, self).__init__()
	
    hidden_feature=int(input_feature*0.5)

    self.block1 = CBRBlock(input_feature, hidden_feature, 3, 1, 1)
    self.block2 = CBRBlock(hidden_feature, hidden_feature, 3, 1, 1)

    self.upsample = UpsampleBlock(hidden_feature, output_feature)


  def forward(self, x):

    out = self.block1(x)
    out = self.block2(out)
   
    out = self.upsample(out)

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

        self.cbr_block = CBRBlock(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.upsample_block = UpsampleBlock(1024, 512)
  
        self.dec5 = DecoderBlock(self.backbone_feature_depth, 256)
        self.dec4 = DecoderBlock(512, 128)
        self.dec3 = DecoderBlock(256, 64)
        self.dec2 = DecoderBlock(128, 32)
        self.dec1 = CBRBlock(64, 32, kernel_size=3, stride=1, padding=1)

  
        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]

     
        self.dropout = nn.Dropout2d(self.drop_prob)

        self.last_channels = 32

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up

        #print("X ", x.size())
        #print("Feats ", feats.size())

        # pentru layerele la care facem upsample, adunam la tensorii obtinuti in urma
        # aplicarii layerului, tensorii din backbone care au acelasi numar de
        # caracteristici
        if os > 1:
            if x.shape[3] < feats.shape[3]:
                os //= 2

                encoder_value = skips[os].detach()

                feats = torch.cat((feats, encoder_value), 1)

        x = feats
        return x, skips, os

    def forward(self, x, skips):
        os = 32
        #print("Decoder")
        # run layers
        x, skips, os = self.run_layer(x, self.cbr_block, skips, os)
        x, skips, os = self.run_layer(x, self.upsample_block, skips, os)
        #print("First UPSAMPLE Block")

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
