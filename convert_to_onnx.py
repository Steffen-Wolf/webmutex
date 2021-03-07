import torch
import torch.nn as nn
from mipnet.models.unet import UNet2dTC

# TODO very simple model!!!!
# conv +  maxpool + transpose

from mipnet.layers import Upsample
class Simplemodel(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, pad_convs=True, depth=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                       kernel_size=3, padding=1),
                                  nn.ReLU())

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = Upsample(in_channels=in_channels,
                        out_channels=out_channels,
                        scale_factor=2, mode='bilinear', ndim=2)

        # self.upsample = nn.ConvTranspose2d(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     stride=2, kernel_size=2)

    def forward(self, input):
        out = self.conv(input)
        out = self.maxpool(out)
        out = self.upsample(out)
        # out = self.conv(out)
        return out



def main():

    pytorch_model = UNet2dTC(1, 4, pad_convs=True, depth=1)
    # load pretrained weights
    # pytorch_model.load_state_dict(torch.load('model.pytorch'))
    pytorch_model.eval()

    dummy_input = torch.rand(1, 1, 256, 256)
    pytorch_model(dummy_input)
    msg = torch.onnx.export(pytorch_model, dummy_input,
                            'onnx_model.onnx',
                            verbose=True,
                            opset_version=9)

    print(msg)

    # test the exported model
    import onnx
    loaded_model = onnx.load('onnx_model.onnx')
    # print(loaded_model)

    # import numpy as np
    # import caffe2.python.onnx.backend
    # from caffe2.python import core, workspace
    # output = caffe2.python.onnx.backend.run_model(loaded_model, dummy_input.numpy().astype(np.float32))

if __name__ == '__main__':
    main()
