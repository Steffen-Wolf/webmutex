import torch
from mipnet.models.unet import UNet2d
# alternative models
from mipnet.models.unet import UNet2dGN


def main():
    
    pytorch_model = UNet2d(1, 1, pad_convs=True)
    # load pretrained weights
    # pytorch_model.load_state_dict(torch.load('model.pytorch'))
    pytorch_model.eval()

    dummy_input = torch.rand(1, 1, 256, 256)
    torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)


if __name__ == '__main__':
    main()