import torch
import torch.nn as nn
import torch.nn.functional as F


class Quad_Conv(nn.Module):
    def __init__(self, in_channels=1, out_channels=15, padding=0):
        super().__init__()

        self.conv_r = nn.Conv2d(in_channels, out_channels, 3, 1, padding, bias=True)
        self.conv_g = nn.Conv2d(in_channels, out_channels, 3, 1, padding, bias=True)
        self.conv_b = nn.Conv2d(in_channels, out_channels, 3, 1, padding, bias=True)
        
        self.truncated_normal_(self.conv_r.weight)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -0.02)

    def truncated_normal_(self, tensor, mean=0, std=0.1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)

            return tensor

    def forward(self, x):
        out = self.conv_r(x) * self.conv_g(x) + self.conv_b(x * x)

        return F.relu(out)


class Quad_Deconv(nn.Module):
    def __init__(self, in_channels=1, out_channels=15, padding=0):
        super().__init__()

        self.deconv_r = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, padding, bias=True)
        self.deconv_g = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, padding, bias=True)
        self.deconv_b = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, padding, bias=True)

        self.truncated_normal_(self.deconv_r.weight)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.constant_(m.bias, -0.02)

    def truncated_normal_(self, tensor, mean=0, std=0.1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)

            return tensor

    def forward(self, x):
        out = self.deconv_r(x) * self.deconv_g(x) + self.deconv_b(x * x)

        return out


class QAE_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.quad_conv1 = Quad_Conv(in_channels=1, out_channels=15, padding=1)
        self.quad_conv2 = Quad_Conv(in_channels=15, out_channels=15, padding=1)
        self.quad_conv3 = Quad_Conv(in_channels=15, out_channels=15, padding=1)
        self.quad_conv4 = Quad_Conv(in_channels=15, out_channels=15, padding=1)
        self.quad_conv5 = Quad_Conv(in_channels=15, out_channels=15, padding=0) # padding = 0 here

        self.quad_deconv5 = Quad_Deconv(in_channels=15, out_channels=15, padding=0) # padding=0 here
        self.quad_deconv4 = Quad_Deconv(in_channels=15, out_channels=15, padding=1)
        self.quad_deconv3 = Quad_Deconv(in_channels=15, out_channels=15, padding=1)
        self.quad_deconv2 = Quad_Deconv(in_channels=15, out_channels=15, padding=1)
        self.quad_deconv1 = Quad_Deconv(in_channels=15, out_channels=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual_1 = x
        
        out = self.quad_conv1(x)
        out = self.quad_conv2(out)

        residual_2 = out
        
        out = self.quad_conv3(out)
        out = self.quad_conv4(out)

        residual_3 = out

        out = self.quad_conv5(out)
        out = self.quad_deconv5(out)
        out = self.relu(out + residual_3)

        out = self.relu(self.quad_deconv4(out))
        out = self.quad_deconv3(out)
        out = self.relu(out + residual_2)

        out = self.relu(self.quad_deconv2(out))
        out = self.quad_deconv1(out)
        out = self.relu(out + residual_1)

        return out

