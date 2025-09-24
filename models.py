import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # weight: (out_channels, in_channels, kH, kW)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        """
        x: (N, C_in, H, W)
        ritorna: (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        kH, kW = self.kernel_size

        H_out = (H + 2*self.padding - kH) // self.stride + 1
        W_out = (W + 2*self.padding - kW) // self.stride + 1

        # output includendo il batch
        out = torch.zeros((N, self.out_channels, H_out, W_out),
                        device=x.device, dtype=x.dtype)

        for n in range(N):                     # loop sul batch
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[n, :, i*self.stride:i*self.stride+kH,
                                j*self.stride:j*self.stride+kW]
                    # patch: (C_in, kH, kW)
                    val = (patch.unsqueeze(0) * self.weight).sum(dim=(1,2,3))
                    if self.bias is not None:
                        val += self.bias
                    out[n, :, i, j] = val
        return out





class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()

        # First conv block
        self.conv1 = SimpleConv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second conv block
        self.conv2 = SimpleConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two conv+pool, image size reduces
        # Original: 48x48 → after pool1 (24x24) → after pool2 (12x12)
        self.fc1 = nn.Linear(self._get_flatten_size(), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_flatten_size(self):
        # pass a dummy tensor to infer the flatten size
        with torch.no_grad():
            x = torch.zeros(1, 1, 48, 48)  # batch=1, 1 channel, 48x48
            x = self.conv1(x); x = self.relu1(x); x = self.pool1(x)
            x = self.conv2(x); x = self.relu2(x); x = self.pool2(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # x: (batch_size, 1, 48, 48)

        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, features)

        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x



class SimpleCNN2(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN2, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # keep size 48x48
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 24x24

        # Second conv block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # keep size
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 12x12

        # Fully connected
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super(ResidualBlock, self).__init__()
        self.conv1 = SimpleConv2D(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = SimpleConv2D(out_channels, out_channels, kernel_size, stride, padding, bias) 
        
    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        out += x
        out = self.relu(out)
        return out

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super(ResidualBlock, self).__init__()
        self.conv1 = SimpleConv2D(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = SimpleConv2D(out_channels, out_channels, kernel_size, stride, padding, bias) 
        
    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        out += x
        out = self.relu(out)
        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # 1x1 conv only if in_channels != out_channels
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.res_conv = None

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))

        # apply 1x1 conv if needed
        if self.res_conv is not None:
            identity = self.res_conv(identity)

        out += identity
        out = self.relu(out)
        return out



class SimpleResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleResNet, self).__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.stage1_block1 = ResidualBlock(16, 16)
        self.stage1_block2 = ResidualBlock(16, 16)

        # Stage 2
        self.downsample1 = nn.MaxPool2d(2, 2)
        self.stage2_block1 = ResidualBlock(16, 32)
        self.stage2_block2 = ResidualBlock(32, 32)

        # Stage 3
        self.downsample2 = nn.MaxPool2d(2, 2)
        self.stage3_block1 = ResidualBlock(32, 64)
        self.stage3_block2 = ResidualBlock(64, 64)

        # Stage 4
        self.downsample3 = nn.MaxPool2d(2, 2)
        self.stage4_block1 = ResidualBlock(64, 128)
        self.stage4_block2 = ResidualBlock(128, 128)

        # Global pooling + fc
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        # Stage 1
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)

        # Stage 2
        x = self.downsample1(x)
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)

        # Stage 3
        x = self.downsample2(x)
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)

        # Stage 4
        x = self.downsample3(x)
        x = self.stage4_block1(x)
        x = self.stage4_block2(x)

        # Global pooling + flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
