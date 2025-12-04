import torch
import torch.nn as nn
import torch.nn.functional as F


# --- BLOQUE BASE ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv_op(x)

# --- BLOQUE DE ATENCIÓN ---
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# --- DOWN Y UP ---
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention = AttentionBlock(F_g=in_channels // 2, F_l=out_channels, F_int=in_channels // 4)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.attention(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

# --- U-NET COMPLETO ---
class AttentionUNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super().__init__()
        self.down1 = DownSample(in_channels, 64, dropout)
        self.down2 = DownSample(64, 128, dropout)
        self.down3 = DownSample(128, 256, dropout)
        self.down4 = DownSample(256, 512, dropout)

        self.bottleneck = DoubleConv(512, 1024, dropout)

        self.up1 = UpSample(1024, 512, dropout)
        self.up2 = UpSample(512, 256, dropout)
        self.up3 = UpSample(256, 128, dropout)
        self.up4 = UpSample(128, 64, dropout)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.down1(x)
        d2, p2 = self.down2(p1)
        d3, p3 = self.down3(p2)
        d4, p4 = self.down4(p3)

        b = self.bottleneck(p4)

        u1 = self.up1(b, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        out = self.out(u4)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )
    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512,1024)


        self.up_convolution_1 = UpSample(1024,512)
        self.up_convolution_2 = UpSample(512,256)
        self.up_convolution_3 = UpSample(256,128)
        self.up_convolution_4 = UpSample(128,64)


        self.out = nn.Conv2d(in_channels= 64, out_channels=num_classes, kernel_size=1)

    def forward(self,x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Padding x1 if needed to match the size of x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class NestedUNet(nn.Module):
    def __init__(self, in_channels, num_classes, filters=[64, 128, 256, 512, 1024]):
        super().__init__()

        # Encoder path
        self.down1 = DownSample(in_channels, filters[0])
        self.down2 = DownSample(filters[0], filters[1])
        self.down3 = DownSample(filters[1], filters[2])
        self.down4 = DownSample(filters[2], filters[3])

        # Bottleneck
        self.bottleneck = DoubleConv(filters[3], filters[4])

        # Nested convolutional blocks for skip pathways
        self.conv0_1 = DoubleConv(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv(filters[1] + filters[2], filters[1])
        self.conv2_1 = DoubleConv(filters[2] + filters[3], filters[2])
        self.conv3_1 = DoubleConv(filters[3] + filters[4], filters[3])

        self.conv0_2 = DoubleConv(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = DoubleConv(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = DoubleConv(filters[2]*2 + filters[3], filters[2])

        self.conv0_3 = DoubleConv(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = DoubleConv(filters[1]*3 + filters[2], filters[1])

        self.conv0_4 = DoubleConv(filters[0]*4 + filters[1], filters[0])

        # Upsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final output
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0, p0 = self.down1(x)
        x1_0, p1 = self.down2(p0)
        x2_0, p2 = self.down3(p1)
        x3_0, p3 = self.down4(p2)

        x4_0 = self.bottleneck(p3)

        # Decoder - nested skip connections
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_2)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_2)], 1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_3)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        out = self.final_conv(x0_4)
        return out