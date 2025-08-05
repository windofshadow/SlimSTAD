
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv




def build_local_edge_index(D, H, W, neighborhood='6', device='cpu'):
    N = D * H * W
    coords = torch.stack(torch.meshgrid(
        torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'
    ), dim=-1).reshape(-1, 3)

    neighbor_offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dz == 0 and dy == 0 and dx == 0):
                    continue
                if neighborhood == '6' and abs(dz) + abs(dy) + abs(dx) != 1:
                    continue
                neighbor_offsets.append((dz, dy, dx))

    edges = []
    for dz, dy, dx in neighbor_offsets:
        z = coords[:, 0] + dz
        y = coords[:, 1] + dy
        x = coords[:, 2] + dx

        valid = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
        src = coords[valid][:, 0] * (H * W) + coords[valid][:, 1] * W + coords[valid][:, 2]
        dst = z[valid] * (H * W) + y[valid] * W + x[valid]
        edges.append(torch.stack([src, dst], dim=0))

    edge_index = torch.cat(edges, dim=1).long().to(device)
    return edge_index


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super(GATLayer, self).__init__()
        assert out_channels % heads == 0, "out_channels must be divisible by heads"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.gat = GATConv(in_channels, out_channels // heads, heads=heads)
        self.adjust_channels = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W

        # 生成 edge_index（静态图）
        edge_index = build_local_edge_index(D, H, W, device=x.device)

        # flatten x 为图节点特征
        x_reshaped = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # 每个样本独立处理 GAT
        out_batch = []
        for i in range(B):
            node_feats = x_reshaped[i]  # [N, C]
            out_node = self.gat(node_feats, edge_index)  # [N, out_channels]
            out_batch.append(out_node)

        out = torch.stack(out_batch, dim=0)  # [B, N, out_channels]

        # reshape 回 5D 格式
        out = out.permute(0, 2, 1).contiguous().view(B, self.out_channels, D, H, W)

        # 残差连接（通道对齐）
        x_res = self.adjust_channels(x)
        out = out + x_res

        return out




class DSCdWithGAT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_heads=3):
        super(DSCdWithGAT, self).__init__()

        # 深度卷积
        self.depthwise_conv = nn.Conv3d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels
        )

        # 替换逐点卷积为GAT层
        self.gat_layer = GATLayer(in_channels, out_channels, num_heads)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.gat_layer(x)
        return x


class C3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, dropout_keep_prob=0.5, num_heads=4):
        super(C3D, self).__init__()
        self.conv1 = DSCdWithGAT(in_channels, 64, kernel_size=(3, 1, 30),  
                                              stride=(2, 1, 1),  
                                              padding=(1, 0, 0), num_heads=4)  

        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.conv2 = DSCdWithGAT(64, 256, kernel_size=(3, 1, 3),
                                              stride=(1, 1, 2), padding=(1, 0, 1), num_heads=4)
        self.bn2 = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        self.conv3 = DSCdWithGAT(256, 512, kernel_size=(3, 1, 3),
                                              stride=(1, 1, 1), padding=(1, 0, 1), num_heads=4)
        self.bn3 = nn.BatchNorm3d(512)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        self.conv4 = DSCdWithGAT(512, 832, kernel_size=(3, 1, 3),
                                              stride=(1, 1, 1), padding=(1, 0, 1), num_heads=4)
        self.bn4 = nn.BatchNorm3d(832)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.global_pool = nn.AdaptiveAvgPool3d((2125, 1, 1))

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # 第二层卷积
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # 第三层卷积
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # 第四层卷积
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # 全局池化
        x = self.global_pool(x)

        return {
            'feature4': x,
        }




