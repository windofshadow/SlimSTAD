import torch.nn as nn
from torch.autograd import Function

import boundary_max_pooling_cuda
from STAD.common.layers import Unit1D, Unit3D

class BoundaryMaxPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, segments):
        output = boundary_max_pooling_cuda.forward(input, segments)
        ctx.save_for_backward(input, segments)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input, segments = ctx.saved_tensors
        grad_input = boundary_max_pooling_cuda.backward(
            grad_output,
            input,
            segments
        )
        return grad_input, None


class BoundaryMaxPooling(nn.Module):
    def __init__(self):
        super(BoundaryMaxPooling, self).__init__()

    def forward(self, input, segments):
        return BoundaryMaxPoolingFunction.apply(input, segments)


class BoundaryConPooling0(nn.Module): # 对锚框中的数据进行卷积
    def __init__(self, in_channels, proposal_channels,kernel,padding):
        super(BoundaryConPooling0, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=8,
                   stride=3,
                   padding=padding,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        # self.lr_conv0 = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels,
        #            kernel_shape=int(kernel/100),
        #            stride=3,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.lr_conv1 = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels,
        #            kernel_shape=int(kernel/100),
        #            stride=3,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.lr_conv2 = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels,
        #            kernel_shape=int(kernel/100),
        #            stride=3,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.lr_conv3 = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels,
        #            kernel_shape=int(7),
        #            stride=1,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.maxpool = nn.MaxPool1d(3, stride=2)
        # self.average_pool = nn.AvgPool1d(266,stride=2)
        # self.average_pool = nn.AvgPool1d(800,stride=2)
        # self.average_pool = nn.AvgPool1d(265,stride=2)改
        self.average_pool = nn.AvgPool1d(265,stride=2)
        
    def forward(self, input): 
        # print('input.shape:',input.shape)
        fm_short = self.cur_point_conv(input)
        # print('input.shape:',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.lr_conv0(fm_short)
        # # print('input.shape:',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.lr_conv1(fm_short)
        # # print('input.shape:',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.lr_conv2(fm_short)
        # # print('input.shape:',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.lr_conv3(fm_short)
        fm_short = self.average_pool(fm_short)
        # fm_short = self.average_pool(input)
        # print('0_input.shape:',input.shape,'fm_short.shape',fm_short.shape)
        return  fm_short



class BoundaryConPooling1(nn.Module): # 对锚框中的数据进行卷积
    def __init__(self, in_channels, proposal_channels,kernel,padding):
        super(BoundaryConPooling1, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=3,
                   stride=1,
                   padding=padding,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        # self.lr_conv0 = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels ,
        #            kernel_shape=int(kernel/5),
        #            stride=3,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels ),
        #     nn.ReLU(inplace=True)
        # )
        # self.lr_conv1 = nn.Sequential(
        #             Unit1D(in_channels=proposal_channels,
        #                 output_channels=proposal_channels,
        #                 kernel_shape=int(kernel/5),
        #                 stride=3,
        #                 padding=padding,
        #                 activation_fn=None),
        #             nn.GroupNorm(32, proposal_channels),
        #             nn.ReLU(inplace=True)
        #         )
        # self.lr_conv2 = nn.Sequential(
        #             Unit1D(in_channels=proposal_channels,
        #                 output_channels=proposal_channels,
        #                 kernel_shape=int(kernel/50+2),
        #                 stride=1,
        #                 padding=padding,
        #                 activation_fn=None),
        #             nn.GroupNorm(32, proposal_channels),
        #             nn.ReLU(inplace=True)
        #         )
        # self.maxpool = nn.MaxPool1d(3, stride=2)
        # self.average_pool = nn.AvgPool1d(66,stride=2)
        # self.average_pool = nn.AvgPool1d(20,stride=2)
        # self.average_pool = nn.AvgPool1d(18,stride=2)改
        self.average_pool = nn.AvgPool1d(18,stride=2)
        
    def forward(self, input): # input.shape: torch.Size([2, 1024, 2000]) segments.shape: torch.Size([2, 2000, 4])
        # print('input.shape:',input.shape,'segments.shape:',anchor.shape,'segments:',anchor)
        # data = input          # 将数据取出
        fm_short = self.cur_point_conv(input)
        # print('1_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.lr_conv0(fm_short)
        # fm_short = self.lr_conv1(fm_short)
        # fm_short = self.lr_conv2(fm_short)
        fm_short = self.average_pool(fm_short)
        # fm_short = self.average_pool(input)
        # print('1_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        return  fm_short

class BoundaryConPooling2(nn.Module): # 对锚框中的数据进行卷积
    def __init__(self, in_channels, proposal_channels,kernel,padding):
        super(BoundaryConPooling2, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=10,#原10
                   stride=1,
                   padding=padding,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        # self.lr_conv0 = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels,
        #            kernel_shape=int(kernel/20),
        #            stride=3,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels ),
        #     nn.ReLU(inplace=True)
        # )
        # self.lr_conv1 = nn.Sequential(
        #         Unit1D(in_channels=in_channels,
        #             output_channels=proposal_channels ,
        #             kernel_shape=int(kernel/20),
        #             stride=3,
        #             padding=padding,
        #             activation_fn=None),
        #         nn.GroupNorm(32, proposal_channels),
        #         nn.ReLU(inplace=True)
        #     )
        # self.average_pool = nn.AvgPool1d(10,stride=2)
        # self.average_pool = nn.AvgPool1d(8,stride=2)
    def forward(self, input): # input.shape: torch.Size([2, 1024, 2000]) segments.shape: torch.Size([2, 2000, 4])
        # print('input.shape:',input.shape,'segments.shape:',anchor.shape,'segments:',anchor)
        # data = input          # 将数据取出
        fm_short = self.cur_point_conv(input)
        # # fm_short = self.lr_conv(fm_short)
        # fm_short = self.lr_conv0(fm_short)
        # fm_short = self.lr_conv1(fm_short)
        # print('2_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.average_pool(fm_short)
        # print('2_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.average_pool(input)
        return  fm_short


class BoundaryConPooling3(nn.Module): # 对锚框中的数据进行卷积
    def __init__(self, in_channels, proposal_channels,kernel,padding):
        super(BoundaryConPooling3, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=5,#原5
                   stride=1,
                   padding=padding,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        # self.lr_conv0 = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels,
        #            kernel_shape=int(kernel/10),
        #            stride=3,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels ),
        #     nn.ReLU(inplace=True)
        # )
        # self.lr_conv1 = nn.Sequential(
        #             Unit1D(in_channels=proposal_channels,
        #                 output_channels=proposal_channels,
        #                 kernel_shape=int(kernel/10),
        #                 stride=3,
        #                 padding=padding,
        #                 activation_fn=None),
        #             nn.GroupNorm(32, proposal_channels),
        #             nn.ReLU(inplace=True)
        #         )
        # self.average_pool = nn.AvgPool1d(5,stride=2)
        # self.average_pool = nn.AvgPool1d(3,stride=2)
    def forward(self, input): # input.shape: torch.Size([2, 1024, 2000]) segments.shape: torch.Size([2, 2000, 4])
        # print('input.shape:',input.shape,'segments.shape:',anchor.shape,'segments:',anchor)
        # 将数据取出
        fm_short = self.cur_point_conv(input)
        # fm_short = self.lr_conv0(fm_short)
        # fm_short = self.lr_conv1(fm_short)
        # print('3_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.average_pool(fm_short)
        # print('3_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.average_pool(input)
        return  fm_short


class BoundaryConPooling4(nn.Module): # 对锚框中的数据进行卷积
    def __init__(self, in_channels, proposal_channels,kernel,padding):
        super(BoundaryConPooling4, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=3,
                   stride=1,
                   padding=padding,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        # self.lr_conv0 = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels,
        #            kernel_shape=int(kernel/5),
        #            stride=3,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.lr_conv1 = nn.Sequential(
        #     Unit1D(in_channels=in_channels,
        #            output_channels=proposal_channels,
        #            kernel_shape=int(kernel/5),
        #            stride=3,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.average_pool = nn.AvgPool1d(3,stride=2)
    def forward(self, input): # input.shape: torch.Size([2, 1024, 2000]) segments.shape: torch.Size([2, 2000, 4])
        # print('input.shape:',input.shape,'segments.shape:',anchor.shape,'segments:',anchor)
        # data = input          # 将数据取出
        fm_short = self.cur_point_conv(input)
        # fm_short = self.lr_conv0(fm_short)
        # fm_short = self.lr_conv1(fm_short)
        # print('4_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.average_pool(input)
        # print('1_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        return  fm_short


class BoundaryConPooling5(nn.Module): # 对锚框中的数据进行卷积
    def __init__(self, in_channels, proposal_channels,kernel,padding):
        super(BoundaryConPooling5, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=2,
                   stride=1,
                   padding=padding,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        # self.lr_conv = nn.Sequential(
        #     Unit1D(in_channels=proposal_channels,
        #            output_channels=proposal_channels ,
        #            kernel_shape=3,
        #            stride=1,
        #            padding=padding,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels ),
        #     nn.ReLU(inplace=True)
        # )
        # self.average_pool = nn.AvgPool1d(2,stride=2)
    def forward(self, input): # input.shape: torch.Size([2, 1024, 2000]) segments.shape: torch.Size([2, 2000, 4])
        # print('input.shape:',input.shape,'segments.shape:',anchor.shape,'segments:',anchor)
        # data = input          # 将数据取出
        fm_short = self.cur_point_conv(input)
        # fm_short = self.lr_conv(fm_short)
        # print('5_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        # fm_short = self.average_pool(input)
        # print('1_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        return  fm_short


class BoundaryConPooling6(nn.Module): # 对锚框中的数据进行卷积
    def __init__(self, in_channels, proposal_channels,kernel,padding):
        super(BoundaryConPooling6, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=2,
                   padding=padding,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        # self.lr_conv = nn.Sequential(
        #     Unit1D(in_channels=in_channels,
        #            output_channels=proposal_channels * 2,
        #            kernel_shape=1,
        #            activation_fn=None),
        #     nn.GroupNorm(32, proposal_channels * 2),
        #     nn.ReLU(inplace=True)
        # )
        # self.average_pool = nn.AvgPool1d(2,stride=2)
    def forward(self, input): # input.shape: torch.Size([2, 1024, 2000]) segments.shape: torch.Size([2, 2000, 4])
        # print('input.shape:',input.shape,'segments.shape:',anchor.shape,'segments:',anchor)
        # data = input          # 将数据取出
        fm_short = self.cur_point_conv(input)
        # fm_short = self.lr_conv(fm_short)
        # fm_short = self.average_pool(input)
        # print('6_input.shape',input.shape,'fm_short.shape',fm_short.shape)
        return  fm_short







