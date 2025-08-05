import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from STAD.common.C3D import C3D
from STAD.common.config import config
from STAD.common.layers import Unit1D, Unit3D
from STAD.prop_pooling.boundary_pooling_op import BoundaryMaxPooling , BoundaryConPooling0, BoundaryConPooling1, BoundaryConPooling2, BoundaryConPooling3, BoundaryConPooling4, BoundaryConPooling5, BoundaryConPooling6

num_classes = config['dataset']['num_classes']
freeze_bn = config['model']['freeze_bn']
freeze_bn_affine = config['model']['freeze_bn_affine']
ConPoolingKernal = config['model']['ConPoolingKernal']
frame_num = config['dataset']['training']['clip_length']
ConfDim_num = config['model']['ConfDim_num']
LocDim_num = config['model']['LocDim_num']
layer_num = 6
conv_channels = 512
feat_t = 256 // 4


class C3D_BackBone(nn.Module):
    def __init__(self,in_channels = 3 ,  freeze_bn=True, freeze_bn_affine=True):
        super(C3D_BackBone, self).__init__()
        self._model = C3D(in_channels=in_channels)
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    def train(self, mode=True):
        """
        调用 train 模式时，选择性冻结 BatchNorm3d。
        """
        super(C3D_BackBone, self).train(mode)
        if self._freeze_bn and mode:
            print("Freezing all BatchNorm3d in C3D backbone.")
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()  # 设置为 eval 模式
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x):
        # print(x.shape,'x')
        return self._model(x)


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)


class ProposalBranch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(ProposalBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels * 2,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.boundary_max_pooling = BoundaryMaxPooling()
        # self.BoundaryConPooling = BoundaryConPooling(in_channels=proposal_channels * 2,proposal_channels=proposal_channels) 
        self.roi_conv = nn.Sequential(
            Unit1D(in_channels=proposal_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 4+400,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments,ConfResult_feature):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        '''-----------------------对边界范围的数据进行卷积提取特征-----------------------------'''
        # Con_feature = self.BoundaryConPooling(feature, anchor) # [2, 2000, 4]
        # prop_Con_feature = self.BoundaryConPooling(frame_level_feature, frame_anchor)
        # prop_Con_feature = self.roi_conv(prop_Con_feature)
        # print('最大池化前feature.shape',feature.shape,'hou:',prop_feature.shape) # 最大池化前feature.shape torch.Size([2, 1024, 63]) hou: torch.Size([2, 1024, 63])
        # print('最大池化前frame_level_feature',frame_level_feature.shape,'hou:',frame_level_feature.shape) # 最大池化前frame_level_feature torch.Size([2, 512, 8000]) hou: torch.Size([2, 512, 8000])
        '''------------------------------------over-----------------------------------------'''
        # prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short], dim=1)
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short,ConfResult_feature], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature


class ProposalBranch_conf(nn.Module):
    def __init__(self, in_channels, proposal_channels,kernel):
        super(ProposalBranch_conf, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels * 2,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.kernel = kernel
        self.boundary_max_pooling = BoundaryMaxPooling()
        '''-------------------------------------------锚框内容卷积的初始化--------------------------------------------------'''
        self.BoundaryConPooling0 = BoundaryConPooling0(in_channels=proposal_channels ,proposal_channels=proposal_channels,kernel=kernel[0],padding='no') 
        self.BoundaryConPooling1 = BoundaryConPooling1(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[1],padding='no')
        self.BoundaryConPooling2 = BoundaryConPooling2(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[2],padding='no')
        self.BoundaryConPooling3 = BoundaryConPooling3(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[3],padding='no')
        self.BoundaryConPooling4 = BoundaryConPooling4(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[4],padding='no')
        self.BoundaryConPooling5 = BoundaryConPooling5(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[5],padding='no')
        self.BoundaryConPooling6 = BoundaryConPooling6(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[6],padding='no')


        '''-----------------------------------------------over-------------------------------------------------------------'''
        self.roi_conv = nn.Sequential(
            Unit1D(in_channels=proposal_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 6,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments,anchor,frame_anchor,order,LocResult_feature):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        # '''----------------------------对预测出的锚框中的数据进行卷积提取特征-----------------------------'''
        # print('juanji anchor',anchor,anchor.shape,'juanji frame_anchor',frame_anchor.shape) # anchor.shape = [2, 2000, 2]
        L1 = len(anchor[0,:,0]) # 
        L2 = len(frame_level_feature[0,0,:]) # 8000
        # print('L1',L1,'L2',L2)
        a1 = np.ones([1, 512])
        b1 = torch.from_numpy(a1).cuda()

        ConFeature = [] # 用于存放卷积特征
        Frame_confeature = [] # 用于存放帧水平的卷积特征
        Kernals=self.kernel
        for i in range(L1):
            Data1 = []
            Data2 = []
            if hasattr(torch.cuda, 'empty_cache'):
             torch.cuda.empty_cache()
            # if hasattr(torch.cuda, 'empty_cache'):
	        #     torch.cuda.empty_cache()
            for j in range(anchor.shape[0]):
                a,b,c = feature.shape
                A,B,C = frame_level_feature.shape
                left1 = torch.clamp(anchor[j,i,0],0,L1)
                right1 = torch.clamp(anchor[j,i,1],0,L1)
                if right1 <=left1:
                    data1 = torch.from_numpy(np.zeros([a,b,Kernals[order+1]]))
                    

                else:
                    # print('left1',left1,'right1',right1)
                    data1 = feature[j,:,int(left1):int(right1)+1]
                left2 = torch.clamp(frame_anchor[j,i,0],0,L2)
                right2 = torch.clamp(frame_anchor[j,i,1],0,L2)
                if right2 <=left2:
                    data2 = torch.from_numpy(np.zeros([A,B,Kernals[0]]))

                else:
                    data2 = frame_level_feature[j,:,int(left2):int(right2)+1]
                # print('data1.shape',data1.shape)
                data1 =data1.unsqueeze(0)
                # print('data1.shape',data1.shape)
                data2 =data2.unsqueeze(0) # [1, 1024, 1]
                # print('data2.shape',data2.shape)
                data1 = F.interpolate(data1, size=Kernals[order+1])
                # print('data1.shape',data1.shape) # [1, 1024, 200]
                data2 = F.interpolate(data2, size=Kernals[0])
                # print('data2.shape',data2.shape)
                data1 =data1.squeeze(0)
                # print('data1.shape',data1.shape) # [1024, 200]
                data2 =data2.squeeze(0)
                # print('data2.shape',data2.shape) # [512, 800]
                Data1.append(data1)
                Data2.append(data2)
            Data1 = torch.stack(Data1, dim=0)  # [batch_size, channels, length]
            Data2 = torch.stack(Data2, dim=0)  # [batch_size, channels, length]
            # print('Data1.shape',Data1.shape) # [2, 1024, 200]
            # print('Data2.shape',Data2.shape) # [2, 512, 800]
            if order == 0:
                con_feature = self.BoundaryConPooling1(Data1)
            if order == 1:
                con_feature = self.BoundaryConPooling2(Data1)
            if order == 2:
                con_feature = self.BoundaryConPooling3(Data1)
            if order == 3:
                con_feature = self.BoundaryConPooling4(Data1)
            if order == 4:
                con_feature = self.BoundaryConPooling5(Data1)
            if order == 5:
                con_feature = self.BoundaryConPooling6(Data1)
            prop_Con_feature = self.BoundaryConPooling0(Data2) # [1, 512, 1]
            con_feature = con_feature.squeeze(-1)
            prop_Con_feature = prop_Con_feature.squeeze(-1)
            # with torch.no_grad():
            # b1 = torch.cat([b1,prop_Con_feature],dim = 0)
            ConFeature.append(con_feature)
            Frame_confeature.append(prop_Con_feature)
            # torch.cuda.empty_cache()
            # torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
            # print('b1.shape:',b1.shape)
        ConFeature = torch.stack(ConFeature,dim = 2)
        Frame_confeature = torch.stack(Frame_confeature,dim = 2)
        # print('ConFeature.shape:',ConFeature.shape,'Frame_confeature.shape:',Frame_confeature.shape)
        prop_Con_feature = self.roi_conv(Frame_confeature)
        # print('最大池化前feature.shape',feature.shape,'hou:',prop_feature.shape) # 最大池化前feature.shape torch.Size([2, 1024, 63]) hou: torch.Size([2, 1024, 63])
        # print('最大池化前frame_level_feature',frame_level_feature.shape,'hou:',frame_level_feature.shape) # 最大池化前frame_level_feature torch.Size([2, 512, 8000]) hou: torch.Size([2, 512, 8000])
        '''------------------------------------over-----------------------------------------'''
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short,ConFeature,prop_Con_feature], dim=1) # 将卷积获得的特征也加进来了
        # prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature







class CoarsePyramid(nn.Module):
    def __init__(self, feat_channels, frame_num,ConPoolingKernal): # 原为256
        super(CoarsePyramid, self).__init__()
        out_channels = conv_channels
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.frame_num = frame_num
        self.layer_num = layer_num
        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[0],
                output_channels=out_channels,
                kernel_shape=[1, 1, 1],
                padding='spatial_valid',
                use_batch_norm=False,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))



        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.loc_proposal_branch = ProposalBranch(out_channels, 512)
        # self.conf_proposal_branch = ProposalBranch(out_channels, 512) # _conf
        self.conf_proposal_branch = ProposalBranch_conf(out_channels, 512, kernel=ConPoolingKernal) # len_anchor:对锚框中的数据卷积时设置的锚框长度

        self.prop_loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None
        )
        self.prop_conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None
        )

        self.center_head = Unit1D(
            in_channels=out_channels,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.deconv = nn.Sequential(
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

        self.average_pool = nn.AvgPool3d((85,1,1),stride=(85,1,1))
        '''------------------------------------------------分类和回归特征的形成-----------------------------------------------------------------'''
        self.conf_linear = nn.Linear(in_features = num_classes, out_features = ConfDim_num) # bias = ture  
        self.loc_linear = nn.Linear(in_features = 2, out_features = LocDim_num)
        self.relu = nn.ReLU()

        '''-----------------------------------------------------------OVER---------------------------------------------------------------'''
        self.priors = []
        t = feat_t

        self.loc_heads.append(ScaleExp())

        # 初始化 priors
        self.priors = []  
        t = 25 # 单层时间维度为 23
        self.priors.append(
            torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
        )


    def forward(self, feat_dict, ssl=False):
        pyramid_feats = []
        locs = []
        confs = []
        centers = []
        prop_locs = []
        prop_confs = []
        trip = []

        x1 = feat_dict['feature4']  
        x1 = self.average_pool(x1)
        batch_num = x1.size(0)
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                # print('x01.shape',x1.shape)
                x = conv(x1)
                # print('x0.shape',x.shape) # [1, 512, 50, 1, 1]
                x = x.squeeze(-1).squeeze(-1)
            elif i == 1:
                # print('x02.shape',x2.shape)
                x = conv(x2)
                # print('x1.shape',x.shape) # [1, 512, 25, 1, 1]
                x = x.squeeze(-1).squeeze(-1)
                x0 = pyramid_feats[-1]
                y = F.interpolate(x, x0.size()[2:], mode='nearest')
                pyramid_feats[-1] = x0 + y
            else:
                x = conv(x)
            pyramid_feats.append(x)

        frame_level_feat = pyramid_feats[0].unsqueeze(-1)
        # print(frame_level_feat.shape)
        frame_level_feat = F.interpolate(frame_level_feat, [self.frame_num, 1]).squeeze(-1)
        # print(frame_level_feat.shape)
        
        frame_level_feat = self.deconv(frame_level_feat)
        # print(frame_level_feat.shape)
        
        trip.append(frame_level_feat.clone())
        start_feat = frame_level_feat[:, :256] # 后面通过激活指导使帧水平特征含盖了边界位置信息
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        for i, feat in enumerate(pyramid_feats):
            # print('i',i)
            loc_feat = self.loc_tower(feat) # 后面通过激活指导使金字塔最底层特征含盖了边界位置信息
            conf_feat = self.conf_tower(feat)
            loc_data =  self.loc_heads[i](self.loc_head(loc_feat)).view(batch_num, 2, -1).permute(0, 2, 1).contiguous()
                    
                    
            locs.append( loc_data)
            conf_data = self.conf_head(conf_feat).view(batch_num, num_classes, -1).permute(0, 2, 1).contiguous()
                    
            confs.append(conf_data )  
            # locs.append(
            #     self.loc_heads[i](self.loc_head(loc_feat))
            #         .view(batch_num, 2, -1)
            #         .permute(0, 2, 1).contiguous()
            # )
            # confs.append(
            #     self.conf_head(conf_feat).view(batch_num, num_classes, -1)
            #         .permute(0, 2, 1).contiguous()
            # )
            t = feat.size(2)
            # print('t',t)
            with torch.no_grad():
                segments = locs[-1] / self.frame_num * t
                # print('self.priors[i]:',len(self.priors[i]))
                # if i ==0:
                #     print('locs[-1]',locs[-1])
                priors = self.priors[i].expand(batch_num, t, 1).to(feat.device) # 因此锚点个数由金字塔卷积的序列长度来定
                new_priors = torch.round(priors * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)

                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                anchor = torch.cat([l_segment,r_segment],dim=-1)
                # print('anchor[0]',anchor[0])
                # if i ==0:
                #     print('plen:',plen,'torch.mean(plen)',torch.mean(plen)) #  torch.mean(plen)
                segments = torch.cat([
                    torch.round(l_segment - out_plen),
                    torch.round(l_segment + in_plen),
                    torch.round(r_segment - in_plen),
                    torch.round(r_segment + out_plen)
                ], dim=-1)

                decoded_segments = torch.cat(
                    [priors[:, :, :1] * self.frame_num - locs[-1][:, :, :1],
                     priors[:, :, :1] * self.frame_num + locs[-1][:, :, 1:]],
                    dim=-1)
                frame_anchor = decoded_segments
                plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                # print('mean_plen_frame:',torch.mean(plen))
                frame_segments = torch.cat([
                    torch.round(decoded_segments[:, :, :1] - out_plen),
                    torch.round(decoded_segments[:, :, :1] + in_plen),
                    torch.round(decoded_segments[:, :, 1:] - in_plen),
                    torch.round(decoded_segments[:, :, 1:] + out_plen)
                ], dim=-1)
            # print('segments',segments[:,0:2,:])
            # print('frame_segments',frame_segments[:,0:2,:])
            '''-------------------------------------分类或回归结果转成特征的第一方法----------------'''
            # print('locs:',loc_data.shape,'confs.shape',conf_data.shape)      # locs: torch.Size([2, 23, 2]) confs.shape torch.Size([2, 23, 8])
            ConfResult_feature = self.conf_linear(conf_data).permute(0, 2, 1).contiguous()
            # print(ConfResult_feature.shape,666)
            LocResult_feature = self.loc_linear(loc_data).permute(0, 2, 1).contiguous()
            ConfResult_feature = self.relu(ConfResult_feature)
            LocResult_feature = self.relu(LocResult_feature)
            # print('ConfResult_feature:',ConfResult_feature.shape,'LocResult_feature.shape',LocResult_feature.shape) # ConfResult_feature: torch.Size([2, 10, 23]) LocResult_feature.shape torch.Size([2, 10, 23])
            '''------------------------------------------------------over------------------------------------------'''
            loc_prop_feat, loc_prop_feat_ = self.loc_proposal_branch(loc_feat, frame_level_feat, # 金字塔每层的特征与同一帧水平特征结合得精细特征
                                                                     segments, frame_segments,ConfResult_feature)
            conf_prop_feat, conf_prop_feat_ = self.conf_proposal_branch(conf_feat, frame_level_feat,
                                                                        segments, frame_segments,anchor,frame_anchor,order=i,LocResult_feature=LocResult_feature)

            '''-------------------------------------根据WiFi信号特征，对锚框中的数据进行卷积提取特征----------------'''







            
            '''------------------------------------------------------over------------------------------------------'''
            if i == 0:
                trip.extend([loc_prop_feat_.clone(), conf_prop_feat_.clone()])
                ndim = loc_prop_feat_.size(1) // 2
                start_loc_prop = loc_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_loc_prop = loc_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                start_conf_prop = conf_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_conf_prop = conf_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                if ssl:
                    return trip
            prop_locs.append(self.prop_loc_head(loc_prop_feat).view(batch_num, 2, -1)
                             .permute(0, 2, 1).contiguous())
            prop_confs.append(self.prop_conf_head(conf_prop_feat).view(batch_num, num_classes, -1)
                              .permute(0, 2, 1).contiguous())
            centers.append(
                self.center_head(loc_prop_feat).view(batch_num, 1, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, num_classes) for o in confs], 1)
        prop_loc = torch.cat([o.view(batch_num, -1, 2) for o in prop_locs], 1)
        prop_conf = torch.cat([o.view(batch_num, -1, num_classes) for o in prop_confs], 1)
        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return loc, conf, prop_loc, prop_conf, center, priors, start, end,\
               start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop


class BDNet(nn.Module):
    def __init__(self, in_channels=3, backbone_model=None, training=True):
        super(BDNet, self).__init__()

        self.coarse_pyramid_detection = CoarsePyramid([832, 1024],frame_num=frame_num,ConPoolingKernal=ConPoolingKernal)
        self.reset_params()

        self.backbone = C3D(in_channels=3)
        # self.backbone = C3D
        
        self.boundary_max_pooling = BoundaryMaxPooling()
        self._training = training
        
        # if self._training:
        #     if backbone_model is None:
        #         self.backbone.load_pretrained_weight()
        #     else:
        #         self.backbone.load_pretrained_weight(backbone_model)
        self.scales = [1, 4, 4]

    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, proposals=None, ssl=False):
        # x should be [B, C, 256, 96, 96] for THUMOS14
        feat_dict = self.backbone(x)
        # print(feat_dict.keys())
        if ssl:
            top_feat = self.coarse_pyramid_detection(feat_dict, ssl)
            decoded_segments = proposals[0].unsqueeze(0)
            plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
            in_plen = torch.clamp(plen / 4.0, min=1.0)
            out_plen = torch.clamp(plen / 10.0, min=1.0)
            frame_segments = torch.cat([
                torch.round(decoded_segments[:, :, :1] - out_plen),
                torch.round(decoded_segments[:, :, :1] + in_plen),
                torch.round(decoded_segments[:, :, 1:] - in_plen),
                torch.round(decoded_segments[:, :, 1:] + out_plen)
            ], dim=-1)
            anchor, positive, negative = [], [], []
            for i in range(3):
                bound_feat = self.boundary_max_pooling(top_feat[i], frame_segments / self.scales[i])
                # for triplet loss
                ndim = bound_feat.size(1) // 2
                anchor.append(bound_feat[:, ndim:, 0])
                positive.append(bound_feat[:, :ndim, 1])
                negative.append(bound_feat[:, :ndim, 2])
            return anchor, positive, negative
        else:
            loc, conf, prop_loc, prop_conf, center, priors, start, end, \
            start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop = \
                self.coarse_pyramid_detection(feat_dict)
            return {
                'loc': loc,
                'conf': conf,
                'priors': priors,
                'prop_loc': prop_loc,
                'prop_conf': prop_conf,
                'center': center,
                'start': start,
                'end': end,
                'start_loc_prop': start_loc_prop,
                'end_loc_prop': end_loc_prop,
                'start_conf_prop': start_conf_prop,
                'end_conf_prop': end_conf_prop
            }


def test_inference(repeats=3, clip_frames=256):
    model = BDNet(training=False)
    model.eval()
    model.cuda()
    import time
    run_times = []
    x = torch.randn([1, 3, clip_frames, 96, 96]).cuda()
    warmup_times = 2
    for i in range(repeats + warmup_times):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        run_times.append(time.time() - start)

    infer_time = np.mean(run_times[warmup_times:])
    infer_fps = clip_frames * (1. / infer_time)
    print('inference time (ms):', infer_time * 1000)
    print('infer_fps:', int(infer_fps))
    # print(y['loc'].size(), y['conf'].size(), y['priors'].size())


if __name__ == '__main__':
    test_inference(20, 256)
