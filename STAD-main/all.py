import random
import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from STAD.common.thumos_dataset import THUMOS_Dataset, get_video_info, \
    load_video_data, detection_collate, get_video_anno
from torch.utils.data import DataLoader
from STAD.thumos14.multisegment_loss import MultiSegmentLoss
from STAD.common.config import config
import numpy as np
import tqdm
import json
from STAD.common import videotransforms
from STAD.common.thumos_dataset import get_class_index_map
from STAD.thumos14.C3DBDNet import BDNet
from STAD.common.segment_utils import softnms_v2 ,soft_nms
import argparse
from STAD.evaluation.eval_detection import ANETdetection
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader,TensorDataset
import time
# from AFSD.common.thumos_dataset1 import get_video_info,data_process0
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']
ngpu = config['ngpu']
save_model_num = config['training']['save_model_num']
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
writer = SummaryWriter("runs/scalar_example")
save_model_num = config['training']['save_model_num']
train_state_path = os.path.join(checkpoint_path, 'training')
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config['training']['resume']


def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('ssl weight: ', config['training']['ssl'])
    print('piou:', config['training']['piou'])
    print('resume: ', resume)
    print('gpu num: ', ngpu)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


GLOBAL_SEED = 1


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def save_model(epoch, model, optimizer):
    # torch.save(model.module.state_dict(),
    #            os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(epoch)))


def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        # model.module.load_state_dict(torch.load(model_path))
        model.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        set_rng_state(state_dict['state'])
        # '''----------------------------加载对照组模型-----------------------------------'''
        # model_path = os.path.join(checkpoint_path, 'checkpoint-60-lw-5_lc-{}.ckpt'.format(resume))
        # model.load_state_dict(torch.load(model_path),strict=False) # ,strict=False
        # '''--------------------------over-------------------------------'''
    return start_epoch


def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    # print('start',len(start.view(-1))) # len(start): 12000
    # print('scores[:, 1].contiguous()',len(scores[:, 1].contiguous().view(-1)))
    # scores = scores.numpy()
    # if 1>=scores.all() >= 0:
    #     print("0")
    # else:
    #     print("有不符合的")
    # # print('scores[:, 1].contiguous().view(-1)',scores[:, 1].contiguous().view(-1))
    # scores = torch.from_numpy(scores)
    loss_start = F.binary_cross_entropy(start.view(-1),
                                        scores[:, 0].contiguous().view(-1).cuda(),
                                        reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1),
                                      scores[:, 1].contiguous().view(-1).cuda(),
                                      reduction='mean')
    # loss_start = F.binary_cross_entropy_with_logits(start.view(-1),
    #                                     scores[:, 0].contiguous().view(-1).cuda(),
    #                                     reduction='mean')
    # loss_end = F.binary_cross_entropy_with_logits(end.view(-1),
    #                                   scores[:, 1].contiguous().view(-1).cuda(),
    #                                   reduction='mean')
    return loss_start, loss_end


def forward_one_epoch(net, clips, targets, scores=None, training=True, ssl=False):
    clips = clips.cuda()
    # print('clips.shape:',clips.shape) # [2, 1, 256, 25, 30]
    targets = [t.cuda() for t in targets]
    # print('targets:',targets)
    if training:
        if ssl:
            # output_dict = net.module(clips, proposals=targets, ssl=ssl)
            output_dict = net(clips, proposals=targets, ssl=ssl)
            # print('you ssl')
        else:
            output_dict = net(clips, ssl=False)
            # print('wu ssl')
    else:
        with torch.no_grad():
            output_dict = net(clips)

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct = CPD_Loss(
            [output_dict['loc'], output_dict['conf'],
             output_dict['prop_loc'], output_dict['prop_conf'],
             output_dict['center'], output_dict['priors'][0]],
            targets)
        # print('scores[:, 0].contiguous().view(-1)',scores[:, 0].contiguous().view(-1).shape)
        # print('start.view(-1):',output_dict['start'].shape)
        loss_start, loss_end = calc_bce_loss(output_dict['start'], output_dict['end'], scores)
        scores_ = F.interpolate(scores, size=25) #  scale_factor=1.0 / 4
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output_dict['start_loc_prop'],
                                                               output_dict['end_loc_prop'],
                                                               scores_)
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output_dict['start_conf_prop'],
                                                                 output_dict['end_conf_prop'],
                                                                 scores_)
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
    return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
    else:
        net.eval()

    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    loss_contras_val = 0
    cost_val = 0
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            # print('clips.shape',clips.shape) #[1, 1, 8000, 9, 30]
            loss_l, loss_c, loss_prop_l, loss_prop_c,\
            loss_ct, loss_start, loss_end = forward_one_epoch(
                net, clips, targets, scores, training=training, ssl=False)
            loss_l = loss_l * config['training']['lw']
            loss_c = loss_c * config['training']['cw']
            loss_prop_l = loss_prop_l * config['training']['lw']
            loss_prop_c = loss_prop_c * config['training']['cw']
            loss_ct = loss_ct * config['training']['cw']
            cost = loss_l + loss_c + loss_prop_l + loss_prop_c + loss_ct + loss_start + loss_end

            ssl_count = 0
            loss_trip = 0
            for i in range(len(flags)):
                if flags[i] and config['training']['ssl'] > 0:
                    loss_trip += forward_one_epoch(net, ssl_clips[i].unsqueeze(0), [ssl_targets[i]],
                                                   training=training, ssl=True) * config['training']['ssl']
                    loss_trip_val += loss_trip.cpu().detach().numpy()
                    ssl_count += 1
            if ssl_count:
                loss_trip_val /= ssl_count
                loss_trip /= ssl_count
            cost = cost + loss_trip
            if training:
                optimizer.zero_grad()
                cost.backward()
                clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
            loss_loc_val += loss_l.cpu().detach().numpy()
            loss_conf_val += loss_c.cpu().detach().numpy()
            loss_prop_l_val += loss_prop_l.cpu().detach().numpy()
            loss_prop_c_val += loss_prop_c.cpu().detach().numpy()
            loss_ct_val += loss_ct.cpu().detach().numpy()
            loss_start_val += loss_start.cpu().detach().numpy()
            loss_end_val += loss_end.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy())))

    loss_loc_val /= (n_iter + 1)
    loss_conf_val /= (n_iter + 1)
    loss_prop_l_val /= (n_iter + 1)
    loss_prop_c_val /= (n_iter + 1)
    loss_ct_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    loss_trip_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)
    print('单次迭代平均lose:',cost_val)
    if training:
        prefix = 'Train'
        # if epoch > max_epoch-5:
        #     save_model(epoch, net, optimizer)
        # if epoch == 150:
        #     save_model(epoch, net, optimizer)
        # if epoch == 60:
        #     save_model(epoch, net, optimizer)
        # if epoch == 90:
        #     save_model(epoch, net, optimizer)
        if epoch in save_model_num:
            save_model(epoch, net, optimizer)
    else:
        prefix = 'Val'

    writer.add_scalar('Total Loss', cost_val, epoch)
    writer.add_scalar('loc', loss_loc_val, epoch)
    writer.add_scalar('conf', loss_conf_val, epoch)



if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    ngpu = 1
    """
    Setup model
    """
    net = BDNet(in_channels=1,
                # backbone_model=config['model']['backbone_model'],
                training=True)
    # for para in net.backbone.parameters():
    #     para.requires_grad = False

    # net = nn.DataParallel(net, device_ids=list(range(ngpu))).cuda()
    net = net.cuda()

    # for k, v in net.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    """
    Setup optimizer
    """
    # optimizer = torch.optim.Adam(net.parameters(),
    #                              lr=learning_rate,
    #                              weight_decay=weight_decay)
    optimizer = optim.RAdam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    """
    Setup loss
    """
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)

    """
    Setup dataloader
    """
    train_video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['dataset']['training']['video_anno_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['dataset']['training']['video_data_path'])
    train_dataset = THUMOS_Dataset(train_data_dict,
                                   train_video_infos,
                                   train_video_annos)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    print('len(train_dataset)',len(train_dataset))
    epoch_step_num = len(train_dataset) // batch_size
    
    """
    Start training
    """
    # start_epoch = resume_training(resume, net, optimizer)
    # for i in range(start_epoch, max_epoch + 1):
    #     print('train_epoch:',i)
    #     start = time.perf_counter()
    #     run_one_epoch(i, net, optimizer, train_data_loader, len(train_dataset) // batch_size)
    #     end = time.perf_counter()
    #     print('one epoch use time:',end-start)
    """
    Start testing
    该测试程序，对测试数据没有使用RGB归一化，所以在训练时也不能使用RGB归一化
    """
    num_classes = config['dataset']['num_classes']
    conf_thresh = config['testing']['conf_thresh']
    top_k = config['testing']['top_k']
    nms_thresh = config['testing']['nms_thresh']
    nms_sigma = config['testing']['nms_sigma']
    clip_length = config['dataset']['testing']['clip_length']
    stride = config['dataset']['testing']['clip_stride']
    max_epoch = config['training']['max_epoch']
    checkpoint_path = config['testing']['checkpoint_path']
    json_name = config['testing']['output_json']
    output_path = config['testing']['output_path']
    softmax_func = True
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fusion = config['testing']['fusion']

    for i in save_model_num: # max_epoch-4, max_epoch+1
        print(i)
        checkpoint_path = checkpoint_path + "checkpoint-"+str(i)+".ckpt"
        json_name = str(i)+json_name
        video_infos = get_video_info(config['dataset']['testing']['video_info_path'])
        video_annos = get_video_anno(video_infos,
                                       config['dataset']['testing']['video_anno_path'])
        originidx_to_idx, idx_to_class = get_class_index_map()
        # print('idx_to_class',idx_to_class)
        npy_data_path = config['dataset']['testing']['video_data_path']

        net = BDNet(in_channels=config['model']['in_channels'],
                training=False)
        # print("checkpoint_path:",checkpoint_path)
        # net = nn.DataParallel(net, device_ids=list(range(ngpu))).cuda()
        net.load_state_dict(torch.load(checkpoint_path),strict=False)
        checkpoint_path = config['testing']['checkpoint_path'] 
        net.eval().cuda()

        if softmax_func:
            score_func = nn.Softmax(dim=-1)
        else:
            score_func = nn.Sigmoid()

        centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])

        result_dict = {}
        list_correct_rate = []
        for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
            sample_count = video_infos[video_name]['sample_count']
            sample_fps = video_infos[video_name]['sample_fps']
            sample_gt = video_annos[video_name]
            total = 0 # 总共的锚点数
            correct = 0 # 预测正确的锚点数
            # print('sample_gt:',sample_gt) # [[126.0, 757.0, 10], [1586.0, 2282.0, 9]] # 显示该序列的真实标签
            if sample_count < clip_length:
                offsetlist = [0]
            else:
                offsetlist = list(range(0, sample_count - clip_length + 1, stride))
                if (sample_count - clip_length) % stride:
                    offsetlist += [sample_count - clip_length]
            
            data = np.load(os.path.join(npy_data_path, str(video_name) + '.npy'))
            data = np.reshape(data, (len(data), -1,30), order='c')
            data = np.expand_dims(data, 0).repeat(1, axis=0)
            data = torch.from_numpy(data)
            data = data.type(torch.FloatTensor)
            data = data.permute(2, 1, 0, 3)
            
            # print('data.shape',data.shape) # [1, 8000, 9, 30]
            output = []
            for cl in range(num_classes):
                output.append([])
            res = torch.zeros(num_classes, top_k, 3)
            clip = data.unsqueeze(0).cuda()
            with torch.no_grad():
                # print('clip',clip.shape)
                # print('Test',clip.shape)
                output_dict = net(clip)

            loc, conf, priors = output_dict['loc'], output_dict['conf'], output_dict['priors'][0]
            prop_loc, prop_conf = output_dict['prop_loc'], output_dict['prop_conf']
            center = output_dict['center']
            loc = loc[0]
            conf = conf[0]
            prop_loc = prop_loc[0]
            prop_conf = prop_conf[0]
            center = center[0]

            pre_loc_w = loc[:, :1] + loc[:, 1:]
            loc = 0.5 * pre_loc_w * prop_loc + loc
            decoded_segments = torch.cat(
                [priors[:, :1] * clip_length - loc[:, :1],
                    priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
            decoded_segments.clamp_(min=0, max=clip_length)
            # print('decoded_segments[0:10,:]:',decoded_segments.shape)  # [3938, 2]
            conf = score_func(conf)
            # print("conf0[0:10,:]",conf[0:10,:])
            prop_conf = score_func(prop_conf)
            # print('conf1',prop_conf)
            center = center.sigmoid()
            # print("center[0:10,:]",center[0:10,:]) # [2954, 11]
            conf = (conf + prop_conf) / 2.0
            # print("conf1[0:10,:]",conf[0:10,:])
            '''------------------------------------------为检验分类效果加了这段---------------------------------------------'''
#             y_train = []
#             place_priors = priors[:, :1] * clip_length
#             for i in range(len(place_priors)):
#                 blank = True
#                 for j in range(len(sample_gt)):
#                     if  sample_gt[j][0] <= place_priors[i] <= sample_gt[j][1]:
#                             y_train.append(sample_gt[j][2])
#                             blank = False
#                 if blank:
#                     y_train.append(0)
#             # print('gt_y:',y_train)
#             _, predicted = torch.max(conf.data, 1)
#             # print(a)
#             print(predicted)
#             # _,的作用是，是使predicted返回output.data行中最大数值所在位置代表的类别
#             total += len(y_train)
#             # print(total)
#             y_train = np.array(y_train)
#             y_train = torch.from_numpy(y_train).cuda()
#             # print(predicted.shape)  # 应为 (batch_size, num_classes)
#             # print(y_train.shape)     # 应为 (batch_size,)

#             correct += (predicted == y_train).sum().item()
#             y_train = []
#             correct_rate = round(correct/total,3)
#             list_correct_rate.append(correct_rate)
#             total = 0
#             correct = 0
            '''-------------------------------------------------------over------------------------------------------------------'''
            conf = conf * center 
            # print("conf2[:,0:10]",conf[0:10,:])    # [3938, 11]
            conf = conf.view(-1, num_classes).transpose(1, 0)
            conf_scores = conf.clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl] >conf_thresh
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    print('bad')
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                # print('l_mask',l_mask.shape)  # [3938, 2]
                segments = decoded_segments[l_mask].view(-1, 2)
                # decode to original time
                # segments = (segments * clip_length + offset) / sample_fps
                offset = 0
                segments = (segments + offset) / 1 # sample_fps
                segments = torch.cat([segments, scores.unsqueeze(1)], -1)
                output[cl].append(segments)
                # np.set_printoptions(precision=3, suppress=True)
            sum_count = 0
            for cl in range(1, num_classes):
                if len(output[cl]) == 0:
                    continue
                tmp = torch.cat(output[cl], 0)
                # print("tmp.shape",tmp.shape) 
                tmp, count = soft_nms(tmp, sigma=nms_sigma, top_k=top_k) # soft_nms(segments, overlap=0.3, sigma=0.5, top_k=1000):
                # print("nms_sigma",nms_sigma)
                # print("\nclass",cl,"tmp1",tmp.shape,"\ntmp1",tmp)
                res[cl, :count] = tmp
                sum_count += count

            sum_count = min(sum_count, top_k)
            flt = res.contiguous().view(-1, 3)
            flt = flt.view(num_classes, -1, 3)
            proposal_list = []
            for cl in range(1, num_classes):
                class_name = idx_to_class[cl]  # 通过键值来对类别名称进行索引 # {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9'}
                # print('class_name',type(class_name))
                tmp = flt[cl].contiguous()
                # print(f"flt[{cl}]: {flt[cl]}")
                # print(f"Filtered tmp for class {class_name}: {tmp}")

                tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
                if tmp.size(0) == 0:
                    continue
                tmp = tmp.detach().cpu().numpy()
                for i in range(tmp.shape[0]):
                    tmp_proposal = {}
                    tmp_proposal['label'] = class_name
                    tmp_proposal['score'] = float(tmp[i, 2])
                    tmp_proposal['segment'] = [float(tmp[i, 0]),
                                                float(tmp[i, 1])]
                    proposal_list.append(tmp_proposal)
                    
            result_dict[video_name] = proposal_list
            # print("len(proposal_list)",len(proposal_list))
        output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
        # print(output_dict)

        # print('平均正确率：',sum(list_correct_rate)/len(list_correct_rate),"\n各条序列准确率",list_correct_rate)

        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)
        json_name = config['testing']['output_json']
    writer.close()

    """
    Start evaluating
    """
    writer = SummaryWriter("runs/eval")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    max_epoch = config['training']['max_epoch']
    max = 0
    max_i = 0
    x = []
    y = []
    for i in save_model_num:   # max_epoch-4, max_epoch + 1
        # gt_json = '/root/tf-logs/output/thumos_gt.json'
        gt_json = 'tf-logs/STAD-main/thumos_annotations/gt_behave2.json'
        output_json = '/root/tf-logs/STAD-main/output/'+ str(i) + 'Amend-3-WIFI.json'           
        tious = [0.3, 0.4, 0.5, 0.6, 0.7] # /root/tf-logs/Amend-2_Backbone_I3D-num0-9/output/86anchor-388_con-avgpool_num0-9.json
        anet_detection = ANETdetection( # '/root/tf-logs/Amend-2_Backbone_I3D-num0-9/output/86anchor-388_con-avgpool_num0-9'
            ground_truth_filename=gt_json, # 
            prediction_filename=output_json,
            subset='test', tiou_thresholds=tious)
        mAPs, average_mAP, ap = anet_detection.evaluate()
        # print(mAPs, "\n", average_mAP, "\n", ap)
        print("epoch", i)
        for (tiou, mAP) in zip(tious, mAPs):
            print("mAP at tIoU {} is {}".format(tiou, mAP))
        print(average_mAP, "\n")

        if average_mAP > max:
            max = average_mAP
            max_i = i

        writer.add_scalar('average_mAP', round(average_mAP, 4), i)
        x.append(i)
        y.append(round(average_mAP, 4))

    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.ylabel("average mAP")
    plt.show()

