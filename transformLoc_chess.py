# -*- coding: utf-8 -*-

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as data
import os
import math
import torch.nn as nn
from model.PoseNet import model_parser
from model.ResNet50 import Res50PoseRess_withDepth
from dataset.Sevenscenesdataloadrer import Imageloder

from model.DepthLoss import MaskedL1Loss

def norm_q(x_q_base):

    Norm = torch.norm(x_q_base, 2, 1)
    norm_q_base = torch.div(torch.t(x_q_base), Norm)

    return torch.t(norm_q_base)


def default_loader(path):
    return Image.open(path).convert('I')

# def default_loader(path):
#     return Image.open(path).convert('RGB')

def median(lst):
    lst.sort()
    if len(lst) % 2 == 1:
        return lst[len(lst) // 2]
    else:
        return (lst[len(lst) // 2 - 1]+lst[len(lst) // 2]) / 2.0




learning_rate = 1e-5
batch_size =8

epochs = 500


cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print_every = 32
torch.cuda.manual_seed(1)

scene = 'chess'
data_dir = '/media/fangxu/Disk4T/LQ/data/'+scene
# label_dir_train = data_dir+'/singleImg/train.txt'
prep_train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.ToTensor()
])
#1246
dataset_train = Imageloder(datadir=data_dir,  Sequen = [1,2,4,6], transform_depth =prep_train_transform,transform_rgb = prep_train_transform)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

###

# label_dir_test = data_dir+'/singleImg/test.txt'
prep_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])

dataset_test = Imageloder(datadir=data_dir, Sequen = [3,5], transform_depth=prep_test_transform,transform_rgb = prep_test_transform)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

model = Res50PoseRess_withDepth()

criterion = nn.MSELoss()
depthLoss = MaskedL1Loss().cuda()

if cuda:
    criterion.cuda()
    model.cuda()

adam = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
Best_Pos_error = 9999.0
Best_Ort_error = 9999.0

loss_t_lst = []
loss_q_lst = []
loss_c_lst = []
loss_lst = []
median_lst = []

for e in range(epochs):
    print('\n\nEpoch {} of {}'.format(e, epochs))

    model.train()

    loss_t_counter = 0.0
    loss_q_counter = 0.0
    loss_counter = 0.0
    t = 0
    for i, (img_rgb,img_d, base_t, base_q) in enumerate(train_loader):

        if i % print_every == 0:
            print('Batch {} of {}'.format(i, len(train_loader)))
        # imgs_base = Variable(img_base.type(dtype))
        img_d = img_d.cuda()
        base_t  = base_t.cuda()
        base_q = base_q.cuda()

        adam.zero_grad()
        x_t_base, x_q_base,depthImg = model(img_d)

        norm_q_base = norm_q(x_q_base)

        loss_t = criterion(x_t_base, base_t)
        loss_q = criterion(x_q_base, base_q)
        loss_d = depthLoss(depthImg, img_d)

        loss_t_counter = loss_t_counter+loss_t.data
        loss_q_counter = loss_q_counter+loss_q.data
        loss = loss_t + loss_q+ 0.1*loss_d


        loss_counter += loss.data

        loss.backward()
        adam.step()
        t = t+1
    print('Average translation loss over epoch = {}'.format(loss_t_counter / (t + 1)))
    print('Average orientation loss over epoch = {}'.format(loss_q_counter / (t + 1)))
    # print('Average content loss over epoch = {}'.format(loss_c_counter / (i + 1)))
    print('Average loss over epoch = {}'.format(loss_counter / (t + 1)))

    pdist = nn.PairwiseDistance(2)

    if (e > -1 and e % 10 == 0):

        model.eval()
        model.isTest = True
        with torch.no_grad():

            dis_Err_Count = []

            ort2_Err_count = []

            loss_counter = 0.

            for i, (img_rgb,img_d, base_t, base_q) in enumerate(test_loader):
                imgs_ba = Variable(img_d.type(dtype))
                # imgs_re = Variable(img_ref.type(dtype))

                # x_t_base, x_q_base = model(imgs_ba)
                x_t = model(imgs_ba)

                base_t = base_t.cuda()
                base_q = base_q.cuda()

                dis_Err = pdist(x_t[0], base_t)
                dis_Err_Count.append(float(dis_Err))

                x_q_base = norm_q(x_t[1])

                Ort_Err2 = float(2 * torch.acos(torch.abs(torch.sum(base_q * x_q_base, 1))) * 180.0 / math.pi)

                ort2_Err_count.append(Ort_Err2)
                # result.append([dis_Err,Ort_Err2])

            dis_Err_i = median(dis_Err_Count)
            ort2_Err_i = median(ort2_Err_count)
            if dis_Err_i < Best_Pos_error:
                Best_Pos_error = dis_Err_i
                Best_Ort_error = ort2_Err_i
                print(Best_Pos_error, Best_Ort_error)
                isExists = os.path.exists(scene + '_Best_pose_depth_params.pt')
                if (isExists):
                    os.remove(scene + '_Best_pose_depth_params.pt')
                torch.save(model.state_dict(),
                           scene + '_Best_pose_depth_params.pt')

            # median_lst.append([dis_Err_i, ort2_Err_i])

            # print('average Distance err  = {} ,average orientation error = {} average Error = {}'.format(loss_counter / j,sum(dis_Err_Count)/j, sum(ort_Err_count)/j))
            # print('Media distance error  = {}, median orientation error2 = {}'.format(dis_Err_i, ort2_Err_i))
            # print(median_lst)
    print(Best_Pos_error, Best_Ort_error)