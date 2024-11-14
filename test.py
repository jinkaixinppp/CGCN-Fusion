import os,sys

import numpy as np

sys.path.append('..')
import time

from model import *
from dataset import *
from fusion_model3 import Fusion
import argparse
from fusion_utlis import *
import torchvision.transforms.functional as TF
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def hyper_parametrs():
    parse = argparse.ArgumentParser(description='Created by liam Zhang')
    parse.add_argument('--encoder_path', type=str,
                       default=r'para\encoder_params_epoch120.pth',
                       help='encoder_path')
    parse.add_argument('--recon_path', type=str,
                       default=r'para\recon_params_epoch120.pth',
                       help='recon_path')

    parse.add_argument('--fusion_path', type=str,
                       default=r'para\fusion_params_epoch120.pth',
                       help='fusion_path')

    #test dataset
    parse.add_argument('--mri_file', type=str, default=r'data\mri', \
                       help='the test mri dataset path')
    parse.add_argument('--ct_file', type=str, default=r'data\ct', \
                       help='the test ct dataset path')

    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = hyper_parametrs()
    print('==>Loading the model')
    encoder=Encoder().to(device)
    fusion=Fusion().to(device)
    recon=Recon().to(device)
    if not os.path.exists(args.encoder_path):
        raise Exception('No pretrained model could be found!')
    state_dict = torch.load(args.encoder_path, map_location=device)
    # # 创建一个新的state_dict字典，只包含模型中存在的键
    new_state_dict = {}
    for key in state_dict.keys():
        if key in encoder.state_dict().keys():
            new_state_dict[key] = state_dict[key]
    encoder.load_state_dict(new_state_dict)


    if not os.path.exists(args.fusion_path):
        raise Exception('No pretrained model could be found!')
    fusion.load_state_dict(torch.load(args.fusion_path, map_location=device))

    if not os.path.exists(args.recon_path):
        raise Exception('No pretrained model could be found!')
    recon.load_state_dict(torch.load(args.recon_path, map_location=device))

    # encoder= encoder.to(device)
    # para = sum([np.prod(list(p.size())) for p in encoder.parameters()])
    # type_size = 4
    # print('encoder {} : params: {:4f}M'.format(encoder._get_name(), para * type_size / 1000 / 1000))
    # fusion = fusion.to(device)
    # para = sum([np.prod(list(p.size())) for p in fusion.parameters()])
    # type_size = 4
    # print('fusion {} : params: {:4f}M'.format(fusion._get_name(), para * type_size / 1000 / 1000))
    # recon = recon.to(device)
    # para = sum([np.prod(list(p.size())) for p in recon.parameters()])
    # type_size = 4
    # print('recon {} : params: {:4f}M'.format(recon._get_name(), para * type_size / 1000 / 1000))




    print('==>Loading the test dataset')
    mri = []
    ct = []


    if os.path.exists(args.mri_file) and os.path.isdir(args.mri_file):
        mri_list = sorted(os.listdir(args.mri_file))
        mri_list.sort(key=lambda x: int(x.split('.')[0]))
        for mri_name in mri_list:
            print(mri_name)
            if mri_name.split('.')[-1] == 'gif' or mri_name.split('.')[-1] == 'png' or mri_name.split('.')[-1] == 'tif' or mri_name.split('.')[-1] == 'jpg':
                mri_name = os.path.join(args.mri_file, mri_name)
                mri_img = Image.open(mri_name)
                # 将图像转为灰度图
                mri_img = mri_img.convert("L")
                # 将图像转为NumPy数组并归一化
                mri_arr = np.array(mri_img, dtype=np.float32) / 255.0
                mri.append(torch.from_numpy(np.expand_dims(mri_arr, axis=0)).view(1, 1, 256, 256))

    if os.path.exists(args.ct_file) and os.path.isdir(args.ct_file):
        ct_list = sorted(os.listdir(args.ct_file))
        ct_list.sort(key=lambda x: int(x.split('.')[0]))
        for ct_name in ct_list:
            if ct_name.split('.')[-1] == 'gif' or ct_name.split('.')[-1] == 'png' or ct_name.split('.')[-1] == 'tif' or ct_name.split('.')[-1] == 'jpg':
                ct_name = os.path.join(args.ct_file, ct_name)
                ct_img = Image.open(ct_name)
                # 将图像转为灰度图
                ct_img = ct_img.convert("L")
                # 将图像转为NumPy数组并归一化
                ct_arr = np.array(ct_img, dtype=np.float32) / 255.0
                print(1)
                ct.append(torch.from_numpy(np.expand_dims(ct_arr, axis=0)).view(1, 1, 256, 256))




    start = time.time()
    with torch.no_grad():
        encoder.eval()
        fusion.eval()
        recon.eval()
        for i in range(20):
            print(i)

            img_c=ct[i]
            f_low, f_high = encoder(img_c)  # 将输入的数据送入编码器网络中
            # 将两个编码器的特征图级
            f_ct=fusion(f_low, f_high)
            # f_ct= torch.cat((f_low, f_high), dim=1)
            img_m = mri[i]
            # img_mri = torch.unsqueeze(test_mri[i], dim=0)
            f1_low, f1_high = encoder(img_m)  # 将输入的数据送入编码器网络中
            # 将两个编码器的特征图级
            # f_mri= torch.cat((f1_low, f1_high), dim=1)
            f_mri=fusion(f1_low, f1_high)
            # f=f_ct + f_mri
            # f=fusion_strategy(f_ct,f_mri,device=device,strategy ="FL1N")
            # Fusion= Fusion_SPA()
            # f=LEM(f_ct,f_mri)
            f=new_fusion_rule1(f_ct,f_mri)
            output = recon(f)  # 将输入的数据送入重建模块
            # 得到重建的输出图像 output。对输出图像进行归一化，将像素值缩放到0-1之间
            output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))

            # 将归一化的输出图像转换为PIL图像
            output_pil = TF.to_pil_image(output[0])
            save_path = os.path.join(r'reslut', f'{i}.gif')
            output_pil.save(save_path)





    print("=" * 80)
    end=time.time()
    print('the total time: %.4fs'%(end-start))






