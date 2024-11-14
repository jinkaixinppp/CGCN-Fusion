
from model import *
from dataset import *
import datetime
import sys
sys.path.append('..')
import torch
from tqdm import tqdm
import  argparse
from torch import  nn
from torch.autograd import Variable
import time
from torch.nn import functional as F

from fusion_model3 import Fusion

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
clip_grad_norm_value =10

parse = argparse.ArgumentParser(description='jin')
parse.add_argument('--train_path', type=str, default='CT-MRI/', help='Train PATH')
# parse.add_argument('--train_path',type=str,default='CT-MRI/',help='Train PATH')
parse.add_argument('--train_epoch', type=int, default=100, help='train epoch')
parse.add_argument('--epoch_gap', type=int, default=1, help='epoch_gap')
parse.add_argument('--train_batchsize', type=int, default=8, help='train batch_size')
parse.add_argument('--lr_rate', type=float, default=0.0001, help='lr_rate')
parse.add_argument('--eval_batchsize', type=int, default=16, help='eval batch_size')
parse.add_argument('--model_path', type=str, default=None, help='pretraining model')

parse.add_argument('--param1', type=float, help='the tradeoff between loss', default=1)
parse.add_argument('--lrf', type=float, default=0.01)
arg = parse.parse_args()


def _init_vit_weights(m):

    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

#特征提取阶段训练  训练解码器和编码器
def train_feature(encoder,fusion ,recon,train_loader1,train_loader2,optimizer1,optimizer2,optimizer3,device, epoch):
    # 进度条
    progress_bar = tqdm(total=len(train_loader1), desc=f'Epoch {epoch + 1}/{arg.train_epoch}', unit='batch')



    gradient_operator = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float, device=device,
                                     requires_grad=False).view(1, 1, 3, 3).contiguous()


    for step, (img_noise, img_data) in enumerate(zip(train_loader1, train_loader2)):
        encoder.train()
        fusion.train()
        recon.train()

        encoder.zero_grad()
        fusion.zero_grad()
        recon.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        data_original = img_data.to(device)
        data_noise = img_noise.to(device)
        data = Variable(data_noise.clone(), requires_grad=False)

        H, W = data.shape[2], data.shape[3]
        f_low, f_high = encoder(data)

        f_fusion = fusion(f_low,f_high)

        output = recon(f_fusion)
        epsilon =1e-8
        output_min = torch.min(output)
        output_max = torch.max(output)
        denominator = output_max - output_min + epsilon
        denominator = torch.where(denominator <= 0, torch.full_like(denominator, epsilon), denominator)


        output = (output - output_min) / denominator
        output = output.to(gradient_operator.device)

        loss1 = F.l1_loss(F.conv2d(output, gradient_operator, None, 1, 1, groups=output.shape[1]),
                          F.conv2d(data_original, gradient_operator, None, 1, 1, groups=data.shape[1])) / H * W

        loss2 = F.l1_loss(data_original, output)

        # loss2=ssim(data_original,output)

        total_loss = arg.param1 * loss1 + arg.param1 * loss2
        total_loss.backward()

        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(fusion.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(recon.parameters(), max_norm=clip_grad_norm_value, norm_type=2)


        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        loss_recon1 = loss2.detach()
        loss_gradient1 = loss1.detach()
        loss_all = total_loss.detach()

        # 更新进度条和其他信息
        progress_bar.set_postfix(loss_gr=f'{loss_gradient1 / (step + 1):.9f}',
                                 loss_re=f'{loss_recon1 / (step + 1):.9f}',
                                 loss_all=f'{loss_all / (step + 1):.9f}')  # 更新进度条显示的内容，控制小数精度
        progress_bar.update()

        # 关闭进度条
    progress_bar.close()




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==>Loading the training and testing datasets')


    # target_dir = r"data\CT-MRI"

    # # ct, mri = get_common_file(target_dir)
    # # noise_img, data_img = load_data1(ct, target_dir)

    # # Save the processed noise data as a file.
    data_file_noise = 'processed_data_noise_jg.pt'
    # torch.save(noise_img, data_file_noise)
    # # Save the raw data as a file.
    data_file = 'processed_data_2.pt'
    # torch.save(data_img, data_file)

    noise_img = torch.load(data_file_noise)
    data_img = torch.load(data_file)

    train_loader1, train_loader2 = get_loader1(noise_img, data_img, arg.train_batchsize)



    print('==>Loading the model')

    if torch.cuda.is_available():

        torch.cuda.manual_seed(1314)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    encoder = Encoder().to(device)
    fusion = Fusion().to(device)
    recon = Recon().to(device)
    fusion.apply(_init_vit_weights)
    save_interval=5

    encoder.apply(_init_vit_weights)
    recon.apply(_init_vit_weights)

    optimizer1 = torch.optim.Adam(
        encoder.parameters(), lr=arg.lr_rate, weight_decay=5E-5)
    optimizer3 = torch.optim.Adam(
        recon.parameters(), lr=arg.lr_rate, weight_decay=5E-5)


    optimizer2 = torch.optim.Adam(
        fusion.parameters(), lr=arg.lr_rate, weight_decay=5E-5)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=20, gamma=0.5)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)


    start=time.time()
    num=0
    for epoch in range(num,arg.train_epoch):

        train_feature(encoder,fusion,recon,train_loader1,train_loader2,optimizer1,optimizer2,optimizer3,device,epoch)

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        if (epoch + 1) % save_interval == 0:
            # 保存状态字典
            torch.save(encoder.state_dict(), r'encoder_params_epoch{}.pth'.format(epoch + 1))
            torch.save(recon.state_dict(), r'recon_params_epoch{}.pth'.format(epoch + 1))
            torch.save(optimizer1.state_dict(), r'optimizer1_state_epoch{}.pth'.format(epoch + 1))
            torch.save(optimizer3.state_dict(), r'optimizer3_state_epoch{}.pth'.format(epoch + 1))
            torch.save(fusion.state_dict(), r'fusion_params_epoch{}.pth'.format(epoch + 1))
            torch.save(optimizer2.state_dict(), r'optimizer2_state_epoch{}.pth'.format(epoch + 1))


    print('training fininshed!!!')
    end = time.time()
    print('total consuming times: {:.4f}s'.format(end - start))

