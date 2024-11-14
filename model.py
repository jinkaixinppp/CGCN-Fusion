
import  cv2
from PIL import Image
from GCN_Encode import GNNEncoder
from torchvision import transforms


from utlis import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Multi_scale_module(nn.Module):
    '''

     1*1 Conv Block
     3*3 Conv Block
     5*5 Conv Block
    '''
    def __init__(self):
        super(Multi_scale_module, self).__init__()

        # multiscale dilation conv2d 1*1
        self.convd1=nn.Sequential(
            nn.BatchNorm2d(num_features=64*3),
            nn.Conv2d(in_channels=64*3,out_channels=64*6,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=64 * 6, out_channels=64 * 3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #3*3Conv Block
        self.convd2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64*3),
            nn.Conv2d(in_channels=64*3,out_channels=64*6,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=64 * 6, out_channels=64 * 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #5*5Conv Block
        self.convd3 = nn.Sequential(
            nn.BatchNorm2d(num_features=64*3),
            nn.Conv2d(in_channels=64*3,out_channels=64*6,kernel_size=5,stride=1,padding=2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=64 * 6, out_channels=64 * 3, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bnorm1 = nn.BatchNorm2d(num_features=576)

    def forward(self, x):
        # dilated convolution

        x1 = self.convd1(x)
        x2 = self.convd2(x)
        x3 = self.convd3(x)

        diltotal = torch.cat([x1, x2, x3], dim=1)  #Connect by channel

        out = self.bnorm1(diltotal)

        return out




class CNNEncoder(nn.Module):
    """
    Low-level feature encoders
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv_Block1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=1,padding=0,stride=1),
            nn.ReLU(inplace=True),
        )
        self.conv_Block2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64*3,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(num_features=64*3),
            nn.ReLU(inplace=True),
        )
        self.network=Multi_scale_module()

        self.conv_Block3 = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=64 , kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x = x.to(dtype=torch.float32)

        self.conv_Block1 = self.conv_Block1.to(dtype=torch.float32,device=device)
        self.conv_Block2=self.conv_Block2.to(dtype=torch.float32,device=device)
        x=self.conv_Block2(self.conv_Block1(x))
        features = x
        x=self.network(x)
        x=self.conv_Block3(x)
        return x,features


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.Conv=nn.Sequential(nn.Conv2d(1, 1, kernel_size=1),
                                nn.BatchNorm2d(num_features=3),)
        self.CNN_Encoder=CNNEncoder()
        self.GCN_Enocoder=GNNEncoder()
    def forward(self,img):
        img=img.to(device)
        self.channel=img.shape[1]
        B,_,img_height,img_width = img.shape#input[B,C,H,W]
        low_,features=self.CNN_Encoder(img)  # 【B,64,H/4,W/4】

        features=F.interpolate(features, size=(img_height, img_width), mode='bilinear', align_corners=False)

        out=self.GCN_Enocoder(img,features)

        resized_low_ = F.interpolate(low_, size=(img_height, img_width), mode='bilinear', align_corners=False)

        return  resized_low_,out #[B,C,H,W]

class Recon(nn.Module):
    def __init__(self):
        super(Recon, self).__init__()
        self.deConv1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # 添加批归一化层
        self.deConv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 添加批归一化层
        self.deConv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # 添加批归一化层
        self.deConv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)  # 添加批归一化层
        self.deConv5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.deConv1(x)
        out = self.bn1(out)
        db1 = self.activate(out)
        out2 = self.deConv2(db1)
        out2 = self.bn2(out2)
        db2 = self.activate(out2)
        out3 = self.deConv3(torch.cat([db1, db2], dim=1))
        out3 = self.bn3(out3)
        db3 = self.activate(out3)
        out4 = self.deConv4(torch.cat([db2, db3], dim=1))
        out4 = self.bn4(out4)
        db4 = self.activate(out4)
        output = self.deConv5(db4)
        return output



def main():
    path = "/ct/2006.png"


    img = Image.open(path).convert('L')


    transformss = transforms.Compose(
        [transforms.ToTensor(),

         ])


    img=transformss(img).unsqueeze(0)



    model = Encoder()
    low_output,high_out = model(img)
    f=torch.cat([low_output,high_out], dim=1)
    model1=Recon()
    f=model1(f)

    print(f.shape,high_out.shape)

if __name__ == '__main__':
    main()






