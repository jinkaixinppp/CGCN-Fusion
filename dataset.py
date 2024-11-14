# Get dataloader for MRI-CT data

# Author: Simon Zhou, last modify Nov. 11, 2022

'''
Change log:
- Simon: file created, implement dataset loader
'''
from torchvision import transforms
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset,ConcatDataset
import skimage.io as io
import cv2
from PIL import Image



class getIndex(Dataset):
    def __init__(self, total_len):
        self.total_len = total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, ind):
        return torch.Tensor([ind])




def get_common_file(target_dir):

    ct = os.path.join(target_dir, "CT")
    mri = os.path.join(target_dir, "MRI")
   # totall=os.path.join(target_dir, "totall")

    ct_file = []
    mri_file = []


    # get file name for ct images
    for file in sorted(os.listdir(ct)):
        ct_file.append(file)

    # get file name for mri images
    for file in sorted(os.listdir(mri)):
        mri_file.append(file)



    diff1 = [file for file in ct_file if file not in mri_file]
    diff2 = [file for file in mri_file if file not in ct_file]


    assert len(diff1) == len(diff2) == 0, "data is somehow not paired"

    return ct_file, mri_file


#
#
# def load_data(file, target_dir, test_num):
#     test_num = min(len(file), test_num)  # 如果test_num大于文件数量，则取文件数量
#
#     test = []
#     for i in range(test_num):
#         test.append(file[i])
#     # print(test)
#
#     HEIGHT = 256
#     WIDTH = 256
#
#
#     # 1 channel image, with shape 256x256
#     data_ct = torch.empty(0, 1, HEIGHT, WIDTH)
#     data_mri = torch.empty(0, 1, HEIGHT, WIDTH)
#     data_ct_t = torch.empty(0, 1, HEIGHT, WIDTH)
#     data_mri_t = torch.empty(0, 1, HEIGHT, WIDTH)
#     for f in file:
#         # read data and normalize
#         img_ct = io.imread(os.path.join(target_dir, "CT", f)).astype(np.float32) / 255.
#         img_mri = io.imread(os.path.join(target_dir, "MRI", f)).astype(np.float32) / 255.
#
#         img_ct = torch.from_numpy(img_ct)
#         img_mri = torch.from_numpy(img_mri)
#         img_ct = img_ct.unsqueeze(0).unsqueeze(0)  # change shape to (1, 1, 256, 256)
#         img_mri = img_mri.unsqueeze(0).unsqueeze(0)
#
#         if f not in test:
#             data_ct = torch.cat((data_ct, img_ct), dim=0)
#             data_mri = torch.cat((data_mri, img_mri), dim=0)
#         else:
#             data_ct_t = torch.cat((data_ct_t, img_ct), dim=0)
#             data_mri_t = torch.cat((data_mri_t, img_mri), dim=0)
#
#     return data_ct, data_mri,data_ct_t, data_mri_t

def add_salt_and_pepper_noise(image_array, salt_prob=0.02, pepper_prob=0.02):
    """
    Add salt and pepper noise to the input image array.

    Parameters:
        image_array (numpy.ndarray): Input image array.
        salt_prob (float): Probability of adding salt noise.
        pepper_prob (float): Probability of adding pepper noise.

    Returns:
        numpy.ndarray: Noisy image array.
    """
    noisy_image_array = np.copy(image_array)

    # Generate salt noise
    salt_mask = np.random.rand(*image_array.shape) < salt_prob
    noisy_image_array[salt_mask] = 255

    # Generate pepper noise
    pepper_mask = np.random.rand(*image_array.shape) < pepper_prob
    noisy_image_array[pepper_mask] = 0

    return noisy_image_array

def add_salt_and_pepper_noise_to_image(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Add salt and pepper noise to the input PIL image.

    Parameters:
        image (PIL.Image.Image): Input image.
        salt_prob (float): Probability of adding salt noise.
        pepper_prob (float): Probability of adding pepper noise.

    Returns:
        PIL.Image.Image: Noisy image.
    """
    # Convert PIL image to numpy array
    image_array = np.array(image)

    # Add salt and pepper noise to the image array
    noisy_image_array = add_salt_and_pepper_noise(image_array, salt_prob, pepper_prob)

    # Convert the noisy image array back to PIL image
    noisy_image = Image.fromarray(noisy_image_array)

    return noisy_image

def add_noise(image):
    # Add Gaussian noise to the image
    noise = np.random.normal(loc=0, scale=20, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def load_data1(file_list, target_dir):
    HEIGHT = 256
    WIDTH = 256



    data_img = torch.empty(0, 1, HEIGHT, WIDTH)
    noise_img= torch.empty(0, 1, HEIGHT, WIDTH)

    i=0
    for i, f in enumerate(file_list[:7000], start=1):

        img = cv2.imread(os.path.join(target_dir, "new", f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))

        img_array = np.array(img, dtype=np.float32)
        img_array /= 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        # For the first 1500 images, add noise.
        if i <= 1500:
            noise1 = add_noise(img)
            noise = add_salt_and_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)

            noise1_array = np.array(noise1, dtype=np.float32)
            noise1_array /= 255.0
            noise1_tensor = torch.from_numpy(noise1_array).unsqueeze(0).unsqueeze(0)

            noise_array = np.array(noise, dtype=np.float32)
            noise_array /= 255.0
            noise_tensor = torch.from_numpy(noise_array).unsqueeze(0).unsqueeze(0)


            combined_img_tensor = torch.cat((img_tensor, noise1_tensor, noise_tensor), dim=0)
            noise_img = torch.cat((noise_img, combined_img_tensor), dim=0)

            original_img = torch.cat((img_tensor, img_tensor, img_tensor), dim=0)
            data_img = torch.cat((data_img, original_img), dim=0)
        else:

            noise_img = torch.cat((noise_img, img_tensor), dim=0)
            data_img = torch.cat((data_img, img_tensor), dim=0)

    print(noise_img.shape)
    print(data_img.shape)

    # Return both the noisy images and the original images
    return noise_img, data_img



#
# def load_data1(file_list, target_dir):
#     HEIGHT = 256
#     WIDTH = 256
#
#
#     # 创建空的张量来存储图像数据
#     data_img = torch.empty(0, 1, HEIGHT, WIDTH)
#
#     for f in file_list[:10284]:
#         img = Image.open(os.path.join(target_dir, "train", f)).convert("L")
#         transform = transforms.Compose([
#             transforms.Resize((HEIGHT, WIDTH)),])
#         img = transform(img)  # 应用中心裁剪
#         # 将图像转换为NumPy数组
#         img_array = np.array(img)
#         # 将数组的数据类型转换为float32
#         img = img_array.astype(np.float32)
#         img = img.astype(np.float32)  # 转换为float32类型
#         img /= 255.0  # 归一化处理
#         img_totall = torch.from_numpy(img)
#         img_totall = img_totall.unsqueeze(0).unsqueeze(0) # change shape to (1, 1, 256, 256)
#         data_img = torch.cat((data_img, img_totall), dim=0)
#     return data_img




def get_loader1(noise_img,original_img, bs):
    train_loader1 = DataLoader(noise_img, batch_size=bs, num_workers=0, shuffle=False, drop_last=False)
    train_loader2 = DataLoader(original_img, batch_size=bs, num_workers=0, shuffle=False, drop_last=False)
    return train_loader1,train_loader2


# def get_loader1(img , bs):
#
#     train_loader = DataLoader(img, batch_size=bs, num_workers=0, shuffle=True, drop_last=False)
#     return train_loader





