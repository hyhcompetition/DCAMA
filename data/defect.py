r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import numbers
import collections
import random
import cv2

def calculate_channel_means(image: np.ndarray) -> list:
    """
    计算输入图像的三个通道的均值。

    参数:
        image (np.ndarray): 使用cv2读取的图像数据。

    返回:
        tuple: 三个通道的均值（B_mean, G_mean, R_mean）。
    """
    # 确保输入图像是三通道的
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是三通道的（BGR）。")

    # 计算每个通道的均值
    b_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    r_mean = np.mean(image[:, :, 2])

    return [b_mean, g_mean, r_mean]
class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=0, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label



class DatasetVISION(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'test' if split in ['val', 'test'] else 'train'
        self.fold = fold # int {0，1，2，3}
        self.nfolds = 4
        self.nclass = 56
        self.benchmark = 'defect'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.datapath = datapath

        self.img_path = os.path.join(datapath, 'images')
        self.ann_path = os.path.join(datapath, 'annotations')
        
        self.transform = transform
        self.class_ids = [ x for x in range(12)]
        self.img_metadata = self.build_img_metadata()
        self.custom_transform = [RandRotate([0,360],[0,0,0]), RandomVerticalFlip(), RandomHorizontalFlip()]

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample, pair_type = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)
        
        query_img = self.transform(query_img)
    
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask = (query_cmask / 255).floor()
        if pair_type =="n":
            query_mask = torch.zeros_like(query_mask)
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            scmask = (scmask / 255).floor()
            support_masks.append(scmask)
        support_masks = torch.stack(support_masks)


        batch = {'query_img': query_img,
                'query_mask': query_mask,
                'query_name': query_name,

                
                'org_query_imsize': org_qry_imsize,

                'support_imgs': support_imgs,
                'support_masks': support_masks,
                'support_names': support_names,

                'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != 0] = 1
        mask[mask == 0] = 0

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]
        
        org_qry_imsize = query_img.size
        if self.split == "train":
            si = support_imgs[0]
            sm = support_masks[0]
            for t in self.custom_transform:
                si,sm=t(si,sm)
                query_img, query_mask = t(query_img, query_mask)
            support_imgs = [Image.fromarray(si)]
            support_masks = [sm]
        else:
            support_imgs = [Image.fromarray(support_imgs[0])]
        support_masks = [torch.tensor(i) for i in support_masks]
        query_mask = torch.tensor(query_mask)

        return Image.fromarray(query_img), query_mask, support_imgs, support_masks, org_qry_imsize
    
    def load_global_frame(self, query_name, support_names):
        query_name = query_name.replace('back','').split("-")[0]
        support_names = [i.replace('back','').split("-")[0] for i in support_names]
        
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]
        
        org_qry_imsize = query_img.size
        
        support_imgs = [Image.fromarray(support_imgs[0])]
        support_masks = [torch.tensor(i) for i in support_masks]
        query_mask = torch.tensor(query_mask)
        
        return Image.fromarray(query_img), query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = cv2.imread(os.path.join(self.ann_path, img_name) + '.png', cv2.IMREAD_GRAYSCALE)
        # mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        image = cv2.imread(os.path.join(self.img_path, img_name) + '.jpg')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # return Image.open(os.path.join(self.img_path, img_name) + '.jpg')
        return image
    def sample_episode(self, idx):
        support_name, query_name, class_sample, pair_type = self.img_metadata[idx]

        support_names = [support_name]
        return query_name, support_names, class_sample, pair_type

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds   # int = 5
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]  # for fold 0 : [0,1,2,3,4]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'train':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):
        metadata = f"{self.datapath}/imageset/{self.split}.txt"
        with open(metadata, 'r') as f:
            metadata = f.read().split('\n')[:-1]
        img_metadata=[]
        for data in metadata:
            img_metadata.append([data.split()[0], data.split()[1], int(data.split()[2]), data.split()[3]])

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

if __name__ == "__main__":
    datapath = "/home/hyh/Documents/Eccv/Dataset/VISION24-data-challenge-train/data_patch"
    tr=  transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(388, 388))])
    mydata = DatasetVISION(datapath, 0,tr,"train",1,False)
    dataloader = DataLoader(mydata, batch_size=4, shuffle=True, num_workers=1)
    for idx, batch in enumerate(dataloader):
        print(batch)
    # image = Image.open("/home/hyh/Documents/Eccv/Dataset/VISION24-data-challenge-train/data_patch_1shot/images/Cable_thunderbolt_000000-block1.jpg")
    # label = Image.open("/home/hyh/Documents/Eccv/Dataset/VISION24-data-challenge-train/data_patch_1shot/annotations/Cable_thunderbolt_000000-block1.png")
    # pad = calculate_channel_means(image)
    # cv2.imshow("o", image)
    # R = RandRotate([0,360], pad )
    # a,b = R(image,label)
    # cv2.imshow("r", a)
    # cv2.imshow("r_b", b)
    # cv2.waitKey(0)
