r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


class DatasetVISION(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'test' if split in ['val', 'test'] else 'train'
        self.fold = fold # int {0，1，2，3}
        self.nfolds = 4
        self.nclass = 56
        self.benchmark = 'defect'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'images')
        self.ann_path = os.path.join(datapath, 'annotations')
        
        self.transform = transform
        self.class_ids = [ x for x in range(12)]
        self.img_metadata = self.build_img_metadata()

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
       # if pair_type =="n":
        #    query_mask = torch.zeros_like(query_mask)
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

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize


    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

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
        metadata = os.path.join('data/splits/defect/patch/%s.txt' % (self.split))
        with open(metadata, 'r') as f:
            metadata = f.read().split('\n')[:-1]
        img_metadata=[]
        for data in metadata:
            if data.split()[3]=="p":
                pass
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
    datapath = "/home/hyh/Documents/Eccv/Dataset/VISION24-data-challenge-train/data"
    tr=  transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(800, 800))])
    mydata = DatasetVISION(datapath,0,tr,"test",1,False)
    dataloader = DataLoader(mydata, batch_size=4, shuffle=True, num_workers=1)
    for idx, batch in enumerate(dataloader):
        print(batch)
