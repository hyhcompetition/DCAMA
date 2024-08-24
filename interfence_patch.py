import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import os
from model.DCAMA import DDCAMA
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
from PIL import Image
from scipy.ndimage import label, center_of_mass
from torchvision import transforms
import tqdm
image_path="../Dataset/VISION24-data-challenge-beta-test/test_pairs/0/query.jpg"


class ImageProcessor:
    def __init__(self, slice_size, overlap, image):
        self.image = image
        self.slice_size = slice_size
        self.overlap = overlap
        self.image_height, self.image_width = self.image.shape[:2]
        self.slices = []

    def get_imagepath(self, image_path):
        self.image = cv2.imread(image_path)
        
    def get_image(self, image):
        self.image = image
    
    def slice_image(self):
        step = self.slice_size - int(self.slice_size * self.overlap)
        padded_image = cv2.copyMakeBorder(
            self.image,
            0,
            (self.slice_size - self.image_height % self.slice_size) % self.slice_size,
            0,
            (self.slice_size - self.image_width % self.slice_size) % self.slice_size,
            cv2.BORDER_CONSTANT,
            value=0
        )
        
        padded_height, padded_width = padded_image.shape[:2]
        
        for y in range(0, padded_height, step):
            for x in range(0, padded_width, step):
                slice = padded_image[y:y + self.slice_size, x:x + self.slice_size]
                original_height = min(self.slice_size, self.image_height - y)
                original_width = min(self.slice_size, self.image_width - x)
                if slice.shape[0] < self.slice_size or slice.shape[1] < self.slice_size:
                    slice = cv2.copyMakeBorder(
                        slice,
                        0,
                        self.slice_size - slice.shape[0],
                        0,
                        self.slice_size - slice.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=0
                    )
                self.slices.append((slice, (x, y), (original_width, original_height)))
        
        return self.slices

    def model_inference(self, image_slice):
        """
        Placeholder for model inference.
        This function should take an image slice, run it through the model,
        and return the result.
        """
        # model_output = model.predict(image_slice)
        model_output = image_slice  # Placeholder: Replace with actual model inference
        return model_output

    def reconstruct_image(self, slices):
        result_image = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        weight_matrix = np.zeros((self.image_height, self.image_width), dtype=np.float32)

        for image_slice, (x, y), (original_width, original_height) in slices:
            if original_width == 0 or original_height == 0:
                continue
            
            x_end = x + original_width 
            y_end = y + original_height 
            
            result_image[y:y_end, x:x_end] += image_slice[:original_height, :original_width]
            weight_matrix[y:y_end, x:x_end] += 1
        
        # Avoid division by zero
        weight_matrix[weight_matrix == 0] = 1
        result_image = result_image / weight_matrix
        
        return result_image.astype(np.uint8)

    def debug_slices(self):
        plt.figure(figsize=(12, 12))
        num_slices = len(self.slices)
        cols = int(np.sqrt(num_slices))
        rows = (num_slices // cols) + 1

        for i, (image_slice, (x, y), (original_width, original_height)) in enumerate(self.slices):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB))
            plt.title(f'Slice {i+1} ({x},{y})')
            plt.axis('off')
    def crop_image_by_mask(self, image, mask, slice_size=384, min_slices=5):
        
        # 找到掩膜中每个独立区域的中心点
        labeled_mask, num_features = label(mask)
        centers = center_of_mass(mask, labeled_mask, range(1, num_features + 1))
        
        image_slices = []
        mask_slices = []
        image_height, image_width = image.shape[:2]
        
        for center in centers:
            center_y, center_x = int(center[0]), int(center[1])
            
            # 计算切片的起始和结束坐标
            x_start = max(center_x - slice_size // 2, 0)
            y_start = max(center_y - slice_size // 2, 0)
            x_end = min(x_start + slice_size, image_width)
            y_end = min(y_start + slice_size, image_height)
            
            # 调整起始坐标以确保切片大小为slice_size
            if x_end - x_start < slice_size:
                x_start = max(x_end - slice_size, 0)
            if y_end - y_start < slice_size:
                y_start = max(y_end - slice_size, 0)
            
            # 裁剪图片和掩膜
            image_slice = image[y_start:y_end, x_start:x_end]
            mask_slice = mask[y_start:y_end, x_start:x_end]
            
            # 保存切片
            image_slices.append(image_slice)
            mask_slices.append(mask_slice)
        
        # 如果切片数量不足5个，则随机复制已有切片
        while len(image_slices) < min_slices:
            index = random.choice(range(len(image_slices)))
            image_slices.append(image_slices[index])
            mask_slices.append(mask_slices[index])
        
        return image_slices[:min_slices], mask_slices[:min_slices]



def test(model, dataloader, nshot):
    r""" Test """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    transform = transforms.ToTensor()
    for idx, batch in enumerate(dataloader):
        
        # 1. forward pass
        # batch = utils.to_cuda(batch)
        # batch = utils.to_cpu(batch)
        support_image = batch['support_imgs'][0]
        support_mask = batch['support_masks'][0]
        query_image = batch['query_img']
        support_image = utils.to_cpu(support_image).numpy().squeeze()
        query_image = utils.to_cpu(query_image).numpy().squeeze()
        support_mask = utils.to_cpu(support_mask).numpy().squeeze()
        
        processor = ImageProcessor(slice_size=384, overlap=0, image=query_image)
        support_imgs ,support_masks = processor.crop_image_by_mask(support_image, support_mask)
        for si,sm in zip(support_imgs,support_masks):
            cv2.imshow("sipport img",si)
            cv2.imshow("support mask",sm)
        query_images = processor.slice_image()
        mask_slices = query_images.copy()
        
        index = batch["query_name"][0].split("_")[0]
        for query in query_images:
            batch={}
            # cv2.imshow("ori",query[0])
            batch['query_img'] = transform(query[0]).unsqueeze()
            batch['support_imgs'] = [transform(i).unsqueeze() for i in support_imgs]
            batch['support_masks'] = [transform(i).unsqueeze() for i in support_masks]
            batch = utils.to_cuda(batch)
            
            pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
            pred_mask = pred_mask.squeeze()
            pred_mask = utils.to_cpu(pred_mask)
            pred_mask = np.array(pred_mask).astype(np.uint8)
            # Convert non-zero values to 255
            pred_mask[pred_mask != 0] = 255
            cv2.imshow("res",pred_mask)
            cv2.waitKey(0)
            query[0] = pred_mask
            mask_slices.append(query)
        reconstruct_array=processor.reconstruct_image(mask_slices)
        save_path = os.path.join("output",f"{index}.png")
        cv2.imwrite(save_path,reconstruct_array)


    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':
    
    
    # Arguments parsing
    args = parse_opts()

    Logger.initialize(args, training=False)

    # Model initialization
    model = DDCAMA(args.feature_extractor_path, "", "")
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained modeln
    if args.load == '': raise Exception('Pretrained model not specified.')
    params = model.state_dict()
    state_dict = torch.load(args.load)

    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)

    model.load_state_dict(state_dict)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(True, './vis')

    # Dataset initialization
    # FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    # FSSDataset.transform =transforms.Compose([transforms.ToTensor(),
    #                                         transforms.Normalize(FSSDataset.img_mean, FSSDataset.img_std)])
    # dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    IMAGESET=f"{args.datapath}/imagesets/test.txt"
    DATAPATH=args.datapath
    with open(IMAGESET,"r") as f:
        metadata = f.read().split('\n')
        metadata = sorted(metadata)
        img_metadata=[]
        for data in metadata:
            if data != '':
                
                img_metadata.append([data.split()[0], data.split()[1], int(data.split()[2]), data.split()[3]])
    # Test
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    T = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(img_mean, img_std)])
    
    with torch.no_grad():
        for idx,(xs,xq,cls,t) in enumerate(img_metadata):
            print(f"{idx}/{len(img_metadata)}")
            support_img = cv2.imread(os.path.join(DATAPATH,"images",xs+'.jpg'))
            query_img = cv2.imread(os.path.join(DATAPATH,"images",xq+'.jpg'))
            Gsupport_img = cv2.resize(support_img.copy(),(384,384))
            Gquery_img = cv2.resize(query_img.copy(),(384,384))
            
            # support_img = cv2.imread("/home/hyh/Documents/Eccv/Dataset/VISION24-data-challenge-beta-test/test_pairs/6/support.jpg")
            # query_img = cv2.imread("/home/hyh/Downloads/query-fotor-20240806194935.jpg")
            
            support_mask =  cv2.imread(os.path.join(DATAPATH,"annotations",xs+'.png'),cv2.IMREAD_GRAYSCALE)
            
            # support_mask =  cv2.imread("/home/hyh/Documents/Eccv/Dataset/VISION24-data-challenge-beta-test/test_pairs/6/support.png",cv2.IMREAD_GRAYSCALE)
            support_mask = (support_mask / 255)
            Gsupport_mask = cv2.resize(support_mask.copy(),(384,384))
            # slice_size = min(query_img.shape[0],query_img.shape[1]) 
            processor = ImageProcessor(slice_size=384, overlap=0, image=query_img)
            support_imgs ,support_masks = processor.crop_image_by_mask(support_img, support_mask)
            # for si,sm in zip(support_imgs,support_masks):
            #     cv2.imshow("sipport img",si)
            #     cv2.imshow("support mask",sm)
            support_imgs = [T(i) for i in support_imgs]
            support_masks = [torch.tensor(i) for i in support_masks]
            Gsupport_imgs = [T(Gsupport_img)]
            Gsupport_masks = [torch.tensor(Gsupport_mask)]
            
            query_images = processor.slice_image()
            # query_images=[[query_img]]
            
            mask_slices = []
            index = xq.split("_")[0]
            for query in query_images:
                q = query[0]
                # q = cv2.resize(q,(384,384),interpolation=cv2.INTER_CUBIC)
                # cv2.imshow("ori",q)
                batch={}
                batch['query_img'] = T(q).unsqueeze(0)
                batch['support_imgs'] = torch.stack(support_imgs).unsqueeze(0)
                batch['support_masks'] = torch.stack(support_masks).unsqueeze(0)
                batch['Gquery_img'] = T(Gquery_img).unsqueeze(0)
                batch['Gsupport_imgs'] = torch.stack(Gsupport_imgs).unsqueeze(0)
                batch['Gsupport_masks'] = torch.stack(Gsupport_masks).unsqueeze(0)
                batch = utils.to_cuda(batch)
                pred_mask = model.module.predict_mask_nshot(batch, nshot=args.nshot)
                pred_mask = pred_mask.squeeze()
                pred_mask = utils.to_cpu(pred_mask)
                pred_mask = np.array(pred_mask).astype(np.uint8)
                # Convert non-zero values to 255
                pred_mask[pred_mask != 0] = 255
                # cv2.imshow("res",pred_mask)
                # cv2.waitKey(0)
                pred_mask = cv2.resize(pred_mask,(384,384))
                mask_slices.append((pred_mask, query[1], query[2]))
                
            reconstruct_array=processor.reconstruct_image(mask_slices)
            # cv2.imshow("dst",reconstruct_array)
            # cv2.waitKey(0)
            save_path = os.path.join("output",f"{index}.png")
            cv2.imwrite(save_path,reconstruct_array)
            
                
        # test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    # Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    # Logger.info('==================== Finished Testing ====================')

