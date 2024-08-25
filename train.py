r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.DCAMA import DCAMA
from model.DCAMA import DDCAMA
from model.DCAMA_CLS import DCAMA
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        初始化 Sigmoid Focal Loss 函数

        参数:
        - alpha: 平衡正负样本的因子 (默认值: 0.25)
        - gamma: 调节难易样本权重的因子 (默认值: 2.0)
        - reduction: 损失的聚合方式，可以是 'none'、'mean' 或 'sum' (默认值: 'mean')
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算 Focal Loss

        参数:
        - inputs: 模型输出的预测值 (logits)
        - targets: 真实标签 (0 或 1)

        返回:
        - Focal Loss 值
        """
        # 使用 sigmoid 函数计算预测概率
        p = torch.sigmoid(inputs)

        # 计算交叉熵损失
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算 modulating factor
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma

        # 计算 alpha factor
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 计算 Focal Loss
        focal_loss = alpha_t * modulating_factor * ce_loss

        # 根据 reduction 参数聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(epoch, model ,dataloader, optimizer, criterion,training):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    total_loss = 0
    total_accuracy = 0
    loss_list=[]
    accuracy_list=[]
    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        output = model(batch['query_img'], batch['support_imgs'].squeeze(1))
        labels = torch.ones_like(output)
        # print(output)
        probabilities = torch.sigmoid(output)

        # 设定阈值为 0.5 来决定类别（0 或 1）
        threshold = 0.5
        predicted_labels = (probabilities >= threshold).long()
    
        # 2. Compute loss & update model parameters
        loss = criterion(output, labels)
        accuracy = (predicted_labels == labels).float().mean()
        
        total_loss += loss
        total_accuracy += accuracy
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (idx + 1) % 100 == 0:
            avg_loss = total_loss / 100
            avg_accuracy = total_accuracy / 100
            print(f'{idx}  Average Loss: {avg_loss:.4f} Average Accuracy: {avg_accuracy * 100:.2f}%')
            loss_list.append(avg_loss)
            accuracy_list.append(avg_accuracy)
            total_loss=0
            total_accuracy=0

        # # 3. Evaluate prediction
        # area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        # average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        
        # average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
    
    # # Write evaluation results
    # average_meter.write_result('Training' if training else 'Validation', epoch)
    # avg_loss = utils.mean(average_meter.loss_buf)
    # miou, fb_iou = average_meter.compute_iou()


    return sum(loss_list)/len(loss_list), sum(accuracy_list)/len(accuracy_list)


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    # ddp backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # Model initialization
    model = DCAMA('swin', args.feature_extractor_path, False)
    device = torch.device("cuda", args.local_rank)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
        # Load trained model
    if args.load != '':
        print("*******loading model***********")
        params = model.state_dict()
        state_dict = torch.load(args.load)

        for k1, k2 in zip(list(state_dict.keys()), params.keys()):
            state_dict[k2] = state_dict.pop(k1)

        model.load_state_dict(state_dict)

    # Helper classes (for training) initialization
    optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
                            "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
    Evaluator.initialize()
    if args.local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    if args.local_rank == 0:
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')
    criterion = SigmoidFocalLoss(alpha=0.25, gamma=2.0)
    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.nepoch):
        dataloader_trn.sampler.set_epoch(epoch)
        trn_loss, trn_miou = train(epoch, model, dataloader_trn, optimizer, criterion,training=True)
        print(f"TRN {epoch}: loss = {trn_loss} acc = {trn_miou}")
        # evaluation
        if args.local_rank == 0:
            with torch.no_grad():
                val_loss, val_miou = train(epoch, model, dataloader_val, optimizer,criterion, training=False)
                print(f"VAL {epoch}: loss = {val_loss} acc = {val_miou}")

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(model, epoch, val_miou)

            # Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            # Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            # Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            # Logger.tbd_writer.flush()

    # if args.local_rank == 0:
    #     Logger.tbd_writer.close()
    #     Logger.info('==================== Finished Training ====================')
