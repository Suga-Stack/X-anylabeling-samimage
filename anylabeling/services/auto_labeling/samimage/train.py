import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks, sigmoid_ce_loss_jit, dice_loss_jit
import utils.misc as misc
import logging
from gather_hx_data import get_all_hx_train, get_all_hx_val
import datetime


def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        # if input_box is not None:
            # show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='orange', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='orange', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    home = os.path.expanduser('~')
    
    parser = argparse.ArgumentParser('spt', add_help=False)
    parser.add_argument('--dataset-path', default=f"{home}/datasets/OpenDataLab___SA-1B")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=16, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    parser.add_argument("--val-data", default='data/ad', type=str)
    parser.add_argument("--val-point-prompt", default=-1, type=int)

    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--obj", default="05ELD", type=str)

    return parser.parse_args()

from torch.utils.data.distributed import DistributedSampler
from glob import glob
from lora import Linear, MergedLinear, ConvLoRA, mark_only_lora_as_trainable, lora_state_dict
from segment_anything_training.modeling.transformer import Attention
from segment_anything_training.modeling.image_encoder import Attention as EncoderAttention

def prepare_lora(model_type, model: nn.Module, r):
    for name, module in model.named_children():
        if 'neck' in name:
            continue
        if isinstance(module, Attention):
            q_proj = module.q_proj
            v_proj = module.v_proj
            new_q_proj = Linear(q_proj.in_features, q_proj.out_features, r=r)
            new_v_proj = Linear(v_proj.in_features, v_proj.out_features, r=r)
            setattr(module, 'q_proj', new_q_proj)
            setattr(module, 'v_proj', new_v_proj)
        elif isinstance(module, EncoderAttention):
            qkv = module.qkv
            setattr(module, 'qkv', MergedLinear(qkv.in_features, qkv.out_features, r, enable_lora=[True, False, True]))
        elif ('rep' in model_type) and isinstance(module, nn.Conv2d) and module.kernel_size[0] == 1 and module.groups==1:
            setattr(model, name, ConvLoRA(module, module.in_channels, module.out_channels, 1, r=r))
        else:
            prepare_lora(model_type, module, r)     

def freeze_bn_stats(model: nn.Module):
    for _, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            

def main(net, train_datasets, valid_datasets, args):
    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        logging.info("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)

    logging.info("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, _ = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    logging.info(f"{len(valid_dataloaders)} valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
    net_without_ddp = net.module

    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        logging.info("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        logging.info(f"restore model from: {args.restore_model}")
        net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"), strict=False)

        evaluate(args, net, valid_dataloaders, valid_datasets, args.visualize)

def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    freeze_bn_stats(net.module)

    best_epoch = -1
    best_iou = 0
    for epoch in range(epoch_start,epoch_num): 
        logging.info(f'epoch:   {epoch} learning rate:  {optimizer.param_groups[0]["lr"]}')
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders,100, logger=logging):
            inputs, labels = data['image'], data['label']


            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            input_keys = ['box','point']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=net.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    assert(False)
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]

                batched_input.append(dict_input)

            batched_output, interm_embeddings = net(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)

            masks_hq = [batched_output[i_l]['low_res_logits'] for i_l in range(batch_len)]
            masks_hq = torch.cat(masks_hq, 0)
            
            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            loss = loss_mask + loss_dice

            refine_loss = torch.tensor(0.0, device=loss.device)
            
            with torch.no_grad():
                for i, box in enumerate(labels_box):
                    x1, y1, x2, y2 = box
                    box_area = (x2 - x1) * (y2 - y1)
                    img_h, img_w = imgs[i].shape[:2]
                    area_ratio = box_area / (img_h * img_w)

                    if area_ratio < 0.0001:
                        continue

                    w = (x2 - x1) / 3
                    h = (y2 - y1) / 3
                    sub_losses = []

                    for r in range(3):
                        for c in range(3):
                            sub_x1 = int(x1 + c * w)
                            sub_y1 = int(y1 + r * h)
                            sub_x2 = int(sub_x1 + w)
                            sub_y2 = int(sub_y1 + h)
                            
                            pred_patch = masks_hq[i:i + 1, :, sub_y1:sub_y2, sub_x1:sub_x2]
                            gt_patch = labels[i:i + 1, :, sub_y1:sub_y2, sub_x1:sub_x2] / 255.0

                            if pred_patch.numel() == 0:
                                sub_losses.append(torch.tensor(0.0, device=loss.device))
                                continue

                            l_mask, l_dice = loss_masks(pred_patch, gt_patch, 1)
                            sub_losses.append(l_mask + l_dice)

                    sub_losses = torch.stack(sub_losses)
                    hardest_idx = torch.argmax(sub_losses)
                    hardest_r, hardest_c = hardest_idx // 3, hardest_idx % 3
                    hardest_box = [
                        x1 + hardest_c * w,
                        y1 + hardest_r * h,
                        x1 + (hardest_c + 1) * w,
                        y1 + (hardest_r + 1) * h,
                    ]
                    hardest_box = torch.tensor(hardest_box, device=net.device).unsqueeze(0)

                    sub_input = {
                        'image': torch.as_tensor(imgs[i].astype(np.uint8), device=net.device)
                        .permute(2, 0, 1).contiguous(),
                        'boxes': hardest_box,
                        'original_size': imgs[i].shape[:2]
                    }

                    sub_output, _ = net([sub_input], multimask_output=False)
                    sub_mask_pred = sub_output[0]['low_res_logits']
                    l_mask, l_dice = loss_masks(sub_mask_pred, labels[i:i + 1] / 255.0, 1)
                    refine_loss += (l_mask + l_dice)

            total_loss = loss + refine_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_dict = {
                "loss_mask": loss_mask,
                "loss_dice": loss_dice,
                "refine_loss": refine_loss.detach()
            }
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()
            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        logging.info(f"Finished epoch:      {epoch}")
        metric_logger.synchronize_between_processes()
        logging.info(f"Averaged stats: {metric_logger}")
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats, candidate_iou = evaluate(args, net, valid_dataloaders, valid_datasets)
        train_stats.update(test_stats)
        if candidate_iou > best_iou:
            best_iou = candidate_iou
            best_epoch = epoch
        logging.info(f"Best epoch: {best_epoch}\t Best iou: {best_iou}")

        net.train()  
        freeze_bn_stats(net.module)

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            logging.info(f"come here save at {args.output + model_name}")
            misc.save_on_master(lora_state_dict(net.module), args.output + model_name)

    # Finish training
    logging.info("Training Reaches The Maximum Epoch Number")


def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

import pandas as pd

@torch.no_grad()
def evaluate(args, net, valid_dataloaders, valid_datasets, visualize=False):
    net.eval()
    logging.info("Validating...")
    test_stats = {}

    df = pd.DataFrame([], columns=['Dataset', 'IoU, BIoU'])
    avg_ious = []
    avg_bious = []
    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        logging.info(f"valid_dataloader len: {len(valid_dataloader)}")

        for data_val in metric_logger.log_every(valid_dataloader,100, logger=logging):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            
            if args.val_point_prompt == -1:
                input_keys = ['box']
            else:
                input_keys = ['point']
                labels_points = misc.masks_sample_points(labels_val[:, 0, :, :], args.val_point_prompt)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=net.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    assert(False)
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = net(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)

            masks_hq = [batched_output[i_l]['low_res_logits'] for i_l in range(batch_len)]
            masks_hq = torch.cat(masks_hq, 0)
                
            iou = compute_iou(masks_hq,labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq,labels_ori)
            
            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        logging.info('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info(f"Averaged stats: {metric_logger}")
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
        avg_ious.append(resstat['val_iou_'+str(k)])
        avg_bious.append(resstat['val_boundary_iou_'+str(k)])
        df.loc[len(df.index)] = [valid_datasets[k]['name'], f'{round(avg_ious[-1], 3)}, {round(avg_bious[-1], 3)}']  

    avg_ious = np.array(avg_ious)
    avg_bious = np.array(avg_bious)
    logging.info(f'\n {df.sort_values("Dataset").to_markdown()}')
    logging.info(f"[{args.model_type}] Final results: IoU, BIoU: {round(avg_ious.mean(), 3)}, {round(avg_bious.mean(), 3)}")
    return test_stats, avg_ious.mean()




if __name__ == "__main__":
    args = get_args_parser()
    misc.init_distributed_mode(args)
    
    if args.rank == 0:
        os.makedirs(args.output, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{args.output}/{'train.log' if not args.eval else 'val.log'}" ),
                logging.StreamHandler()
            ]
        )
    else:
        logging.info = lambda x: x

    logging.info('world size: {}'.format(args.world_size))
    logging.info('rank: {}'.format(args.rank))
    logging.info('local_rank: {}'.format(args.local_rank))
    logging.info("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True

    if "hx-25-1" in args.output:
        num = 1
    elif "hx-25-2" in args.output:
        num = 2
    elif "hx-25-3" in args.output:
        num = 3
    train_datasets = get_all_hx_train(num, [args.obj])
    valid_datasets = get_all_hx_val(num, [args.obj])
    
    net = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    state = net.state_dict()
    prepare_lora(args.model_type, net, args.lora_r)
    results = net.load_state_dict(state, strict=False)
    mark_only_lora_as_trainable(net)

    main(net, train_datasets, valid_datasets, args)
