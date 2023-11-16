import torch
import torch.nn.functional as TorchFunction
from torch.utils.data import DataLoader
# import numpy as np
from tqdm import tqdm
import os
# import cv2

import logging
from utils import *
from options import parse_args
from model import Model
from floorplan_dataset import FloorplanDataset
from IP import reconstructFloorplan


def main(options):
    if not os.path.exists(options.checkpoint_dir):
        try:
            os.system("mkdir -p %s" % options.checkpoint_dir)
        except Exception as e:
            os.mkdir(options.checkpoint_dir)
    
    if not os.path.exists(options.test_dir):
        try:
            os.system("mkdir -p %s" % options.test_dir)
        except Exception as e:
            os.mkdir(options.test_dir)
    
    trandata_dir = '/Users/hehao/Desktop/Henry/IKEA/Prometheus/IKEA_img2floorplan/models/datasets/'
    dataset = FloorplanDataset(options, 'train', trandata_dir, random=True)
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=True, num_workers=16)
    
    model = Model(options)
    # model.cuda()
    model.train()
    
    if options.gpuFlag == 1:
        if not torch.cuda.is_available():
            options.gpuFlag = 0
    
    if options.restore == 1:  # 1表示继续训练
        print('restore')
        checkpoint_file_path = os.path.join(options.checkpoint_dir, 'checkpoint.pth')
        model.load_state_dict(torch.load(checkpoint_file_path, map_location='cpu'))
        pass
    
    if options.task == 'test':
        dataset_test = FloorplanDataset(options, split='test', random=False)
        testOneEpoch(options, model, dataset_test)
        exit(1)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=options.LR)
    optimizer = torch.optim.Adam(model.parameters())
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim.pth'))
        pass
    
    for epoch in range(options.numEpochs):
        epoch_losses = []
        data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
        for sampleIndex, sample in enumerate(data_iterator):
            optimizer.zero_grad()
            
            images, corner_gt, icon_gt, room_gt = sample[0], sample[1], sample[2], sample[3]  # gt表示groundTruth标注数据
            corner_pred, = model(images)
            corner_loss = torch.nn.functional.binary_cross_entropy(corner_pred, corner_gt)  # 交叉熵损失函数
            a = True if epoch % 5 == 0 else False
            if epoch != 0 and a:
                check_predict_data(options, corner_pred, corner_gt)
            losses = [corner_loss]
            loss = sum(losses)
            
            loss_values = [l.data.item() for l in losses]
            epoch_losses.append(loss_values)
            status = str(epoch + 1) + ' loss: '
            for l in loss_values:
                status += '%0.5f ' % l
                continue
            data_iterator.set_description(status)
            loss.backward()
            optimizer.step()
            
            if sampleIndex % 500 == 0:
                visualizeBatch(
                    options, images.detach().cpu().numpy(), [('gt',
                    {'corner': corner_gt.detach().cpu().numpy()}),
                        (
                            'pred', {
                                'corner': corner_pred.max(-1)[
                                    1].detach().cpu().numpy()
                            })]
                    )
            continue
        print('loss', np.array(epoch_losses).mean(0))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint_{0}.pth'.format(epoch))
            torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_{0}.pth'.format(epoch))
            pass
        
        # testOneEpoch(options, model, dataset_test)
        continue
    return


def check_predict_data(options, corner_pred, corner_gt):
    try:
        max_diff = 0.0
        max_count = 0
        
        largest_diff = -100.0
        
        min_diff = 0.0
        min_count = 0
        for batchIndex in range(corner_pred.shape[0]):
            tmp_corner_index = corner_pred[batchIndex]
            tmp_corner_data = tmp_corner_index.detach().cpu().numpy()
            
            tmp_corner_gt_data = corner_gt[batchIndex].detach().cpu().numpy()
            
            for i in range(NUM_CORNERS):
                tmp_corner_pred_heatmap = tmp_corner_data[:, :, i]
                max_pred_value = np.max(tmp_corner_pred_heatmap)
                tmp_corner_gt_heatmap = tmp_corner_gt_data[:, :, i]
                max_gt_value = np.max(tmp_corner_gt_heatmap)
                
                if max_gt_value > 0.5:
                    diff = max_gt_value - max_pred_value
                    max_diff += diff
                    if diff > largest_diff:
                        largest_diff = diff
                    max_count += 1
                else:
                    min_diff += max_pred_value
                    min_count += 1
        # fpLog.info("Avg Diff = {0}, Max Diff = {1}".format(max_diff / max_count, largest_diff))
        print("Avg Diff = {0}, Max Diff = {1}".format(max_diff / max_count, largest_diff))
    except Exception as err:
        logging.error(err)


def testOneEpoch(options, model, dataset):
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)
    
    epoch_losses = []
    data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
    for sampleIndex, sample in enumerate(data_iterator):
        
        images, corner_gt, icon_gt, room_gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda()
        
        corner_pred, icon_pred, room_pred = model(images)
        corner_loss = torch.nn.functional.binary_cross_entropy(corner_pred, corner_gt)
        icon_loss = torch.nn.functional.cross_entropy(icon_pred.view(-1, NUM_ICONS + 2), icon_gt.view(-1))
        room_loss = torch.nn.functional.cross_entropy(room_pred.view(-1, NUM_ROOMS + 2), room_gt.view(-1))
        losses = [corner_loss, icon_loss, room_loss]
        
        loss = sum(losses)
        
        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f ' % l
            continue
        data_iterator.set_description(status)
        
        if sampleIndex % 500 == 0:
            visualizeBatch(
                options, images.detach().cpu().numpy(), [('gt', {
                    'corner': corner_gt.detach().cpu().numpy(),
                    'icon': icon_gt.detach().cpu().numpy(),
                    'room': room_gt.detach().cpu().numpy()
                }), (
                    'pred', {
                        'corner': corner_pred.max(-1)[
                            1].detach().cpu().numpy(),
                        'icon': icon_pred.max(-1)[
                            1].detach().cpu().numpy(),
                        'room': room_pred.max(-1)[
                            1].detach().cpu().numpy()
                    })]
                )
            for batchIndex in range(len(images)):
                corner_heatmaps = corner_pred[batchIndex].detach().cpu().numpy()
                icon_heatmaps = torch.nn.functional.softmax(icon_pred[batchIndex], dim=-1).detach().cpu().numpy()
                room_heatmaps = torch.nn.functional.softmax(room_pred[batchIndex], dim=-1).detach().cpu().numpy()
                reconstructFloorplan(
                    corner_heatmaps[:, :, :NUM_WALL_CORNERS],
                    corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4],
                    corner_heatmaps[:, :, -4:], icon_heatmaps, room_heatmaps,
                    output_prefix=options.test_dir + '/' + str(batchIndex) + '_', densityImage=None,
                    gt_dict=None, gt=False, gap=-1, distanceThreshold=-1, lengthThreshold=-1,
                    debug_prefix='test', heatmapValueThresholdWall=None,
                    heatmapValueThresholdDoor=None, heatmapValueThresholdIcon=None,
                    enableAugmentation=True
                    )
                continue
            if options.visualizeMode == 'debug':
                exit(1)
                pass
        continue
    print('validation loss', np.array(epoch_losses).mean(0))
    
    model.train()
    return


def visualizeBatch(options, images, dicts, indexOffset=0, prefix=''):
    # cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    # pointColorMap = ColorPalette(20).getColorMap()
    images = ((images.transpose((0, 2, 3, 1)) + 0.5) * 255).astype(np.uint8)
    for batchIndex in range(len(images)):
        image = images[batchIndex].copy()
        filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_image.png'
        cv2.imwrite(filename, image)
        for name, result_dict in dicts:
            for info in ['corner']:
                cv2.imwrite(
                    filename.replace('image', info + '_' + name),
                    drawSegmentationImage(result_dict[info][batchIndex], blackIndex=0, blackThreshold=0.5)
                    )
                continue
            continue
        continue
    return


if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'floorplan'
    args.restore = 1  # restore是0表示从头开始训练，1表示从上一次训练的模型继续训练
    cur_folder_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_folder_path = os.path.join(cur_folder_path, "checkpoint")
    args.checkpoint_dir = 'checkpoint/'
    args.checkpoint_dir = checkpoint_folder_path
    args.test_dir = 'test/' + args.keyname
    args.model_type = 1
    args.batchSize = 4
    args.outputWidth = 512
    args.outputHeight = 512
    args.restore = 0
    args.numEpochs = 500
    logging.info('keyname=%s task=%s started' % (args.keyname, args.task))
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
        pass
    main(args)
