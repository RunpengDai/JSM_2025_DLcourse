import os
import argparse
import time
import itertools
import random
import shutil
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn

from config import config as cfg
from dataset_v2 import NYCDataset
from net import STFORMER
from transformers import InformerConfig, InformerModel
from transformers import AutoformerConfig, AutoformerForPrediction

from utils.misc import mkdir

from utils.logger import setup_logger
from utils.collect_env import collect_env_info
from utils import comm




def main():
    global best_wmape
    global best_mape
    global best_rmse
    global best_wmape_epoch
    global best_mape_epoch
    global best_rmse_epoch
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = False

    if comm.is_main_process() and cfg.output_dir:
        mkdir(cfg.output_dir)
        mkdir(cfg.checkpoint_dir)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    # setup logger
    logger = setup_logger("cikm-short", cfg.output_dir, comm.get_rank(),
                          filename='train_log.txt')

    logger.info("Rank of current process: {}. World size: {}".format(
                comm.get_rank(), comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))
    shutil.copyfile('./config.py', cfg.output_dir+'/config.py' )
    shutil.copyfile('./net.py', cfg.output_dir+'/net.py' )

    train_dataset = NYCDataset(cfg, is_train=True)
    test_dataset = NYCDataset(cfg, is_train=False)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            sampler=train_sampler,
            pin_memory=False,
            drop_last=False,
            shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            sampler=test_sampler,
            drop_last=False,
            shuffle=False)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=False,
            shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=False,
            shuffle=False)
    # model
    configuration = AutoformerConfig(prediction_length=1, input_size=399, lags_sequence=[1], num_time_features=1, encoder_ffn_dim = 4,decoder_ffn_dim = 4, d_model=4, encoder_layers=1, decoder_layers=1)
    model = AutoformerForPrediction(configuration)
    print("finishing the model")
    data = [data for data in train_dataloader]

    print("past_value", data[0]["dec_x"].flatten(start_dim=-2).shape,"past_time_features", data[0]["enc_time"].unsqueeze(-1).shape, "future_time_features", data[0]["dec_time"].unsqueeze(-1).shape, "future_values", data[0]["dec_y"].flatten(start_dim=-2).shape)


    model(past_values = data[0]["dec_x"].flatten(start_dim=-2), past_observed_mask=None, past_time_features = data[0]["enc_time"].unsqueeze(-1), future_time_features = data[0]["dec_time"].unsqueeze(-1), future_values = data[0]["dec_y"].flatten(start_dim=-2), static_categorical_features=None, static_real_features =None)

    #model = STFORMER(cfg).to(torch.float32)
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
            #nn.init.kaiming_normal_(p)
            #nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p)
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'{total_trainable_params:,} training parameters.')
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True)
    # import pdb; pdb.set_trace()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, betas=(0.5, 0.999))
    print(args.resume)

    if args.resume:
        resume_path = cfg.checkpoint_dir + '/model_best.pth.tar'
        if os.path.isfile(resume_path):
            logger.info("=> loading checkpoint '{}'".format(resume_path))
        # Map model to be loaded to specified single gpu.

        # Load the new state dictionary into the model
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume_path, checkpoint['epoch']))
        logger.info("==> testing")
        # evaluate on validation set
        wmape, mape, rmse, _, _ = validate(logger,
                                     test_dataloader,
                                     model,
                                     args,
                                     cfg)

    # training loop
    logger.info("----------Start Training----------")

    for epoch in range(args.start_epoch, cfg.SOLVER.MAX_EPOCH):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, cfg, t=10)

        # train for one epoch
        train(logger, train_dataloader, model, optimizer, epoch, args)

        # evaluate on validation set
        wmape, mape, rmse, test_gts, test_preds = validate(logger, test_dataloader, model, args, cfg)

        if comm.is_main_process():
            # remember best mape and save checkpoint
            if wmape < best_wmape:
                best_wmape_epoch = epoch
            if mape < best_mape:
                best_mape_epoch = epoch
            if rmse < best_rmse:
                best_rmse_epoch = epoch
            is_best = rmse <= best_rmse
            best_wmape = min(wmape, best_wmape)
            best_mape = min(mape, best_mape)
            best_rmse = min(rmse, best_rmse)
        logger.info(f"best_wmape {best_wmape:.3f} at epoch {best_wmape_epoch}")
        logger.info(f"best_mape {best_mape:.3f} at epoch {best_mape_epoch}")
        logger.info(f"best_rmse {best_rmse:.3f} at epoch {best_rmse_epoch}")
        if comm.is_main_process():
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mape': best_mape,
                'best_wmape': best_wmape,
                'best_rmse': best_rmse,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=cfg.checkpoint_dir + '/checkpoint.pth.tar')
            save_inference(is_best, test_gts, test_preds, base_path = cfg.checkpoint_dir)


def train(logger, train_loader, model, optimizer, epoch, args):
    len_loader = len(train_loader)
    # switch to train mode
    model.train()

    end = time.time()
    epoch_b_time = time.time()
    for i, (batch_data) in enumerate(train_loader):
        # compute loss
        loss = model(batch_data)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()
        lr = optimizer.param_groups[0]['lr']
        if i % args.print_freq == 0:
            i = "%03d" % i
            logger.info(f"Train:[{epoch}][{i}/{len_loader}] batch_time: {batch_time:.4f} loss: {loss:.4f} lr: {lr:.6f}")
    epoch_time = time.time() - epoch_b_time
    logger.info("epoch time : {}".format(epoch_time))


def validate(val_loader, model, args, cfg):
    len_loader = len(val_loader)
    # switch to evaluate mode
    model.eval()
    gts = []
    preds = []

    with torch.no_grad():
        end = time.time()
        for i, (batch_data) in enumerate(val_loader):
            # compute output
            outs, targets, loss = model(batch_data)
            # img_idx
            img_idxs = batch_data["img_idx"]
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for img_idx, out, target in zip(img_idxs, outs, targets):
                out_result = {}
                target_result = {}
                out_result[img_idx.item()] = out.cpu()
                target_result[img_idx.item()] = target.cpu()
                preds.append(out_result)
                gts.append(target_result)

            # measure accuracy and record loss
            wmape, rmse = accuracy(out, target)

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if i % args.print_freq == 0:
                i = "%03d" % i
                print(f"Test: [{i}/{len_loader}] batch_time: {batch_time:.4f} loss: {loss:.4f}")

    preds_result = []
    gts_result = []
    gts = list(sorted(gts, key=lambda a: list(a.keys())))
    preds = list(sorted(preds, key=lambda a: list(a.keys())))
    for gt, pred in zip(gts, preds):
        for key in gt.keys():
            gt = gt[key]
            pred = pred[key]
            preds_result.append(pred)
            gts_result.append(gt)
    gts = torch.stack(gts_result, dim=0)
    preds = torch.stack(preds_result, dim=0)
    rmse = RMSE(preds, gts, 0)
    wmape = WMAPE(preds, gts, 0)
    mape = MAPE(preds, gts, 0)
    cpc = CPC(preds, gts, 0)
    print(f'Current WMAPE {wmape:.3f} MAPE {mape:.3f} RMSE {rmse:.3f} CPC {cpc:.3f}')
    return wmape, mape, rmse, gts.numpy(), preds.numpy()


def RMSE(preds, gts, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(gts, mask_value)
        preds = torch.masked_select(preds, mask)
        gts = torch.masked_select(gts, mask)
    return torch.sqrt(torch.mean((preds - gts) ** 2))


def WMAPE(preds, gts, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(gts, mask_value)
        preds = torch.masked_select(preds, mask)
        gts = torch.masked_select(gts, mask)
    return torch.sum(torch.abs(preds - gts) / gts.sum())

def CPC(preds, gts, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(gts, mask_value)
        preds = torch.masked_select(preds, mask)
        gts = torch.masked_select(gts, mask)
    _tmp = torch.minimum(preds, gts)
    cpc = (2 * torch.sum(_tmp)) / (torch.sum(gts ) + torch.sum(preds ))
    return cpc


def MAPE(preds, gts, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(gts, mask_value)
        preds = torch.masked_select(preds, mask)
        gts = torch.masked_select(gts, mask)
    return torch.mean(torch.abs(torch.div((gts - preds), gts)))


def save_checkpoint(state, is_best, filename=cfg.checkpoint_dir + '/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, cfg.checkpoint_dir + '/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, cfg, t):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    #if epoch < t:
    #    alpha = (0.9 * epoch/t + 0.1)
    #else:
        #alpha = 0.5*(1+math.cos(math.pi*(epoch - t)/ (cfg.SOLVER.MAX_EPOCH - t)))
    alpha = (0.5 ** (epoch // 15))
    lr = cfg.SOLVER.LR * alpha

    #lr = cfg.SOLVER.LR * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(preds, gts):
    """Compute MAPE and RMSE"""
    with torch.no_grad():
        mape = MAPE(preds, gts, 0)
        rmse = RMSE(preds, gts)
    return mape, rmse

def save_inference(is_best, gts, preds, base_path = cfg.checkpoint_dir):
    if is_best:
        np.save(base_path + '/gts.npy', gts)
        np.save(base_path + '/preds.npy', preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training')
    parser.add_argument('--resume', '-r', action='store_true', default=False,
                        help='resume from checkpoint')
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--seed", type=int, default=1227, help="random seed")
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    args = parser.parse_args()
    best_wmape = float('inf')
    best_mape = float('inf')
    best_rmse = float('inf')
    best_wmape_epoch = float('inf')
    best_rmse_epoch = float('inf')
    best_mape_epoch = float('inf')
    main()
