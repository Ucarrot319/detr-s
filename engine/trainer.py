import torch
from torch.utils.data import DataLoader, DistributedSampler
import time
import json
import datetime
from typing import Iterable
import math
import sys

from models import build_model
import util.misc as utils
from .loss import build_criterion
from data.datasets import build_dataset, get_coco_api_from_dataset

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model, self.criterion, self.postprocessors = self.build_mcp(args)
        self.model.to(self.device)
        self.model_without_ddp = self.model.module if args.distributed else self.model
        self.optimizer, self.lr_scheduler = self.build_optimizer(args)
        self.dataset_train = build_dataset(image_set='train', args=args)
        self.dataset_val = build_dataset(image_set='val', args=args)
        self.data_loader_train = self.build_data_loader(self.dataset_train, is_train=True)
        self.data_loader_val = self.build_data_loader(self.dataset_val, is_train=False)
        self.base_ds = get_coco_api_from_dataset(self.dataset_val)

    def build_mcp(self, args):
        # Build your model here
        device = torch.device(args.device)
        model, postprocessors = build_model(args)
        criterion = build_criterion(args)
        criterion.to(device)
        return model, criterion, postprocessors

    def build_optimizer(self, args):
        # Build your optimizer and lr_scheduler here
        param_dicts = [
            {"params": [p for n, p in self.model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        return optimizer, lr_scheduler

    def build_data_loader(self, dataset, is_train):
        if self.args.distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset) if is_train else torch.utils.data.SequentialSampler(dataset)

        if is_train:
            batch_sampler = torch.utils.data.BatchSampler(sampler, self.args.batch_size, drop_last=True)
            data_loader = DataLoader(dataset, batch_sampler=batch_sampler,collate_fn=utils.collate_fn,
                                     num_workers=self.args.num_workers)
        else:
            data_loader = DataLoader(dataset, self.args.batch_size, sampler=sampler, drop_last=False,
                                        collate_fn=utils.collate_fn, num_workers=self.args.num_workers)
        return data_loader

    def train_one_epoch(self, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
        model.train()
        criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10

        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train(self):
        print("Start training")
        start_time = time.time()
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.data_loader_train.sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(epoch)
            self.lr_scheduler.step()
            if self.args.output_dir:
                checkpoint_paths = [self.args.output_dir / 'checkpoint.pth']
                if (epoch + 1) % self.args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(self.args.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    self.save_checkpoint(checkpoint_path, epoch)

            test_stats = self.evaluate()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch}

            if self.args.output_dir and utils.is_main_process():
                with (self.args.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # if self.base_ds is not None:
                #     (self.args.output_dir / 'eval').mkdir(exist_ok=True)
                #     if "bbox" in self.base_ds:
                #         filenames = ['latest.pth']
                #         if epoch % 50 == 0:
                #             filenames.append(f'{epoch:03}.pth')
                #         for name in filenames:
                #             torch.save(self.base_ds["bbox"].eval, self.args.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))