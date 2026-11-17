import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        if os.path.isabs(cfg['save_path']):  # [UAV port] absolute --out directory
            self.output_dir = cfg['save_path']
        self.tester = None
        # [UAV port] smoke-test / bookkeeping hooks (no effect on optimisation)
        self.max_iters = int(cfg.get('max_iters', 0) or 0)
        self.keep_every = int(cfg.get('keep_every', 0) or 0)
        self.global_iter = 0
        self.loss_log = None
        self.n_skipped = 0  # [UAV port] steps skipped because the loss / gradients were not finite
        if cfg.get('loss_log'):
            os.makedirs(os.path.dirname(cfg['loss_log']), exist_ok=True)
            self.loss_log = open(cfg['loss_log'], 'a')

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger,
                lr_scheduler=self.lr_scheduler)  # [UAV port] scheduler state restored too (if stored)
            # [UAV port] official sets last_epoch = epoch - 1, which delays every later decay by one epoch after a resume
            # (the saved checkpoint is written after scheduler.step()). The scheduler state is restored above, so
            # last_epoch == epoch already; set it explicitly for old checkpoints without scheduler state.
            self.lr_scheduler.last_epoch = self.epoch
            self.logger.info("[UAV port] resumed at epoch %d, lr %s" % (self.epoch, [g['lr'] for g in self.optimizer.param_groups]))
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        
    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1
            if self.max_iters and self.global_iter >= self.max_iters:
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch,
                                                     self.lr_scheduler),
                                os.path.join(self.output_dir, 'checkpoint'))
                self.logger.info("max_iters %d reached, stop" % self.max_iters)
                break

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
               
                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch,
                                         self.lr_scheduler),
                    ckpt_name)
                # [UAV port] model-only snapshot every keep_every epochs AND always at the last epoch
                # (the queue treats model_epoch_<max_epoch>.pth as "training finished")
                if (self.keep_every and self.epoch % self.keep_every == 0) or self.epoch == self.cfg['max_epoch']:  # [UAV port] model-only snapshots for selection on sim_val
                    save_checkpoint(get_checkpoint_state(self.model, None, self.epoch, best_result, best_epoch),
                                    os.path.join(self.output_dir, 'model_epoch_%d' % self.epoch))

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            targets = self.prepare_targets(targets, inputs.shape[0])
            ##dn
            dn_args = None
            if self.cfg["use_dn"]:
                dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])
            ###
            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
            mask_dict=None
            #ipdb.set_trace()
            detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

            weight_dict = self.detr_loss.weight_dict
            detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
            detr_losses = sum(detr_losses_dict_weighted)

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            flags = [True] * 5
            if batch_idx % 30 == 0:
                print("----", batch_idx, "----")
                print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " %(key, val), end="")
                print("")
                print("")

            # [UAV port] non-finite guard (no interactive debugging in background runs): skip this update and log it
            if not torch.isfinite(detr_losses):
                self.n_skipped += 1
                self.logger.info("[UAV port] non-finite loss at epoch %d iter %d (%s) -> step skipped (total skipped %d)" % (
                    epoch, batch_idx, {k: v for k, v in detr_losses_dict_log.items() if not np.isfinite(v)}, self.n_skipped))
                self.optimizer.zero_grad()
            else:
                detr_losses.backward()
                # total grad norm only for the finiteness check (max_norm=inf -> grads are NOT rescaled)
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                if not torch.isfinite(gnorm):
                    self.n_skipped += 1
                    self.logger.info("[UAV port] non-finite gradient at epoch %d iter %d -> step skipped (total skipped %d)" % (
                        epoch, batch_idx, self.n_skipped))
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()

            self.global_iter += 1
            if self.loss_log is not None:
                import json
                rec = {'it': self.global_iter, 'epoch': epoch, 'loss_detr': detr_losses_dict_log['loss_detr']}
                rec.update({k: v for k, v in detr_losses_dict_log.items() if k[-2] != '_' and 'inter' not in k})
                self.loss_log.write(json.dumps(rec) + chr(10)); self.loss_log.flush()
            if self.max_iters and self.global_iter >= self.max_iters:
                break

            progress_bar.update()
        progress_bar.close()

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list

