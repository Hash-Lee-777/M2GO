#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings
import math

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from M2GO.common.solver.build import build_optimizer, build_scheduler
from M2GO.common.utils.checkpoint import CheckpointerV2
from M2GO.common.utils.logger import setup_logger
from M2GO.common.utils.metric_logger import MetricLogger
from M2GO.common.utils.torch_util import set_random_seed
from M2GO.models.build import build_model_2d, build_model_3d
from M2GO.data.build import build_dataloader
from M2GO.data.utils.validate import validate
from M2GO.models.losses import entropy_loss


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger


def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('M2GO.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build optimizer
    optimizer_2d = build_optimizer(cfg, model_2d)
    optimizer_3d = build_optimizer(cfg, model_3d)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        os.makedirs(tb_dir, exist_ok=True)
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None
    val_round_idx = 0
    total_val_rounds = math.ceil(max_iteration / val_period) if val_period > 0 else 0
    sgr_base_flag = bool(getattr(cfg.VAL, 'SGR_ENABLED', False))
    sgr_interval = 5
    sgr_last_rounds = 3

    def should_enable_sgr(val_round):
        if not sgr_base_flag:
            return False
        if sgr_interval <= 0 and sgr_last_rounds <= 0:
            return True
        enable = False
        if sgr_interval > 0 and val_round % sgr_interval == 0:
            enable = True
        if total_val_rounds > 0 and sgr_last_rounds > 0 and val_round > total_val_rounds - sgr_last_rounds:
            enable = True
        return enable

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }
    best_metric_iter = {'2d': -1, '3d': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')
    # Train-time EMA threshold for U (to match validation behavior)
    tau_ema_train = None

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    for iteration in range(start_iteration, max_iteration):
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        data_time = time.time() - end
        # copy data from cpu to gpu
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
            # target
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
                # optional confidences for head-expansion with PL
                if 'pseudo_conf_2d' in data_batch_trg and data_batch_trg['pseudo_conf_2d'] is not None:
                    data_batch_trg['pseudo_conf_2d'] = data_batch_trg['pseudo_conf_2d'].cuda()
                if 'pseudo_conf_3d' in data_batch_trg and data_batch_trg['pseudo_conf_3d'] is not None:
                    data_batch_trg['pseudo_conf_3d'] = data_batch_trg['pseudo_conf_3d'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_src)
        preds_3d = model_3d(data_batch_src)

        # segmentation loss: cross entropy
        seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        train_metric_logger.update(seg_loss_src_2d=seg_loss_src_2d, seg_loss_src_3d=seg_loss_src_3d)
        loss_2d = seg_loss_src_2d
        loss_3d = seg_loss_src_3d

        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
            # Only perform KL on non-100 points
            with torch.no_grad():
                valid_src = (data_batch_src['seg_label'] != -100)  # (N,)

            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            kl2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                            F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                            reduction='none').sum(1)
            kl3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1), 
                            F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                            reduction='none').sum(1)

            # Only average to valid_src
            denom = valid_src.float().sum().clamp_min(1.)
            xm_loss_src_2d = (kl2d * valid_src.float()).sum() / denom
            xm_loss_src_3d = (kl3d * valid_src.float()).sum() / denom

            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d

        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_src)
            train_metric_3d.update_dict(preds_3d, data_batch_src)

        # backward
        loss_2d.backward()
        loss_3d.backward()

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)

        loss_2d = []
        loss_3d = []

        # Always compute softmax for target branch to be used by KL/MinEnt, regardless of OPENSET
        P2D = torch.softmax(preds_2d['seg_logit'], dim=1)  # (N,C)
        P3D = torch.softmax(preds_3d['seg_logit'], dim=1)  # (N,C)

        # ==== A-lite: Point by point unknown degree U and obtain candidate known/unknown masks ====
        if 'OPENSET' in cfg and cfg.OPENSET.ENABLE:
            with torch.no_grad():

                def entropy(P, eps=1e-8):
                    return -(P * (P.add(eps).log())).sum(1)        # (N,)

                def kl(P, Q, eps=1e-8):
                    return (P * (P.add(eps).log() - Q.add(eps).log())).sum(1)  # (N,)
                
                H2D = entropy(P2D)
                H3D = entropy(P3D)
                M   = 0.5 * (P2D + P3D)
                JS  = 0.5 * kl(P2D, M) + 0.5 * kl(P3D, M)

                # robust 0-1 normalization (P10–P90)
                def robust_norm01(x, eps=1e-6):
                    p10 = torch.quantile(x, 0.10)
                    p90 = torch.quantile(x, 0.90)
                    x = (x - p10) / (p90 - p10 + eps)
                    return torch.clamp(x, 0.0, 1.0)

                a = cfg.OPENSET.UNK_SCORE.alpha
                b = cfg.OPENSET.UNK_SCORE.beta
                c = cfg.OPENSET.UNK_SCORE.gamma
                U = a * robust_norm01(H2D) + b * robust_norm01(H3D) + c * robust_norm01(JS)   # (N,)

                q   = cfg.OPENSET.THRESHOLD.q
                tau_b = torch.quantile(U.detach(), q)
                tau_ema_train = tau_b if (tau_ema_train is None) else (0.9 * tau_ema_train + 0.1 * tau_b)
                tau = tau_ema_train
                mask_kn  = (U <= tau)          # candidate known
                mask_unk = ~mask_kn            # candidate unknown

            # Warmup: only statistics without gate
            use_gate = (iteration >= cfg.OPENSET.WARMUP_ITERS)

            if iteration % cfg.OPENSET.LOG_HIST_EVERY == 0:
                train_metric_logger.update(unk_ratio=mask_unk.float().mean().item(), unk_tau=float(tau))
            
        else:
            # Non Open set: All considered known
            mask_kn  = torch.ones_like(preds_2d['seg_logit'][..., 0], dtype=torch.bool)
            mask_unk = ~mask_kn
            use_gate = False

        # ===== Unknown supervised CE on target (C+1) =====
        # Apply only after warmup and when enabled
        if ('OPENSET' in cfg and cfg.OPENSET.ENABLE 
            and use_gate):
            # assume the last id is Unknown
            unknown_id_2d = int(cfg.MODEL_2D.NUM_CLASSES) - 1
            unknown_id_3d = int(cfg.MODEL_3D.NUM_CLASSES) - 1

            def unknown_ce_loss(logits, mask, unknown_id):
                if mask.any():
                    num_pos = int(mask.long().sum().item())
                    target = torch.full((num_pos,), unknown_id, device=logits.device, dtype=torch.long)
                    return F.cross_entropy(logits[mask], target, weight=class_weights)
                else:
                    return torch.zeros((), device=logits.device)

            # linear ramp for lambda_unk after warmup
            lambda_unk_cfg = float(getattr(cfg.OPENSET, 'LAMBDA_UNK', 0.0))
            ramp_steps = int(getattr(cfg.OPENSET, 'UNK_RAMP_ITERS', 0))
            if lambda_unk_cfg > 0.0:
                if iteration < cfg.OPENSET.WARMUP_ITERS:
                    lambda_unk_eff = 0.0
                elif ramp_steps > 0:
                    t = (iteration - cfg.OPENSET.WARMUP_ITERS) / float(ramp_steps)
                    lambda_unk_eff = lambda_unk_cfg * float(max(0.0, min(1.0, t)))
                else:
                    lambda_unk_eff = lambda_unk_cfg
            else:
                lambda_unk_eff = 0.0

            if lambda_unk_eff > 0.0:

                # ---------------- [Start of Modification] ----------------
                # 策略：Asymmetric Cross-Modal Supervision (2D Guided 3D)
                # 逻辑：只有 2D 也高熵困惑的点，才让 3D 学 Unknown            
                with torch.no_grad():

                    # Step 1: 计算 2D 熵的 q=0.8 分位数（即高熵阈值）
                    thr_2d_guide = torch.quantile(H2D.detach(), 0.97)

                    # Step 2: 计算更严格的 3D unknown mask
                    mask_unk_3d_guided = mask_unk & (H2D.detach() > thr_2d_guide)

                # Step 3: 计算 Unknown CE Loss
                # 2D 保持原逻辑
                unk_ce_2d = unknown_ce_loss(preds_2d['seg_logit'], mask_unk, unknown_id_2d)

                # 3D 使用严格 mask
                unk_ce_3d = unknown_ce_loss(preds_3d['seg_logit'], mask_unk_3d_guided, unknown_id_3d)

                train_metric_logger.update(unk_ce_2d=unk_ce_2d,
                                        unk_ce_3d=unk_ce_3d,
                                        lambda_unk=lambda_unk_eff)

                loss_2d.append(lambda_unk_eff * unk_ce_2d)
                loss_3d.append(lambda_unk_eff * unk_ce_3d)

        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence (optionally CosMix teacher)
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            # limit KL to known classes only (exclude Unknown channel)
            known_classes = int(cfg.MODEL_2D.NUM_CLASSES) - 1

            use_cosmix = bool(getattr(cfg, 'COSMIX', None) is not None and cfg.COSMIX.ENABLE)
            if use_cosmix:
                eps = float(getattr(cfg.COSMIX, 'EPS', 1e-6))
                a_min, a_max = tuple(getattr(cfg.COSMIX, 'ALPHA_CLAMP', (0.05, 0.95)))
                a_min = float(a_min)
                a_max = float(a_max)
                # confidence proxy: max prob on known classes (teacher-only, no grad)
                conf2d = P2D[:, :known_classes].detach().max(1).values
                conf3d = P3D[:, :known_classes].detach().max(1).values
                if bool(getattr(cfg.COSMIX, 'GATE_BY_CONF', True)):
                    # Optional confidence gating: if both modalities are low-conf, fall back to alpha=0.5;
                    # if only one is high-conf, trust that modality.
                    use_conf_gate = bool(getattr(cfg, 'CONFIDENCE', None) is not None and cfg.CONFIDENCE.ENABLE)
                    if use_conf_gate:
                        conf_type = str(getattr(cfg.CONFIDENCE, 'TYPE', 'fixed'))
                        if conf_type == 'quantile':
                            q = float(getattr(cfg.CONFIDENCE, 'Q', 0.8))
                            thr2d = torch.quantile(conf2d.detach(), q)
                            thr3d = torch.quantile(conf3d.detach(), q)
                        else:
                            tau = float(getattr(cfg.CONFIDENCE, 'TAU', 0.95))
                            thr2d = tau
                            thr3d = tau
                        high2d = conf2d >= thr2d
                        high3d = conf3d >= thr3d
                        alpha = (conf2d / (conf2d + conf3d + eps)).detach()
                        alpha = torch.where(high2d & (~high3d), torch.ones_like(alpha), alpha)
                        alpha = torch.where((~high2d) & high3d, torch.zeros_like(alpha), alpha)
                        alpha = torch.where((~high2d) & (~high3d), torch.full_like(alpha, 0.5), alpha)

                        log_every = int(getattr(cfg.CONFIDENCE, 'LOG_EVERY', 500))
                        if log_every > 0 and iteration % log_every == 0:
                            train_metric_logger.update(conf_thr2d=float(thr2d) if not torch.is_tensor(thr2d) else float(thr2d.item()),
                                                       conf_thr3d=float(thr3d) if not torch.is_tensor(thr3d) else float(thr3d.item()))
                    else:
                        alpha = (conf2d / (conf2d + conf3d + eps)).detach()
                else:
                    alpha = torch.full_like(conf2d, 0.5)
                # make sure CosMix teacher does not couple 2D/3D graphs
                alpha = alpha.detach()
                alpha = torch.clamp(alpha, a_min, a_max)
                alpha_col = alpha.unsqueeze(1)  # (N,1)
                # Mixed teacher distribution (detach to avoid gradients through teacher)
                Pmix = (alpha_col * P2D.detach() + (1.0 - alpha_col) * P3D.detach())[:, :known_classes]
                kl2d_cos = F.kl_div(F.log_softmax(seg_logit_2d[:, :known_classes], dim=1),
                                    Pmix, reduction='none').sum(1)
                kl3d_cos = F.kl_div(F.log_softmax(seg_logit_3d[:, :known_classes], dim=1),
                                    Pmix, reduction='none').sum(1)

                if bool(getattr(cfg.COSMIX, 'REPLACE_XM', True)):
                    kl2d = kl2d_cos
                    kl3d = kl3d_cos
                else:
                    # also keep the original cross-modal KL, and add CosMix as an extra regularizer
                    P3D_known = P3D[:, :known_classes].detach()
                    P2D_known = P2D[:, :known_classes].detach()
                    kl2d_base = F.kl_div(F.log_softmax(seg_logit_2d[:, :known_classes], dim=1),
                                         P3D_known, reduction='none').sum(1)
                    kl3d_base = F.kl_div(F.log_softmax(seg_logit_3d[:, :known_classes], dim=1),
                                         P2D_known, reduction='none').sum(1)
                    kl2d = kl2d_base + kl2d_cos
                    kl3d = kl3d_base + kl3d_cos
            else:
                P3D_known = P3D[:, :known_classes].detach()
                P2D_known = P2D[:, :known_classes].detach()
                kl2d = F.kl_div(F.log_softmax(seg_logit_2d[:, :known_classes], dim=1),
                                P3D_known, reduction='none').sum(1)
                kl3d = F.kl_div(F.log_softmax(seg_logit_3d[:, :known_classes], dim=1),
                                P2D_known, reduction='none').sum(1)

            if cfg.OPENSET.ENABLE and cfg.OPENSET.GATE_XM and use_gate:
                denom = mask_kn.float().sum().clamp_min(1.)
                xm_loss_trg_2d = (kl2d * mask_kn.float()).sum() / denom
                xm_loss_trg_3d = (kl3d * mask_kn.float()).sum() / denom
            else:
                xm_loss_trg_2d = kl2d.mean()
                xm_loss_trg_3d = kl3d.mean()

            if use_cosmix:
                xm_lambda = float(cfg.TRAIN.XMUDA.lambda_xm_trg) * float(getattr(cfg.COSMIX, 'LAMBDA', 1.0))
                train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                           xm_loss_trg_3d=xm_loss_trg_3d,
                                           cosmix_alpha_mean=float(alpha.mean().item()))
            else:
                xm_lambda = float(cfg.TRAIN.XMUDA.lambda_xm_trg)
                train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                           xm_loss_trg_3d=xm_loss_trg_3d)

            loss_2d.append(xm_lambda * xm_loss_trg_2d)
            loss_3d.append(xm_lambda * xm_loss_trg_3d)

        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            ignore = -100
            # uni-modal self-training loss with pseudo labels
            pl2d = data_batch_trg['pseudo_label_2d'].clone()
            pl3d = data_batch_trg['pseudo_label_3d'].clone()
            # Head-expansion with PL: map low-confidence PL to Unknown (C)
            if hasattr(cfg.TRAIN.XMUDA, 'PL_TO_UNK') and cfg.TRAIN.XMUDA.PL_TO_UNK.enable:
                unknown_id_2d = int(cfg.MODEL_2D.NUM_CLASSES) - 1
                unknown_id_3d = int(cfg.MODEL_3D.NUM_CLASSES) - 1
                # thresholds for 2D
                if 'pseudo_conf_2d' in data_batch_trg and data_batch_trg['pseudo_conf_2d'] is not None:
                    conf2d = data_batch_trg['pseudo_conf_2d']
                    # per-modality override: thr2d / q2d
                    if getattr(cfg.TRAIN.XMUDA.PL_TO_UNK, 'thr2d', None) is not None:
                        thr2d = float(cfg.TRAIN.XMUDA.PL_TO_UNK.thr2d)
                    elif cfg.TRAIN.XMUDA.PL_TO_UNK.thr is not None:
                        thr2d = float(cfg.TRAIN.XMUDA.PL_TO_UNK.thr)
                    else:
                        q2d = getattr(cfg.TRAIN.XMUDA.PL_TO_UNK, 'q2d', None)
                        if q2d is not None:
                            thr2d = torch.quantile(conf2d.detach(), float(q2d))
                        elif cfg.TRAIN.XMUDA.PL_TO_UNK.q is not None:
                            thr2d = torch.quantile(conf2d.detach(), float(cfg.TRAIN.XMUDA.PL_TO_UNK.q))
                        else:
                            # safe fallback when all quantiles are None
                            thr2d = torch.quantile(conf2d.detach(), 0.2)
                    low2d = (conf2d < thr2d)
                    # Only re-label to Unknown where U is high (intersection with mask_unk)
                    map2d = (low2d & mask_unk)
                    pl2d[map2d] = unknown_id_2d
                    # log ratio
                    if map2d.numel() > 0:
                        train_metric_logger.update(pl_low_ratio_2d=map2d.float().mean())
                # thresholds for 3D (if available)
                if 'pseudo_conf_3d' in data_batch_trg and data_batch_trg['pseudo_conf_3d'] is not None:
                    conf3d = data_batch_trg['pseudo_conf_3d']
                    # per-modality override: thr3d / q3d
                    if getattr(cfg.TRAIN.XMUDA.PL_TO_UNK, 'thr3d', None) is not None:
                        thr3d = float(cfg.TRAIN.XMUDA.PL_TO_UNK.thr3d)
                    elif cfg.TRAIN.XMUDA.PL_TO_UNK.thr is not None:
                        thr3d = float(cfg.TRAIN.XMUDA.PL_TO_UNK.thr)
                    else:
                        q3d = getattr(cfg.TRAIN.XMUDA.PL_TO_UNK, 'q3d', None)
                        if q3d is not None:
                            thr3d = torch.quantile(conf3d.detach(), float(q3d))
                        elif cfg.TRAIN.XMUDA.PL_TO_UNK.q is not None:
                            thr3d = torch.quantile(conf3d.detach(), float(cfg.TRAIN.XMUDA.PL_TO_UNK.q))
                        else:
                            # safe fallback when all quantiles are None
                            thr3d = torch.quantile(conf3d.detach(), 0.2)
                    low3d = (conf3d < thr3d)
                    map3d = (low3d & mask_unk)
                    pl3d[map3d] = unknown_id_3d
                    if map3d.numel() > 0:
                        train_metric_logger.update(pl_low_ratio_3d=map3d.float().mean())
            # Gate PL: ignore only high-U points that are not already mapped to Unknown by low-conf
            if ('OPENSET' in cfg and cfg.OPENSET.ENABLE and cfg.OPENSET.GATE_PL and use_gate):
                if 'pseudo_conf_2d' in data_batch_trg and data_batch_trg['pseudo_conf_2d'] is not None and \
                   hasattr(cfg.TRAIN.XMUDA, 'PL_TO_UNK') and cfg.TRAIN.XMUDA.PL_TO_UNK.enable:
                    # recompute low2d mask
                    conf2d = data_batch_trg['pseudo_conf_2d']
                    thr2d_cfg = getattr(cfg.TRAIN.XMUDA.PL_TO_UNK, 'thr2d', None)
                    if thr2d_cfg is not None:
                        thr2d = float(thr2d_cfg)
                    elif cfg.TRAIN.XMUDA.PL_TO_UNK.thr is not None:
                        thr2d = float(cfg.TRAIN.XMUDA.PL_TO_UNK.thr)
                    else:
                        q2d = getattr(cfg.TRAIN.XMUDA.PL_TO_UNK, 'q2d', None)
                        if q2d is not None:
                            thr2d = torch.quantile(conf2d.detach(), float(q2d))
                        elif cfg.TRAIN.XMUDA.PL_TO_UNK.q is not None:
                            thr2d = torch.quantile(conf2d.detach(), float(cfg.TRAIN.XMUDA.PL_TO_UNK.q))
                        else:
                            thr2d = torch.quantile(conf2d.detach(), 0.2)
                    low2d = (conf2d < thr2d)
                    ignore2d = (mask_unk & (~low2d))
                    pl2d[ignore2d] = ignore
                else:
                    pl2d[mask_unk] = ignore
                if 'pseudo_conf_3d' in data_batch_trg and data_batch_trg['pseudo_conf_3d'] is not None and \
                   hasattr(cfg.TRAIN.XMUDA, 'PL_TO_UNK') and cfg.TRAIN.XMUDA.PL_TO_UNK.enable:
                    conf3d = data_batch_trg['pseudo_conf_3d']
                    thr3d_cfg = getattr(cfg.TRAIN.XMUDA.PL_TO_UNK, 'thr3d', None)
                    if thr3d_cfg is not None:
                        thr3d = float(thr3d_cfg)
                    elif cfg.TRAIN.XMUDA.PL_TO_UNK.thr is not None:
                        thr3d = float(cfg.TRAIN.XMUDA.PL_TO_UNK.thr)
                    else:
                        q3d = getattr(cfg.TRAIN.XMUDA.PL_TO_UNK, 'q3d', None)
                        if q3d is not None:
                            thr3d = torch.quantile(conf3d.detach(), float(q3d))
                        elif cfg.TRAIN.XMUDA.PL_TO_UNK.q is not None:
                            thr3d = torch.quantile(conf3d.detach(), float(cfg.TRAIN.XMUDA.PL_TO_UNK.q))
                        else:
                            thr3d = torch.quantile(conf3d.detach(), 0.2)
                    low3d = (conf3d < thr3d)
                    ignore3d = (mask_unk & (~low3d))
                    pl3d[ignore3d] = ignore
                else:
                    pl3d[mask_unk] = ignore

            pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'], pl2d, ignore_index=ignore)
            pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'], pl3d, ignore_index=ignore)
            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)

        if cfg.TRAIN.XMUDA.lambda_minent > 0:
            # MinEnt
            if ('OPENSET' in cfg and cfg.OPENSET.ENABLE and cfg.OPENSET.GATE_MINENT and use_gate):
                ent2d = -(P2D * (P2D.add(1e-8).log())).sum(1)
                ent3d = -(P3D * (P3D.add(1e-8).log())).sum(1)
                minent_loss_trg_2d = ent2d[mask_unk].mean()
                minent_loss_trg_3d = ent3d[mask_unk].mean()
            else:
                minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d['seg_logit'], dim=1))
                minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d['seg_logit'], dim=1))

            train_metric_logger.update(minent_loss_trg_2d=minent_loss_trg_2d,
                                       minent_loss_trg_3d=minent_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_3d)

        sum(loss_2d).backward()
        sum(loss_3d).backward()

        optimizer_2d.step()
        optimizer_3d.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            val_round_idx += 1
            sgr_enabled_now = should_enable_sgr(val_round_idx)
            total_rounds_display = total_val_rounds if total_val_rounds > 0 else '?'
            logger.info('Validation #{}/{} | SGR {}'.format(
                val_round_idx,
                total_rounds_display,
                'ON' if sgr_enabled_now else 'OFF'))

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger,
                     sgr_enabled=sgr_enabled_now)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from M2GO.common.config import purge_cfg
    from M2GO.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        os.makedirs(output_dir, exist_ok=True)

    # run name & per-run subdir (keep all artifacts inside)
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)
    run_dir = osp.join(output_dir, 'train.{:s}'.format(run_name)) if output_dir else ''
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger('M2GO', run_dir or output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, run_dir or output_dir, run_name)


if __name__ == '__main__':
    main()