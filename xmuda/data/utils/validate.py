import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

from xmuda.data.utils.evaluate import Evaluator


def validate(cfg,
             model_2d,
             model_3d,
             dataloader,
             val_metric_logger,
             pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None

    # --- Open-set evaluators: 5 known + 1 unknown ---
    if cfg.OPENSET.ENABLE:
        class_names_os = list(class_names) + ['unknown']
        evaluator_os_2d = Evaluator(class_names_os)
        evaluator_os_3d = Evaluator(class_names_os) if model_3d else None
        evaluator_os_ens = Evaluator(class_names_os) if model_3d else None
    else:
        evaluator_os_2d = evaluator_os_3d = evaluator_os_ens = None

    # ===== Open-set: init =====
    openset_enabled = hasattr(cfg, 'OPENSET') and getattr(cfg.OPENSET, 'ENABLE', False)
    open_acc = {}
    for m in ['2d', '3d', 'fused']:
        open_acc[m] = {
            'inter': torch.zeros(6, dtype=torch.long, device='cpu'),
            'union': torch.zeros(6, dtype=torch.long, device='cpu'),
        }
    unk_ratio_sum, n_points_sum = 0.0, 0
    tau_ema = None  # Validation period quantile threshold EMA

    if openset_enabled:
        if hasattr(dataloader.dataset, 'fine_class_names'):
            fine_order = list(dataloader.dataset.fine_class_names)
        else:
            fine_order = [
                "car","truck","bus","trailer","construction_vehicle",
                "pedestrian","motorcycle","bicycle","traffic_cone","barrier","background"
            ]
        
        unknown_ids = set([fine_order.index(n) for n in cfg.OPENSET.UNKNOWN_CLASSES])

        # Validation period threshold strategy: prioritize fixed τ, otherwise use quantile EMA
        use_fixed_tau = hasattr(cfg.OPENSET, 'VAL_TAU') and (cfg.OPENSET.VAL_TAU is not None)
        fixed_tau = float(cfg.OPENSET.VAL_TAU) if use_fixed_tau else None
        q_val = getattr(cfg.OPENSET, 'VAL_Q', getattr(getattr(cfg.OPENSET, 'THRESHOLD', None), 'q', 0.80))

        def robust_norm01(x, eps=1e-6):
            # P10-P90 robust scaling, anti-outliers
            p10 = torch.quantile(x, 0.10)
            p90 = torch.quantile(x, 0.90)
            x = (x - p10) / (p90 - p10 + eps)
            return torch.clamp(x, 0.0, 1.0)

        def hmean(a, b, eps=1e-8):
            return (2 * a * b) / (a + b + eps)


    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None

            # ===== Open-set: batch unknownness U and threshold =====
            if openset_enabled and model_3d is not None:  # need 2D+3D
                with torch.no_grad():
                    def entropy(P, eps=1e-8):  # P (N,C)
                        return -(P * (P.add(eps).log())).sum(1)

                    def kl(P, Q, eps=1e-8):
                        return (P * (P.add(eps).log() - Q.add(eps).log())).sum(1)

                    H2D = entropy(probs_2d)
                    H3D = entropy(probs_3d)
                    M   = 0.5 * (probs_2d + probs_3d)
                    JS  = 0.5 * kl(probs_2d, M) + 0.5 * kl(probs_3d, M)

                    a = cfg.OPENSET.UNK_SCORE.alpha
                    b = cfg.OPENSET.UNK_SCORE.beta
                    c = cfg.OPENSET.UNK_SCORE.gamma
                    U_all = a * robust_norm01(H2D) + b * robust_norm01(H3D) + c * robust_norm01(JS)  # (N,)

                    if use_fixed_tau:
                        tau = torch.tensor(fixed_tau, device=U_all.device, dtype=U_all.dtype)
                    else:
                        tau_b = torch.quantile(U_all.detach(), q_val)
                        tau_ema = tau_b if (tau_ema is None) else (0.9 * tau_ema + 0.1 * tau_b)
                        tau = tau_ema

                    pred_unk_mask_all = (U_all > tau)  # (N,)


            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert torch.all(curr_points_idx).item()

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None
                
                # ===== Open-set: single sample slice and accumulate 6 classes IoU =====
                if openset_enabled and model_3d is not None:
                    # current sample range
                    mask_slice = pred_unk_mask_all[left_idx:right_idx].cpu().numpy()

                    # GT 6 classes: use fine labels to identify unknown; the rest use the mapped 5 classes
                    # Note: current validate has 'orig_seg_label' (5 classes) and 'orig_points_idx'.
                    # We use 'orig_seg_label_fine' (11 classes) to construct unknown GT.
                    
                    # check data field availability (only first time)
                    if batch_ind == 0 and iteration == 0:
                        logger.info(f"[OpenSet] Validation enabled - orig_seg_label_fine: {'orig_seg_label_fine' in data_batch}")
                    
                    if 'orig_seg_label_fine' in data_batch:
                        fine_full = data_batch['orig_seg_label_fine'][batch_ind].cpu().numpy()
                        pts_mask  = np.asarray(points_idx[batch_ind].cpu().numpy(), dtype=bool)
                        fine_in   = fine_full[pts_mask]  # align with pred
                        y6 = curr_seg_label.cpu().numpy().copy()
                        is_unk_gt = np.isin(fine_in, list(unknown_ids))
                        y6[is_unk_gt] = 5  # Unknown class id = 5
                    else:
                        # skip open-set accumulation if fine labels are missing
                        y6 = None

                    if y6 is not None:
                        # three-path prediction: first 5 classes argmax, then cover Unknown with U
                        yhat6_2d = pred_label_2d.copy()
                        yhat6_2d[mask_slice] = 5

                        if pred_label_3d is not None:
                            yhat6_3d = pred_label_3d.copy()
                            yhat6_3d[mask_slice] = 5
                        else:
                            yhat6_3d = None

                        if pred_label_ensemble is not None:
                            yhat6_f = pred_label_ensemble.copy()
                            yhat6_f[mask_slice] = 5
                        else:
                            yhat6_f = None

                        # accumulate 6 classes intersection and union
                        for m, yhat6 in [('2d', yhat6_2d), ('3d', yhat6_3d), ('fused', yhat6_f)]:
                            if yhat6 is None: 
                                continue
                            for k in range(6):
                                pred_k = (yhat6 == k)
                                gt_k   = (y6    == k)
                                inter  = int((pred_k & gt_k).sum())
                                union  = int((pred_k | gt_k).sum())
                                open_acc[m]['inter'][k] += inter
                                open_acc[m]['union'][k] += union

                        # A module effectiveness monitoring: unknown ratio
                        unk_ratio_sum += float(mask_slice.mean()) * len(mask_slice)
                        n_points_sum  += int(len(mask_slice))

                    # after build y6 / yhat6_2d / yhat6_3d / yhat6_f
                    if evaluator_os_2d is not None and y6 is not None:
                        evaluator_os_2d.update(yhat6_2d, y6)
                        if evaluator_os_3d is not None and yhat6_3d is not None:
                            evaluator_os_3d.update(yhat6_3d, y6)
                        if evaluator_os_ens is not None and yhat6_f is not None:
                            evaluator_os_ens.update(yhat6_f, y6)


                # evaluate
                evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    assert np.all(pred_label_2d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx]
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy(),
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label'])
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
        eval_list = [('2D', evaluator_2d)]
        if model_3d:
            eval_list.extend([('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)])
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))
        
        # ===== Open-set: summarize Common/Private/H-Score and monitoring metrics =====
        if openset_enabled and (n_points_sum > 0):
            def iou_from_inter_union(inter, union, eps=1e-8):
                inter = inter.to(torch.float32)
                union = union.to(torch.float32)
                return inter / (union + eps)

            def summarize(mod):
                iou6 = iou_from_inter_union(open_acc[mod]['inter'], open_acc[mod]['union'])  # (6,)
                common  = float(iou6[:5].mean().item())  # known 5 classes mIoU
                private = float(iou6[5].item())          # Unknown class IoU
                hscore  = float(hmean(common, private))
                return common, private, hscore

            res = {}
            for mod in ['2d', '3d', 'fused']:
                c, p, h = summarize(mod)
                val_metric_logger.update(**{
                    f'open_common_iou_{mod}': c,
                    f'open_private_iou_{mod}': p,
                    f'open_hscore_{mod}': h,
                })
                res[mod] = (c, p, h)

            unk_ratio_avg = float(unk_ratio_sum / max(1, n_points_sum))
            val_metric_logger.update(open_unk_ratio=unk_ratio_avg)
            if tau_ema is not None:
                val_metric_logger.update(open_tau=float(tau_ema))
            # --- Open-set tables (6 classes with Unknown) ---
            if evaluator_os_2d is not None:
                logger.info('Open-Set 2D overall IOU={:.2f}'.format(100.0 * evaluator_os_2d.overall_iou))
                logger.info('Open-Set 2D class-wise segmentation accuracy and IoU.\n{}'.format(
                    evaluator_os_2d.print_table()))

            if evaluator_os_3d is not None:
                logger.info('Open-Set 3D overall IOU={:.2f}'.format(100.0 * evaluator_os_3d.overall_iou))
                logger.info('Open-Set 3D class-wise segmentation accuracy and IoU.\n{}'.format(
                    evaluator_os_3d.print_table()))

            if evaluator_os_ens is not None:
                logger.info('Open-Set 2D+3D overall IOU={:.2f}'.format(100.0 * evaluator_os_ens.overall_iou))
                logger.info('Open-Set 2D+3D class-wise segmentation accuracy and IoU.\n{}'.format(
                    evaluator_os_ens.print_table()))

            # friendly print
            logger.info('[OpenSet-2D]   Common mIoU={:.2f}  Private IoU={:.2f}  H-Score={:.2f}'
                        .format(res['2d'][0]*100, res['2d'][1]*100, res['2d'][2]*100))
            if model_3d:
                logger.info('[OpenSet-3D]   Common mIoU={:.2f}  Private IoU={:.2f}  H-Score={:.2f}'
                            .format(res['3d'][0]*100, res['3d'][1]*100, res['3d'][2]*100))
                logger.info('[OpenSet-2D+3D] Common mIoU={:.2f}  Private IoU={:.2f}  H-Score={:.2f}'
                            .format(res['fused'][0]*100, res['fused'][1]*100, res['fused'][2]*100))
            logger.info('[OpenSet] unk_ratio={:.2f}%  tau={}'.format(
                unk_ratio_avg*100.0,
                ('fixed@{:.4f}'.format(fixed_tau) if use_fixed_tau else '{:.4f}'.format(float(tau_ema)))
            ))
