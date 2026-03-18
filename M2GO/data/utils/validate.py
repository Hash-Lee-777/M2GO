import numpy as np
import logging
import time
import os
import os.path as osp

import torch
import torch.nn.functional as F

from M2GO.data.utils.evaluate import Evaluator
from M2GO.data.utils.visualize import draw_points_image_labels, draw_points_image_depth

# Optional sklearn metrics for AUROC/AUPR (fallback to numpy implementation if unavailable)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _SKLEARN_METRICS = True
except Exception:
    roc_auc_score = None
    average_precision_score = None
    _SKLEARN_METRICS = False


def _infer_viz_palette_from_cfg(cfg):
    palette = getattr(cfg.VAL, 'VIZ_PALETTE', None)
    if palette:
        return palette
    dataset_type = getattr(cfg.DATASET_TARGET, 'TYPE', '')
    dataset_type = dataset_type.lower()
    if 'kitti' in dataset_type:
        return 'SemanticKITTI'
    if 'a2d2' in dataset_type:
        return 'A2D2'
    if 'nuscenes' in dataset_type:
        return 'NuScenes'
    return 'NuScenes'


def validate(cfg,
             model_2d,
             model_3d,
             dataloader,
             val_metric_logger,
             pselab_path=None):
    logger = logging.getLogger('M2GO.validate')
    logger.info('Validation')

    viz_palette_type = _infer_viz_palette_from_cfg(cfg)
    viz_point_size = getattr(cfg.VAL, 'VIZ_POINT_SIZE', 1.0)

    # evaluator
    class_names = dataloader.dataset.class_names
    num_known_classes = len(class_names)
    unknown_class_id = num_known_classes
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None

    # --- Open-set evaluators: known classes + 1 unknown ---
    if cfg.OPENSET.ENABLE:
        class_names_os = list(class_names) + ['unknown']
        evaluator_os_2d = Evaluator(class_names_os)
        evaluator_os_3d = Evaluator(class_names_os) if model_3d else None
        evaluator_os_ens = Evaluator(class_names_os) if model_3d else None
    else:
        evaluator_os_2d = evaluator_os_3d = evaluator_os_ens = None

    # ===== Open-set: init =====
    openset_enabled = hasattr(cfg, 'OPENSET') and getattr(cfg.OPENSET, 'ENABLE', False)
    num_open_classes = num_known_classes + 1 if openset_enabled else num_known_classes
    open_acc = None
    if openset_enabled:
        open_acc = {}
        for m in ['2d', '3d', 'fused']:
            open_acc[m] = {
                'inter': torch.zeros(num_open_classes, dtype=torch.long, device='cpu'),
                'union': torch.zeros(num_open_classes, dtype=torch.long, device='cpu'),
            }
    unk_ratio_sum, n_points_sum = 0.0, 0
    tau_ema = None  # kept for backward-compat logs; not used for head-based prediction

    if openset_enabled:
        if hasattr(dataloader.dataset, 'fine_class_names'):
            fine_order = list(dataloader.dataset.fine_class_names)
        else:
            fine_order = [
                "car","truck","bus","trailer","construction_vehicle",
                "pedestrian","motorcycle","bicycle","traffic_cone","barrier","background"
            ]
        
        unknown_ids = set([fine_order.index(n) for n in cfg.OPENSET.UNKNOWN_CLASSES])

        def hmean(a, b, eps=1e-8):
            return (2 * a * b) / (a + b + eps)

        # Head-expansion unknown threshold (fixed)
        head_unknown_thr_2d = getattr(cfg.VAL, 'HEAD_UNKNOWN_THR_2D', 0.2)
        head_unknown_thr_3d = getattr(cfg.VAL, 'HEAD_UNKNOWN_THR_3D', 0.05)

        # Buffers for threshold-free detection metrics (AUROC/AUPR)
        det_y_true_2d, det_y_score_2d = [], []
        det_y_true_3d, det_y_score_3d = [], []
        det_y_true_f , det_y_score_f  = [], []


    pselab_data_list = []
    # diagnostics for head-expansion unknown prediction
    diag_tot, diag_pu2d_over, diag_pu3d_over, diag_puf_over = 0, 0, 0, 0
    diag_arg2d_unk, diag_arg3d_unk, diag_argf_unk = 0, 0, 0

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

            logits_2d = preds_2d['seg_logit']
            logits_3d = preds_3d['seg_logit'] if model_3d else None

            # Argmax over all logits (includes Unknown head when available)
            pred_label_voxel_2d = logits_2d.argmax(1).cpu().numpy()
            pred_label_voxel_3d = logits_3d.argmax(1).cpu().numpy() if model_3d else None

            # Closed-set argmax for standard evaluator: restrict to known classes
            pred_label_voxel_2d_known = logits_2d[:, :num_known_classes].argmax(1).cpu().numpy()
            pred_label_voxel_3d_known = logits_3d[:, :num_known_classes].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(logits_2d, dim=1)
            probs_3d = F.softmax(logits_3d, dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None
            # Closed-set (known classes only) for ensemble as well
            pred_label_voxel_ensemble_known = (probs_2d[:, :num_known_classes] + probs_3d[:, :num_known_classes]).argmax(1).cpu().numpy() if model_3d else None

            if openset_enabled and probs_2d.shape[1] <= unknown_class_id:
                raise RuntimeError('Open-set validation requires the 2D head to output an Unknown logit (C+1 classes).')
            if openset_enabled and model_3d and probs_3d.shape[1] <= unknown_class_id:
                raise RuntimeError('Open-set validation requires the 3D head to output an Unknown logit (C+1 classes).')

            # Head-expansion evaluation: use C+1 head predictions for Unknown
            # We keep probabilities for ensemble, but Unknown decision comes from argmax of 6-way heads

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
                yhat_os_2d = None
                yhat_os_3d = None
                yhat_os_f = None
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None

                # Known-class predictions for closed-set evaluation
                pred_label_2d_known = pred_label_voxel_2d_known[left_idx:right_idx]
                pred_label_3d_known = pred_label_voxel_3d_known[left_idx:right_idx] if model_3d else None
                pred_label_ensemble_known = pred_label_voxel_ensemble_known[left_idx:right_idx] if model_3d else None
                
            # ===== Open-set: single sample slice and accumulate IoU =====
                if openset_enabled:
                    # GT (known + unknown): use fine labels to identify unknown; the rest use mapped known classes
                    # Note: current validate has 'orig_seg_label' (known classes) and 'orig_points_idx'.
                    # We use 'orig_seg_label_fine' to construct unknown GT.

                    # check data field availability (only first time)
                    if batch_ind == 0 and iteration == 0:
                        logger.info(f"[OpenSet] Validation enabled - orig_seg_label_fine: {'orig_seg_label_fine' in data_batch}")

                    if 'orig_seg_label_fine' in data_batch:
                        fine_full = data_batch['orig_seg_label_fine'][batch_ind].cpu().numpy()
                        pts_mask  = np.asarray(points_idx[batch_ind].cpu().numpy(), dtype=bool)
                        fine_in   = fine_full[pts_mask]  # align with pred
                        y_os = curr_seg_label.cpu().numpy().copy()
                        is_unk_gt = np.isin(fine_in, list(unknown_ids))
                        y_os[is_unk_gt] = unknown_class_id  # Unknown class id = len(known)
                    else:
                        y_os = None

                    if y_os is not None:
                        # Head-expansion prediction: Unknown if p_unknown > thr; else choose best among known classes
            
                        # 2D
                        p_unk_2d = probs_2d[left_idx:right_idx, unknown_class_id].detach().cpu().numpy()
                        base_known_2d = pred_label_voxel_2d_known[left_idx:right_idx].copy()
                        yhat_os_2d = base_known_2d
                        yhat_os_2d[p_unk_2d > head_unknown_thr_2d] = unknown_class_id

                        # 3D (if available)
                        if model_3d:
                            p_unk_3d = probs_3d[left_idx:right_idx, unknown_class_id].detach().cpu().numpy()
                            base_known_3d = pred_label_voxel_3d_known[left_idx:right_idx].copy()
                            yhat_os_3d = base_known_3d
                            yhat_os_3d[p_unk_3d > head_unknown_thr_3d] = unknown_class_id
                        else:
                            yhat_os_3d = None

                        # Fused (if available)
                        if model_3d:
                            pro_f = (probs_2d[left_idx:right_idx] + probs_3d[left_idx:right_idx]) / 2.0
                            p_unk_f = pro_f[:, unknown_class_id].detach().cpu().numpy()
                            base_known_f = pred_label_voxel_ensemble_known[left_idx:right_idx].copy()
                            yhat_os_f = base_known_f
                            yhat_os_f[p_unk_f > head_unknown_thr_3d] = unknown_class_id
                        else:
                            yhat_os_f = None

                        # accumulate detection labels/scores for AUROC/AUPR (positive = Unknown)
                        y_true_slice = (y_os == unknown_class_id)
                        det_y_true_2d.append(y_true_slice.astype('uint8'))
                        det_y_score_2d.append(p_unk_2d.astype('float32'))
                        if model_3d:
                            det_y_true_3d.append(y_true_slice.astype('uint8'))
                            det_y_score_3d.append(p_unk_3d.astype('float32'))
                            det_y_true_f.append(y_true_slice.astype('uint8'))
                            det_y_score_f.append(p_unk_f.astype('float32'))

                        # accumulate (known + unknown) classes intersection and union
                        for m, yhat_os in [('2d', yhat_os_2d), ('3d', yhat_os_3d), ('fused', yhat_os_f)]:
                            if yhat_os is None: 
                                continue
                            for k in range(num_open_classes):
                                pred_k = (yhat_os == k)
                                gt_k   = (y_os    == k)
                                inter  = int((pred_k & gt_k).sum())
                                union  = int((pred_k | gt_k).sum())
                                open_acc[m]['inter'][k] += inter
                                open_acc[m]['union'][k] += union

                        # Monitoring: predicted unknown ratio from fused (fallback to 2d)
                        pred_ref = yhat_os_f if yhat_os_f is not None else yhat_os_2d
                        unk_ratio_sum += float((pred_ref == unknown_class_id).mean()) * len(pred_ref)
                        n_points_sum  += int(len(pred_ref))

                        # diagnostics aggregation
                        diag_tot += int(len(pred_ref))
                        diag_pu2d_over += int((p_unk_2d > head_unknown_thr_2d).sum())
                        if model_3d:
                            diag_pu3d_over += int((p_unk_3d > head_unknown_thr_3d).sum())
                            diag_puf_over  += int((p_unk_f > head_unknown_thr_3d).sum())
                        diag_arg2d_unk += int((yhat_os_2d == unknown_class_id).sum())
                        if yhat_os_3d is not None:
                            diag_arg3d_unk += int((yhat_os_3d == unknown_class_id).sum())
                        if yhat_os_f is not None:
                            diag_argf_unk  += int((yhat_os_f == unknown_class_id).sum())

                    # after build y_os / yhat_os_2d / yhat_os_3d / yhat_os_f
                    if evaluator_os_2d is not None and y_os is not None:
                        evaluator_os_2d.update(yhat_os_2d, y_os)
                        if evaluator_os_3d is not None and yhat_os_3d is not None:
                            evaluator_os_3d.update(yhat_os_3d, y_os)
                        if evaluator_os_ens is not None and yhat_os_f is not None:
                            evaluator_os_ens.update(yhat_os_f, y_os)


                # evaluate
                evaluator_2d.update(pred_label_2d_known, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d_known, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble_known, curr_seg_label)

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

                # keep current sample slice for optional visualization
                slice_l, slice_r = left_idx, right_idx
                left_idx = right_idx

                # ---- minimal visualization (optional) ----
                if getattr(cfg.VAL, 'SAVE_VIZ', False) and (iteration % int(cfg.VAL.VIZ_EVERY) == 0):
                    # prepare image and indices for current slice
                    img = data_batch['img'][batch_ind].detach().cpu().numpy()
                    img = np.moveaxis(img, 0, 2)
                    # img_indices is numpy; build boolean mask of points that entered network
                    pts_mask_np = np.asarray(points_idx[batch_ind].cpu().numpy(), dtype=bool)
                    img_idx = data_batch['img_indices'][batch_ind][pts_mask_np]
                    # 2D known-class overlay
                    import matplotlib.pyplot as plt
                    viz_dir = osp.join(osp.dirname(logger.handlers[0].baseFilename) if logger.handlers else '.', 'viz')
                    os.makedirs(viz_dir, exist_ok=True)
                    plt.figure()
                    plt.imshow(img)
                    scatter = plt.scatter(
                        img_idx[:, 1],
                        img_idx[:, 0],
                        c=pred_label_2d_known,
                        cmap=None,
                        s=1.5,
                        alpha=0.9
                    )
                    plt.axis('off')
                    scatter.remove()
                    overlay_palette = 'NuScenes'
                    overlay_labels = pred_label_2d_known.copy()
                    overlay_point_size = 1.0

                    if openset_enabled:
                        fused_labels = None
                        if yhat_os_f is not None:
                            fused_labels = yhat_os_f
                        elif yhat_os_2d is not None:
                            fused_labels = yhat_os_2d
                        if fused_labels is not None:
                            overlay_palette = viz_palette_type
                            overlay_point_size = viz_point_size
                            overlay_labels = np.asarray(fused_labels, dtype=np.int64)

                    draw_points_image_labels(
                        img.copy(),
                        img_idx,
                        overlay_labels,
                        show=False,
                        color_palette_type=overlay_palette,
                        point_size=overlay_point_size,
                        save_path=osp.join(viz_dir, f'iter{iteration:06d}_b{batch_ind}_2d.png')
                    )
                    plt.close()

                    if openset_enabled:
                        fused_labels = None
                        if yhat_os_f is not None:
                            fused_labels = yhat_os_f
                        elif yhat_os_2d is not None:
                            fused_labels = yhat_os_2d

                        if fused_labels is not None:
                            draw_points_image_labels(
                                img.copy(),
                                img_idx,
                                np.asarray(fused_labels, dtype=np.int64),
                                show=False,
                                color_palette_type=viz_palette_type,
                                point_size=viz_point_size,
                                save_path=osp.join(viz_dir, f'iter{iteration:06d}_b{batch_ind}_osfused.png')
                            )

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
                iou_vec = iou_from_inter_union(open_acc[mod]['inter'], open_acc[mod]['union'])  # (known+1,)
                common  = float(iou_vec[:num_known_classes].mean().item())  # known classes mIoU
                private = float(iou_vec[unknown_class_id].item())           # Unknown class IoU
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
            # tau_ema is not used for head-based prediction; keep omitted
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
            logger.info('[OpenSet] unk_ratio={:.2f}%'.format(unk_ratio_avg*100.0))
            if diag_tot > 0:
                logger.info('[Diag] p_unk>thr ratio - 2D:{:.2f}% 3D:{:.2f}% Fused:{:.2f}% | argmax Unknown ratio - 2D:{:.2f}% 3D:{:.2f}% Fused:{:.2f}%'.format(
                    100.0*diag_pu2d_over/diag_tot,
                    100.0*(diag_pu3d_over/diag_tot if model_3d else 0.0),
                    100.0*(diag_puf_over/diag_tot if model_3d else 0.0),
                    100.0*diag_arg2d_unk/diag_tot,
                    100.0*(diag_arg3d_unk/diag_tot if model_3d else 0.0),
                    100.0*(diag_argf_unk/diag_tot if model_3d else 0.0)))

            # Threshold-free detection metrics (AUROC/AUPR)
            def _auc_trapezoid(x, y):
                return float(np.trapz(y, x)) if len(x) > 1 else 0.0

            def _roc_auc(y_true, y_score):
                y_true = np.asarray(y_true, dtype=np.uint8)
                y_score = np.asarray(y_score, dtype=np.float32)
                if y_true.size == 0:
                    return 0.0
                order = np.argsort(-y_score)
                y_true = y_true[order]
                tp = np.cumsum(y_true)
                fp = np.cumsum(1 - y_true)
                P = float(tp[-1]) if tp.size else 0.0
                N = float(fp[-1]) if fp.size else 0.0
                if P == 0.0 or N == 0.0:
                    return 0.0
                tpr = tp / P
                fpr = fp / N
                fpr = np.concatenate([[0.0], fpr, [1.0]])
                tpr = np.concatenate([[0.0], tpr, [1.0]])
                return _auc_trapezoid(fpr, tpr)

            def _pr_auc(y_true, y_score):
                y_true = np.asarray(y_true, dtype=np.uint8)
                y_score = np.asarray(y_score, dtype=np.float32)
                if y_true.size == 0:
                    return 0.0
                order = np.argsort(-y_score)
                y_true = y_true[order]
                tp = np.cumsum(y_true)
                fp = np.cumsum(1 - y_true)
                P = float(tp[-1]) if tp.size else 0.0
                if P == 0.0:
                    return 0.0
                recall = tp / P
                precision = tp / np.maximum(tp + fp, 1e-8)
                recall = np.concatenate([[0.0], recall])
                precision = np.concatenate([[1.0], precision])
                return _auc_trapezoid(recall, precision)

            def _compute_and_log(tag, ys, ss):
                if len(ys) == 0 or len(ss) == 0:
                    return
                y = np.concatenate(ys)
                s = np.concatenate(ss)
                # Prefer sklearn (robust & battle-tested); fallback to numpy implementation
                if _SKLEARN_METRICS:
                    try:
                        auroc = float(roc_auc_score(y, s))
                    except Exception:
                        auroc = _roc_auc(y, s)
                    try:
                        aupr = float(average_precision_score(y, s))
                    except Exception:
                        aupr = _pr_auc(y, s)
                else:
                    auroc = _roc_auc(y, s)
                    aupr  = _pr_auc(y, s)
                logger.info('[OpenSet-Det-{}] AUROC={:.2f}  AUPR={:.2f}'.format(tag, 100.0*auroc, 100.0*aupr))

            _compute_and_log('2D', det_y_true_2d, det_y_score_2d)
            if model_3d:
                _compute_and_log('3D', det_y_true_3d, det_y_score_3d)
                _compute_and_log('2D+3D', det_y_true_f, det_y_score_f)
