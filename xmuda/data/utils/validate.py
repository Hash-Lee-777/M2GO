import numpy as np
import logging
import time
import os
import os.path as osp

import torch
import torch.nn.functional as F

from xmuda.data.utils.evaluate import Evaluator
from xmuda.data.utils.visualize import draw_points_image_labels, draw_points_image_depth


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
        head_unknown_thr = 0.2


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

            # 6-class argmax (includes Unknown)
            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # Closed-set (5-class) argmax for standard evaluator: ignore Unknown logit
            pred_label_voxel_2d_5 = preds_2d['seg_logit'][:, :5].argmax(1).cpu().numpy()
            pred_label_voxel_3d_5 = preds_3d['seg_logit'][:, :5].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None
            # Closed-set (5-class) for ensemble as well
            pred_label_voxel_ensemble_5 = (probs_2d[:, :5] + probs_3d[:, :5]).argmax(1).cpu().numpy() if model_3d else None

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
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None

                # 5-class predictions for closed-set evaluation
                pred_label_2d_5 = pred_label_voxel_2d_5[left_idx:right_idx]
                pred_label_3d_5 = pred_label_voxel_3d_5[left_idx:right_idx] if model_3d else None
                pred_label_ensemble_5 = pred_label_voxel_ensemble_5[left_idx:right_idx] if model_3d else None
                
            # ===== Open-set: single sample slice and accumulate 6 classes IoU =====
                if openset_enabled:
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
                        y6 = None

                    if y6 is not None:
                        # Head-expansion prediction: Unknown if p_unknown > thr; else choose best among first 5
                        # 2D
                        p_unk_2d = probs_2d[left_idx:right_idx, -1].detach().cpu().numpy()
                        base5_2d = pred_label_voxel_2d_5[left_idx:right_idx].copy()
                        yhat6_2d = base5_2d
                        yhat6_2d[p_unk_2d > head_unknown_thr] = 5

                        # 3D (if available)
                        if model_3d:
                            p_unk_3d = probs_3d[left_idx:right_idx, -1].detach().cpu().numpy()
                            base5_3d = pred_label_voxel_3d_5[left_idx:right_idx].copy()
                            yhat6_3d = base5_3d
                            yhat6_3d[p_unk_3d > head_unknown_thr] = 5
                        else:
                            yhat6_3d = None

                        # Fused (if available)
                        if model_3d:
                            pro_f = (probs_2d[left_idx:right_idx] + probs_3d[left_idx:right_idx]) / 2.0
                            p_unk_f = pro_f[:, -1].detach().cpu().numpy()
                            base5_f = pred_label_voxel_ensemble_5[left_idx:right_idx].copy()
                            yhat6_f = base5_f
                            yhat6_f[p_unk_f > head_unknown_thr] = 5
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

                        # Monitoring: predicted unknown ratio from fused (fallback to 2d)
                        pred_ref = yhat6_f if yhat6_f is not None else yhat6_2d
                        unk_ratio_sum += float((pred_ref == 5).mean()) * len(pred_ref)
                        n_points_sum  += int(len(pred_ref))

                        # diagnostics aggregation
                        diag_tot += int(len(pred_ref))
                        diag_pu2d_over += int((p_unk_2d > head_unknown_thr).sum())
                        if model_3d:
                            diag_pu3d_over += int((p_unk_3d > head_unknown_thr).sum())
                            diag_puf_over  += int((p_unk_f > head_unknown_thr).sum())
                        diag_arg2d_unk += int((yhat6_2d == 5).sum())
                        if yhat6_3d is not None:
                            diag_arg3d_unk += int((yhat6_3d == 5).sum())
                        if yhat6_f is not None:
                            diag_argf_unk  += int((yhat6_f == 5).sum())

                    # after build y6 / yhat6_2d / yhat6_3d / yhat6_f
                    if evaluator_os_2d is not None and y6 is not None:
                        evaluator_os_2d.update(yhat6_2d, y6)
                        if evaluator_os_3d is not None and yhat6_3d is not None:
                            evaluator_os_3d.update(yhat6_3d, y6)
                        if evaluator_os_ens is not None and yhat6_f is not None:
                            evaluator_os_ens.update(yhat6_f, y6)


                # evaluate
                evaluator_2d.update(pred_label_2d_5, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d_5, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble_5, curr_seg_label)

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
                    # 2D 5-class overlay
                    draw_points_image_labels(img.copy(), img_idx, pred_label_2d_5.copy(), show=False, color_palette_type='NuScenes', point_size=1.0)
                    import matplotlib.pyplot as plt
                    viz_dir = osp.join(osp.dirname(logger.handlers[0].baseFilename) if logger.handlers else '.', 'viz')
                    os.makedirs(viz_dir, exist_ok=True)
                    plt.savefig(osp.join(viz_dir, f'iter{iteration:06d}_b{batch_ind}_2d.png'), dpi=200); plt.close()
                    # Unknown heat (fused p_unknown if available, else 2D)
                    if model_3d:
                        pro_f = (probs_2d[slice_l:slice_r] + probs_3d[slice_l:slice_r]) / 2.0
                        p_unk_vis = pro_f[:, -1].detach().cpu().numpy()
                    else:
                        p_unk_vis = probs_2d[slice_l:slice_r, -1].detach().cpu().numpy()
                    draw_points_image_depth(img.copy(), img_idx, p_unk_vis, show=False, point_size=1.0)
                    plt.savefig(osp.join(viz_dir, f'iter{iteration:06d}_b{batch_ind}_punk.png'), dpi=200); plt.close()

                    # Mismatch vs GT-Unknown overlap visualization
                    try:
                        import matplotlib.pyplot as plt
                        # closed-set argmax (5-class) for this slice
                        pred2d5_v = pred_label_voxel_2d_5[slice_l:slice_r]
                        pred3d5_v = pred_label_voxel_3d_5[slice_l:slice_r] if model_3d else None
                        if model_3d and pred3d5_v is not None:
                            mismatch = (pred2d5_v != pred3d5_v)
                        else:
                            mismatch = np.zeros_like(pred2d5_v, dtype=bool)

                        # GT unknown from fine labels
                        if 'orig_seg_label_fine' in data_batch:
                            fine_full = data_batch['orig_seg_label_fine'][batch_ind].cpu().numpy()
                            gt_unk = np.isin(fine_full[pts_mask_np], list(unknown_ids))
                        else:
                            gt_unk = np.zeros_like(pred2d5_v, dtype=bool)

                        overlap = mismatch & gt_unk
                        mismatch_only = mismatch & (~gt_unk)
                        gt_unk_only = gt_unk & (~mismatch)

                        plt.imshow(img)
                        # red: mismatch only
                        pts = img_idx[mismatch_only]
                        if len(pts) > 0:
                            plt.scatter(pts[:,1], pts[:,0], c='#FF3B30', s=3, alpha=0.9, linewidths=0)
                        # green: gt unknown only
                        pts = img_idx[gt_unk_only]
                        if len(pts) > 0:
                            plt.scatter(pts[:,1], pts[:,0], c='#34C759', s=3, alpha=0.9, linewidths=0)
                        # yellow: overlap
                        pts = img_idx[overlap]
                        if len(pts) > 0:
                            plt.scatter(pts[:,1], pts[:,0], c='#FFCC00', s=4, alpha=0.95, linewidths=0)
                        plt.axis('off')
                        plt.savefig(osp.join(viz_dir, f'iter{iteration:06d}_b{batch_ind}_mis_unk.png'), dpi=200); plt.close()
                    except Exception as e:
                        logger.info(f"[VIZ] mismatch/gt_unk viz skipped: {e}")

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
