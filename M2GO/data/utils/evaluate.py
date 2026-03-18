import numpy as np
from sklearn.metrics import confusion_matrix as CM

class Evaluator(object):
    def __init__(self, class_names, labels=None):
        self.class_names = tuple(class_names)
        self.num_classes = len(class_names)
        self.labels = np.arange(self.num_classes) if labels is None else np.array(labels)
        assert self.labels.shape[0] == self.num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_label, gt_label):
        """Update per instance

        Args:
            pred_label (np.ndarray): (num_points)
            gt_label (np.ndarray): (num_points,)
        """
        
        gt = np.array(gt_label, copy=True)
        pr = np.array(pred_label, copy=False)

        # convert ignore_label to num_classes
        gt[gt == -100] = self.num_classes
        confusion_matrix = CM(gt.flatten(), pr.flatten(), labels=self.labels)
        self.confusion_matrix += confusion_matrix

    def batch_update(self, pred_labels, gt_labels):
        assert len(pred_labels) == len(gt_labels)
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            self.update(pred_label, gt_label)

    @property
    def overall_acc(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    @property
    def overall_iou(self):
        class_iou = np.array(self.class_iou.copy())
        class_iou[np.isnan(class_iou)] = 0
        return np.mean(class_iou)

    @property
    def class_seg_acc(self):
        return [self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i])
                for i in range(self.num_classes)]

    @property
    def class_iou(self):
        iou_list = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            p = self.confusion_matrix[:, i].sum()
            g = self.confusion_matrix[i, :].sum()
            union = p + g - tp
            if union == 0:
                iou = float('nan')
            else:
                iou = tp / union
            iou_list.append(iou)
        return iou_list

    # ===================== add: openset evaluation ======================
    def _unknown_index(self, unknown_name='unknown', unknown_id=None):
        """return the index of unknown class (default find the class named 'unknown'). return None if not found."""
        if unknown_id is not None:
            return unknown_id if 0 <= unknown_id < self.num_classes else None
        try:
            return self.class_names.index(unknown_name)
        except ValueError:
            return None

    def openset_common_iou(self, unknown_name='unknown', unknown_id=None):
        """known (common) mIoU: exclude unknown and then do nanmean."""
        unk = self._unknown_index(unknown_name, unknown_id)
        iou = np.array(self.class_iou, dtype=float)
        if unk is not None:
            known_mask = np.ones(self.num_classes, dtype=bool)
            known_mask[unk] = False
            vals = iou[known_mask]
        else:
            vals = iou  # if no unknown, then fall back to closed set mean
        return float(np.nanmean(vals)) if vals.size > 0 else float('nan')

    def openset_private_iou(self, unknown_name='unknown', unknown_id=None):
        """unknown (private) IoU: if NaN, then treat as 0 (more intuitive for H-score)."""
        unk = self._unknown_index(unknown_name, unknown_id)
        if unk is None:
            return float('nan')
        val = float(self.class_iou[unk])
        return 0.0 if np.isnan(val) else val

    def openset_hscore(self, unknown_name='unknown', unknown_id=None, eps=1e-8):
        """H-Score = 2 * (common_mIoU * private_IoU) / (common_mIoU + private_IoU + eps)"""
        c = self.openset_common_iou(unknown_name, unknown_id)
        p = self.openset_private_iou(unknown_name, unknown_id)
        if np.isnan(c) or np.isnan(p):
            return float('nan')
        return float(2.0 * c * p / (c + p + eps))

    def print_table(self):
        from tabulate import tabulate
        header = ['Class', 'Accuracy', 'IOU', 'Total']
        seg_acc_per_class = self.class_seg_acc
        iou_per_class = self.class_iou
        table = []
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name,
                          seg_acc_per_class[ind] * 100,
                          iou_per_class[ind] * 100 if not np.isnan(iou_per_class[ind]) else float('nan'),
                          int(self.confusion_matrix[ind].sum()),
                          ])
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate
        header = ('overall acc', 'overall iou') + self.class_names
        table = [[self.overall_acc, self.overall_iou] + self.class_iou]
        with open(filename, 'w') as f:
            f.write(tabulate(table, headers=header, tablefmt='tsv', floatfmt='.5f',
                             numalign=None, stralign=None))
