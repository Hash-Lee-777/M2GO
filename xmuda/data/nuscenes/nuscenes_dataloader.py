import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

from xmuda.data.utils.refine_pseudo_labels import refine_pseudo_labels
from xmuda.data.utils.augmentation_3d import augment_and_scale_3d


class NuScenesBase(Dataset):
    """NuScenes dataset"""

    class_names = [
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
        "barrier",
        "background",
    ]

    # use those categories if merge_classes == True
    categories = {
        "vehicle": ["car", "truck", "bus", "trailer", "construction_vehicle"],
        "pedestrian": ["pedestrian"],
        "bike": ["motorcycle", "bicycle"],
        "traffic_boundary": ["traffic_cone", "barrier"],
        "background": ["background"]
    }

    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False,
                 pselab_paths=None,
                 output_orig=False,    # add output_orig parameter
                 # === new open-set ===
                 openset=False,
                 domain=None,          # 'source' or 'target'
                 unknown_classes=None
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir
        self.output_orig = output_orig  # set output_orig attribute

        print(f"Initialize Nuscenes dataloader - split: {split}, output_orig: {output_orig}, openset: {openset}, domain: {domain}")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        self.pselab_data = None
        if pselab_paths:
            assert isinstance(pselab_paths, tuple)
            print('Load pseudo label data ', pselab_paths)
            self.pselab_data = []
            for curr_split in pselab_paths:
                self.pselab_data.extend(np.load(curr_split, allow_pickle=True))

            # check consistency of data and pseudo labels
            assert len(self.pselab_data) == len(self.data)
            for i in range(len(self.pselab_data)):
                assert len(self.pselab_data[i]['pseudo_label_2d']) == len(self.data[i]['seg_labels'])

            # refine 2d pseudo labels
            probs2d = np.concatenate([data['probs_2d'] for data in self.pselab_data])
            pseudo_label_2d = np.concatenate([data['pseudo_label_2d'] for data in self.pselab_data]).astype(np.int64)
            pseudo_label_2d = refine_pseudo_labels(probs2d, pseudo_label_2d)

            # refine 3d pseudo labels
            # fusion model has only one final prediction saved in probs_2d
            if 'probs_3d' in self.pselab_data[0].keys():
                probs3d = np.concatenate([data['probs_3d'] for data in self.pselab_data])
                pseudo_label_3d = np.concatenate([data['pseudo_label_3d'] for data in self.pselab_data]).astype(np.int64)
                pseudo_label_3d = refine_pseudo_labels(probs3d, pseudo_label_3d)
            else:
                pseudo_label_3d = None

            # undo concat
            left_idx = 0
            for data_idx in range(len(self.pselab_data)):
                right_idx = left_idx + len(self.pselab_data[data_idx]['probs_2d'])
                self.pselab_data[data_idx]['pseudo_label_2d'] = pseudo_label_2d[left_idx:right_idx]
                if pseudo_label_3d is not None:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = pseudo_label_3d[left_idx:right_idx]
                else:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = None
                left_idx = right_idx

        
        self.fine_class_names = [
            "car","truck","bus","trailer","construction_vehicle",
            "pedestrian","motorcycle","bicycle","traffic_cone","barrier","background"
        ]
        fine_names = self.fine_class_names  
        unknown_set = set(unknown_classes or [])

        if merge_classes:
            self.label_mapping = -100 * np.ones(len(fine_names), dtype=int)
            for cat_idx, (cat_name, cat_list) in enumerate(self.categories.items()):
                for cls in cat_list:
                    fid = fine_names.index(cls)
                    # strict open-set: only map unknown classes to ignore in source-train
                    is_source_train = (domain == 'source') and any('train' in s for s in self.split)
                    if openset and is_source_train and (cls in unknown_set):
                        # keep -100 = ignore_index (not merged into 5 classes)
                        continue
                    self.label_mapping[fid] = cat_idx
            self.class_names = list(self.categories.keys())  # 5 types of names
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class NuScenesSCN(NuScenesBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 nuscenes_dir='',
                 pselab_paths=None,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 use_image=False,
                 resize=(400, 225),
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_x=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 output_orig=False,
                 openset=False,
                 domain=None,                 # 'source' or 'target'
                 unknown_classes=None
                 ):
        super().__init__(split,
                         preprocess_dir,
                         merge_classes=merge_classes,
                         pselab_paths=pselab_paths,
                         output_orig=output_orig,  
                         openset=openset,
                         domain=domain,
                         unknown_classes=unknown_classes)

        self.nuscenes_dir = nuscenes_dir
        
        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
        self.rot_z = rot_z
        self.transl = transl

        # image parameters
        self.use_image = use_image
        if self.use_image:
            self.resize = resize
            self.image_normalizer = image_normalizer

            # data augmentation
            self.fliplr = fliplr
            self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

    def __getitem__(self, index):
        data_dict = self.data[index]

        points = data_dict['points'].copy()
        # keep one copy of 11 types of "fine" labels
        seg_label_fine = data_dict['seg_labels'].astype(np.int64)

        # then do the 11→5 mapping (or keep fine labels)
        seg_label = seg_label_fine
        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]

        out_dict = {}

        keep_idx = np.ones(len(points), dtype=np.bool_)
        if self.use_image:
            points_img = data_dict['points_img'].copy()
            img_path = osp.join(self.nuscenes_dir, data_dict['camera_path'])
            image = Image.open(img_path)

            if self.resize:
                if not image.size == self.resize:
                    # check if we do not enlarge downsized images
                    assert image.size[0] > self.resize[0]

                    # scale image points
                    points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
                    points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])

                    # resize image
                    image = image.resize(self.resize, Image.BILINEAR)

            img_indices = points_img.astype(np.int64)

            assert np.all(img_indices[:, 0] >= 0)
            assert np.all(img_indices[:, 1] >= 0)
            assert np.all(img_indices[:, 0] < image.size[1])
            assert np.all(img_indices[:, 1] < image.size[0])

            # 2D augmentation
            if self.color_jitter is not None:
                image = self.color_jitter(image)
            # PIL to numpy
            image = np.array(image, dtype=np.float32, copy=False) / 255.
            # 2D augmentation
            if np.random.rand() < self.fliplr:
                image = np.ascontiguousarray(np.fliplr(image))
                img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

            # normalize image
            if self.image_normalizer:
                mean, std = self.image_normalizer
                mean = np.asarray(mean, dtype=np.float32)
                std = np.asarray(std, dtype=np.float32)
                image = (image - mean) / std

            out_dict['img'] = np.moveaxis(image, -1, 0)
            out_dict['img_indices'] = img_indices

        # 3D data augmentation and scaling from points to voxel indices
        # nuscenes lidar coordinates: x (right), y (front), z (up)
        coords = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_x=self.flip_x, rot_z=self.rot_z, transl=self.transl)

        # cast to integer
        coords = coords.astype(np.int64)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        out_dict['feats'] = np.ones([len(idxs), 1], np.float32)  # simply use 1 as feature
        out_dict['seg_label'] = seg_label[idxs]

        if self.use_image:
            out_dict['img_indices'] = out_dict['img_indices'][idxs]

        if self.pselab_data is not None:
            out_dict.update({
                'pseudo_label_2d': self.pselab_data[index]['pseudo_label_2d'][keep_idx][idxs],
                'pseudo_label_3d': self.pselab_data[index]['pseudo_label_3d'][keep_idx][idxs]
            })
            # also export per-point confidence (max prob) for head-expansion training
            if 'probs_2d' in self.pselab_data[index]:
                out_dict['pseudo_conf_2d'] = self.pselab_data[index]['probs_2d'][keep_idx][idxs].astype(np.float32)
            else:
                out_dict['pseudo_conf_2d'] = None
            if 'probs_3d' in self.pselab_data[index] and self.pselab_data[index]['probs_3d'] is not None:
                out_dict['pseudo_conf_3d'] = self.pselab_data[index]['probs_3d'][keep_idx][idxs].astype(np.float32)
            else:
                out_dict['pseudo_conf_3d'] = None

        if self.output_orig:
            import torch
            out_dict.update({
                'orig_seg_label': torch.from_numpy(seg_label).long(),
                'orig_seg_label_fine': torch.from_numpy(seg_label_fine).long(),
                'orig_points_idx': torch.from_numpy(idxs.astype(np.bool_)),   # back to numpy
            })


        return out_dict


def test_NuScenesSCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_bird_eye_view
    import numpy as np
    import random

    preprocess_dir = '/home/Hash-Lee/paper3/3D_Openset_UDA/data/preprocess'
    nuscenes_dir = '/home/Hash-Lee/paper3/3D_Openset_UDA/data/nuscenes'

    # the unknown classes we selected (OS-A)
    unknown_classes = ['bus', 'construction_vehicle']
    # the order of 11 types of fine classes (consistent with preprocess)
    fine_class_names = [
        "car", "truck", "bus", "trailer", "construction_vehicle",
        "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
        "background"
    ]

    print('=== Sanity 1: Source-Train (should be "open-set")===')
    ds_src = NuScenesSCN(
        split=('train_usa',),
        preprocess_dir=preprocess_dir,
        nuscenes_dir=nuscenes_dir,
        merge_classes=True,
        use_image=True,
        noisy_rot=0.1, flip_x=0.5, rot_z=2*np.pi, transl=True,
        fliplr=0.5, color_jitter=(0.4, 0.4, 0.4),
        output_orig=True,               # << must open: return fine labels + points mask
        openset=True, domain='source',  # << key: source-train should be "open-set"
        unknown_classes=unknown_classes
    )

    fine_order_src = ds_src.fine_class_names
    unknown_ids = np.array([fine_order_src.index(n) for n in unknown_classes], dtype=np.int64)
    
    # randomly sample some samples to check "open-set" effect
    for i in random.sample(range(0, len(ds_src)), k=30):
        data = ds_src[i]
        coords = data['coords']
        seg_label_in = data['seg_label']                 # the mapped labels after network (5 classes or -100)
        fine_full = data['orig_seg_label_fine']          # the full fine labels (11 types)
        idxs = data['orig_points_idx']                   # which points entered the network
        fine_in = fine_full[idxs]                        # the fine labels aligned with seg_label_in

        mask_unk_fine = np.isin(fine_in, unknown_ids)    # these should be "open-set"
        # assert: the mapped labels of unknown fine classes must be -100 (ignore)
        if mask_unk_fine.any():
            assert np.all(seg_label_in[mask_unk_fine] == -100), \
                "Source-Train: the unknown fine classes should be mapped to ignore(-100)!"
        # assert: the mapped labels of non-unknown fine classes should be in [0..4]
        if (~mask_unk_fine).any():
            assert np.all((seg_label_in[~mask_unk_fine] >= 0) & (seg_label_in[~mask_unk_fine] <= 4)), \
                "Source-Train: the illegal labels appear on the non-unknown points!"

        unk_ratio = mask_unk_fine.mean() * 100.0
        print(f'[SRC idx={i}] N={len(coords)}, unknown(should be ignored) ratio={unk_ratio:.2f}%')
        # visualize (map ignore points to background/skip drawing)
        vis_lbl = seg_label_in.copy()
        vis_lbl[vis_lbl == -100] = 4   # when visualizing, map ignore to background color
        img = np.moveaxis(data['img'], 0, 2)
        draw_points_image_labels(img, data['img_indices'][idxs], vis_lbl, color_palette_type='NuScenes', point_size=3)
        draw_bird_eye_view(coords)

    print('=== Sanity 2: Target-Val (should not be "open-set"; but should be able to construct unk_gt)===')
    ds_trg = NuScenesSCN(
        split=('val_singapore',),
        preprocess_dir=preprocess_dir,
        nuscenes_dir=nuscenes_dir,
        merge_classes=True,
        use_image=True,
        output_orig=True,                # << must open: construct unk_gt
        openset=True, domain='target',   # << target domain should not be "open-set", just keep information
        unknown_classes=unknown_classes
    )

    fine_order_trg = ds_trg.fine_class_names
    unknown_ids = np.array([fine_order_trg.index(n) for n in unknown_classes], dtype=np.int64)

    for i in random.sample(range(0, len(ds_trg)), k=3):
        data = ds_trg[i]
        coords = data['coords']
        seg_label_in = data['seg_label']                 # the mapped labels after network (should be in [0..4])
        fine_full = data['orig_seg_label_fine']
        idxs = data['orig_points_idx']
        fine_in = fine_full[idxs]

        # assert: the mapped labels of target domain should not be -100 (not "open-set")
        assert np.all(seg_label_in != -100), \
            "Target-Val: the ignore(-100) should not appear, indicating that the target domain is \"open-set\"!"


        # construct unk_gt, used for open-set metrics (AUROC/AUPR/FPR95, etc.)
        unk_gt = np.isin(fine_in, unknown_ids)
        unk_ratio = unk_gt.mean() * 100.0
        print(f'[TRG idx={i}] N={len(coords)}, unknown(used for evaluation) ratio={unk_ratio:.2f}%')

        # visualize: normal draw 5 types of labels; also draw the points of unk_gt (here is omitted)
        img = np.moveaxis(data['img'], 0, 2)
        draw_points_image_labels(img, data['img_indices'][idxs], seg_label_in, color_palette_type='NuScenes', point_size=3)
        draw_bird_eye_view(coords)

    print('✓ Open-set dataloader self-check passed: source domain "open-set" effect, target domain can construct unk_gt.')



def compute_class_weights():
    preprocess_dir = '/home/Hash-Lee/paper3/3D_Openset_UDA/data/preprocess'
    split = ('train_usa', 'test_usa')
    # split = ('train_day', 'test_day')
    dataset = NuScenesBase(split,
                           preprocess_dir,
                           merge_classes=True
                           )
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        points_per_class += np.bincount(dataset.label_mapping[data['seg_labels']], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    test_NuScenesSCN()
    # compute_class_weights()