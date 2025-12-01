"""xMUDA experiments configuration"""
import os.path as osp

from xmuda.common.config.base import CN, _C

# public alias
cfg = _C
_C.VAL.METRIC = 'seg_iou'

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN.CLASS_WEIGHTS = []

# ---------------------------------------------------------------------------- #
# Visualization options (validate/test)
# ---------------------------------------------------------------------------- #
_C.VAL.SAVE_VIZ = False
_C.VAL.VIZ_EVERY = 10
_C.VAL.VIZ_MAX = 50
_C.VAL.HEAD_UNKNOWN_THR_2D = 0.2
_C.VAL.HEAD_UNKNOWN_THR_3D = 0.05

# ---------------------------------------------------------------------------- #
# xMUDA options
# ---------------------------------------------------------------------------- #
_C.TRAIN.XMUDA = CN()
_C.TRAIN.XMUDA.lambda_xm_src = 0.0
_C.TRAIN.XMUDA.lambda_xm_trg = 0.0
_C.TRAIN.XMUDA.lambda_pl = 0.0
_C.TRAIN.XMUDA.lambda_minent = 0.0
_C.TRAIN.XMUDA.lambda_logcoral = 0.0

# Head-expansion with pseudo labels: map low-confidence PL to Unknown
_C.TRAIN.XMUDA.PL_TO_UNK = CN()
_C.TRAIN.XMUDA.PL_TO_UNK.enable = False
# use fixed threshold on max prob; if None, use quantile q
_C.TRAIN.XMUDA.PL_TO_UNK.thr = None
_C.TRAIN.XMUDA.PL_TO_UNK.q = 0.2
# optional per-modality overrides (fallback to thr/q above when None)
_C.TRAIN.XMUDA.PL_TO_UNK.thr2d = None
_C.TRAIN.XMUDA.PL_TO_UNK.thr3d = None
_C.TRAIN.XMUDA.PL_TO_UNK.q2d = None
_C.TRAIN.XMUDA.PL_TO_UNK.q3d = None

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASET_SOURCE = CN()
_C.DATASET_SOURCE.TYPE = ''
_C.DATASET_SOURCE.TRAIN = tuple()

_C.DATASET_TARGET = CN()
_C.DATASET_TARGET.TYPE = ''
_C.DATASET_TARGET.TRAIN = tuple()
_C.DATASET_TARGET.VAL = tuple()
_C.DATASET_TARGET.TEST = tuple()

# NuScenesSCN
_C.DATASET_SOURCE.NuScenesSCN = CN()
_C.DATASET_SOURCE.NuScenesSCN.preprocess_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.nuscenes_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.NuScenesSCN.scale = 20
_C.DATASET_SOURCE.NuScenesSCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.NuScenesSCN.use_image = True
_C.DATASET_SOURCE.NuScenesSCN.resize = (400, 225)
_C.DATASET_SOURCE.NuScenesSCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation = CN()
_C.DATASET_SOURCE.NuScenesSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.NuScenesSCN.augmentation.flip_x = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.NuScenesSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.NuScenesSCN = CN(_C.DATASET_SOURCE.NuScenesSCN)
_C.DATASET_TARGET.NuScenesSCN.pselab_paths = tuple()

# A2D2SCN
_C.DATASET_SOURCE.A2D2SCN = CN()
_C.DATASET_SOURCE.A2D2SCN.preprocess_dir = ''
_C.DATASET_SOURCE.A2D2SCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.A2D2SCN.scale = 20
_C.DATASET_SOURCE.A2D2SCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.A2D2SCN.use_image = True
_C.DATASET_SOURCE.A2D2SCN.resize = (480, 302)
_C.DATASET_SOURCE.A2D2SCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.A2D2SCN.augmentation = CN()
_C.DATASET_SOURCE.A2D2SCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.A2D2SCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.A2D2SCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.A2D2SCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.A2D2SCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.A2D2SCN.augmentation.color_jitter = (0.4, 0.4, 0.4)

# SemanticKITTISCN
_C.DATASET_SOURCE.SemanticKITTISCN = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.preprocess_dir = ''
_C.DATASET_SOURCE.SemanticKITTISCN.semantic_kitti_dir = ''
_C.DATASET_SOURCE.SemanticKITTISCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.SemanticKITTISCN.scale = 20
_C.DATASET_SOURCE.SemanticKITTISCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.SemanticKITTISCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.bottom_crop = (480, 302)
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.SemanticKITTISCN = CN(_C.DATASET_SOURCE.SemanticKITTISCN)
_C.DATASET_TARGET.SemanticKITTISCN.pselab_paths = tuple()

# ---------------------------------------------------------------------------- #
# Model 2D
# ---------------------------------------------------------------------------- #
_C.MODEL_2D = CN()
_C.MODEL_2D.TYPE = ''
_C.MODEL_2D.CKPT_PATH = ''
_C.MODEL_2D.NUM_CLASSES = 5
_C.MODEL_2D.DUAL_HEAD = False
# ---------------------------------------------------------------------------- #
# UNetResNet34 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.UNetResNet34 = CN()
_C.MODEL_2D.UNetResNet34.pretrained = True

# ---------------------------------------------------------------------------- #
# Model 3D
# ---------------------------------------------------------------------------- #
_C.MODEL_3D = CN()
_C.MODEL_3D.TYPE = ''
_C.MODEL_3D.CKPT_PATH = ''
_C.MODEL_3D.NUM_CLASSES = 5
_C.MODEL_3D.DUAL_HEAD = False
# ----------------------------------------------------------------------------- #
# SCN options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SCN = CN()
_C.MODEL_3D.SCN.in_channels = 1
_C.MODEL_3D.SCN.m = 16  # number of unet features (multiplied in each layer)
_C.MODEL_3D.SCN.block_reps = 1  # block repetitions
_C.MODEL_3D.SCN.residual_blocks = False  # ResNet style basic blocks
_C.MODEL_3D.SCN.full_scale = 4096
_C.MODEL_3D.SCN.num_planes = 7

# ----------------------------------------------------------------------------- #
# Open-set options
# ----------------------------------------------------------------------------- #
_C.OPENSET = CN()
_C.OPENSET.ENABLE = False
_C.OPENSET.UNKNOWN_CLASSES = []
_C.OPENSET.UNK_SCORE = CN()
_C.OPENSET.UNK_SCORE.alpha = 1.0
_C.OPENSET.UNK_SCORE.beta = 1.0
_C.OPENSET.UNK_SCORE.gamma = 1.0
_C.OPENSET.THRESHOLD = CN()
_C.OPENSET.THRESHOLD.type = 'quantile'
_C.OPENSET.THRESHOLD.q = 0.8
_C.OPENSET.LAMBDA_UNK = 0.0
_C.OPENSET.UNK_RAMP_ITERS = 0  # linear ramp steps after WARMUP_ITERS; 0=disabled (constant)
_C.OPENSET.WARMUP_ITERS = 1000
_C.OPENSET.LOG_HIST_EVERY = 500
_C.OPENSET.GATE_XM = True
_C.OPENSET.GATE_PL = False
_C.OPENSET.GATE_MINENT = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# @ will be replaced by config path
_C.OUTPUT_DIR = osp.expanduser('~/workspace/outputs/xmuda/@')