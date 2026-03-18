
```
## Preparation
### Prerequisites
Tested with
* PyTorch 1.4
* CUDA 10.0
* Python 3.8
* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

### Installation
As 3D network we use SparseConvNet. It requires to use CUDA 10.0 (it did not work with 10.1 when we tried).
We advise to create a new conda environment for installation. PyTorch and CUDA can be installed, and SparseConvNet
installed/compiled as follows:
```
$ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
$ pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
```

### Datasets
#### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org) and extract it.

You need to perform preprocessing to generate the data for M2GO first.
The preprocessing subsamples the 360° LiDAR point cloud to only keep the points that project into
the front camera image. It also generates the point-wise segmentation labels using
the 3D objects by checking which points lie inside the 3D boxes. 
All information will be stored in a pickle file (except the images which will be 
read frame by frame by the dataloader during training).

Please edit the script `M2GO/data/nuscenes/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the NuScenes dataset
* `out_dir` should point to the desired output directory to store the pickle files

#### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and
additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip)
from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract
everything into the same folder.

Similar to NuScenes preprocessing, we save all points that project into the front camera image as well
as the segmentation labels to a pickle file.

Please edit the script `M2GO/data/semantic_kitti/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the SemanticKITTI dataset
* `out_dir` should point to the desired output directory to store the pickle files

## Training
### M2GO
You can run the training with
```
$ cd <root dir of this repo>
$ python M2GO/train_m2go.py --cfg=configs/nuscenes/usa_singapore/M2GO.yaml

```
You can start the trainings on the other UDA scenarios (Day/Night) analogously.

### M2GO<sub>PL</sub>
After having trained the M2GO model, generate the pseudo-labels as follows:
```
$ python M2GO/test.py --cfg=configs/nuscenes/usa_singapore/M2GO.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train_singapore',)"
```
Note that we use the last model at 100,000 steps to exclude supervision from the validation set by picking the best
weights. The pseudo labels and maximum probabilities are saved as `.npy` file.

Please edit the `pselab_paths` in the config file, e.g. `configs/nuscenes/usa_singapore/M2GO_pl.yaml`,
to match your path of the generated pseudo-labels.

Then start the training. The pseudo-label refinement (discard less confident pseudo-labels) is done
when the dataloader is initialized.
```
$ python M2GO/train_m2go.py --cfg=configs/nuscenes/usa_singapore/M2GO_pl.yaml
```

You can start the trainings on the other UDA scenarios (Day/Night) analogously:
```
$ python M2GO/test.py --cfg=configs/nuscenes/day_night/M2GO.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train_night',)"
$ python M2GO/train_M2GO.py --cfg=configs/nuscenes/day_night/M2GO_pl.yaml
```

## Testing
You can provide which checkpoints you want to use for testing. We used the ones
that performed best on the validation set during training (the best val iteration for 2D and 3D is
shown at the end of each training). Note that `@` will be replaced
by the output directory for that config file. For example:
```
$ cd <root dir of this repo>
$ python M2GO/test.py --cfg=configs/nuscenes/usa_singapore/M2GO.yaml @/model_2d_0100000.pth @/model_3d_100000.pth
```
You can also provide an absolute path without `@`. 
