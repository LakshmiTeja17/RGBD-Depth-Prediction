sparse-to-dense
============================
This repo can be used for training and testing of
- RGB based depth prediction
- sparse depth based depth prediction
- RGBd (i.e., both RGB and sparse depth) based depth prediction

This branch has our work on the model that was expected of us in the problem statement  (VGG + Random Sampling + Nearest Neighbor Interpolation). Do remember to checkout the branch adhyay2000 for our work on training the same model using self-supervised learning.


## Contents
0. [Task_Statement](#task_statement)
0. [Additional_Work](#additional_work)
0. [Dependencies](#dependencies)
0. [Training](#training)
0. [Testing](#testing)
0. [Models](#models)
0. [Metrics](#metrics)
0. [Results](#results)
0. [References](#references)

## Task_statement
Just train on a small subset of KITTI. (We have trained on the whole KITTI odometry dataset)

1) Replace the feature extractor from RESNET-18 to VGGNet for KITTI

2) Use Nearest Neighbour upsampling instead of bilinear interpolation.

3) Use Uniform random sampling with depth points restricted in numbers to 20000.

Compare results with the paper.

## Additional_work
Improvements have been used over the proposed model. A self supervised framework was used for getting better accuracies. A Plug-and-play module was also used (https://arxiv.org/pdf/1812.08350.pdf) to generate better results during evaluation of the model.

## Dependencies
- Install [PyTorch](http://pytorch.org/) on a machine with CUDA GPU.
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) and other dependencies (files in our pre-processed datasets are in HDF5 formats).
	```bash
	sudo apt-get update
	sudo apt-get install -y libhdf5-serial-dev hdf5-tools
	pip3 install h5py matplotlib imageio scikit-image
	pip3 install opencv-python==3.4.2.16
	pip3 install opencv-contrib-python==3.4.2.16
	```
## Training
The training scripts come with several options, which can be listed with the `--help` flag.
```bash
python3 main.py --help
```

For instance, run the following command to train a network with ResNet18 as the encoder, upprojection as the decoder, and both RGB and 100 random sparse depth samples as the input to the network and uniform random sampling as the sparsifier. (For the sparsifier using Random Sampling as required by the problem statement: use --sparsifier ran instead)
```bash
python3 main.py -a resnet18 -d upproj -m rgbd -s 100 --data kitti --sparsifier uar
```

We have trained using the following options:
```bash
python3 main.py -a vgg16 -d upproj -m rgbd -s 100 --data kitti --sparsifier ran
python3 main.py -a vgg16 -d upproj -m rgb -s 100 --data kitti --sparsifier ran
python3 main.py -a resnet18 -d upproj -m rgbd -s 100 --data kitti --sparsifier uar
python3 main.py -a resnet18 -d upproj -m rgb -s 100 --data kitti --sparsifier uar
```

Training results will be saved in a folder under the `results` folder. The folder's name contain the command-line arguments we used (If some argument not mentioned as command-line argument, a default value is used). To resume a previous training (A checkpoint is saved after every epoch), run
```bash
python3 main.py --resume [path_to_previous_model]
```

## Testing
To test the performance of a trained model without training, simply run main.py with the `--evaluate` option. For instance,
```bash
python3 main.py --evaluate [path_to_trained_model]
```
To test the performance using the plug-and-play module:
```bash
python3 main.py --evaluate [path_to_trained_model] --pnp yes
```
Remember that plug and play module can be used only on models trained using rgbd input. It won't work if only rgb is given as input. (That is how the algorithm is designed)

## Models
Our trained models are available [here](https://drive.google.com/drive/folders/19IoDXg-lS6gPHgZh4m_63zrEdaTA0aQ_?usp=sharing).

The names of the models are self explanatory.
This folder also has graphs: rmse.png, absrel.png, delta1.png and delta2.png to graphically visualize the performance of our model on



## Metrics
- Error metrics on KITTI dataset:

	| MODEL     |  RMSE(in mm)  |  ABSREL  | DELTA1 | DELTA2 |
	|-----------|:-------------:|:--------:|:------:|:------:|
	| VGG_RGB   | 4780.1        | 0.118    | 84.9   | 95.38  |
	| VGG_RGBD  | 3729.031      | 0.0712   | 93.00  | 97.34  |
	| RESNET_RGB| 4858.7        | 0.1205   | 84.51  | 95.22  |
	| RESNET_RGBD| 3798.221     | 0.0712   | 92.79  | 97.18  |
	| SELF_VGG_RGBD| 2486.115   | 0.058    | 96.15  | 98.13  |
	| SELF_VGG_RGBD_PNP| 2434.896 | 0.056  | 96.74  | 98.25  |
	| VGG_RGBD_PNP | 3724.902   | 0.0697   | 93.01  | 97.31  |
## Results
<p float="left">
  <img src="/images/rmse.png" width="300" />
  <img src="/images/delta1.png" width="300" />
</p>
<p float="left">
  <img src="/images/delta2.png" width="300" />
  <img src="/images/absrel.png" width="300" />
</p>
Results plotted against number of samples
<br>
<p float="left">
  <img src="/images/resnet_rgbd.png" width="300" />
  <img src="/images/vgg_rgbd.png" width="300" />
</p>
<p float="left">
  <img src="/images/resnet_rgb.png" width="300" />
  <img src="/images/vgg_rgb.png" width="300" />
</p>
Top Row showing the result on rgbd modality for ResNet(Left) and VGG(right)
<br>
Bottom Row showing the result on rgb modality for ResNet(Left) and VGG(right)


## References
We have used below sources for the purpose of this project and acknowledge the use of code from these sources:

	Fangchang Ma, et al. "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image." (2017).

	Ma, Fangchang et al. "Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera". arXiv preprint arXiv:1807.00275. (2018).

	Tsun-Hsuan Wang, et al. "Plug-and-Play: Improve Depth Estimation via Sparse Data Propagation." (2018).
