sparse-to-dense
============================
This repo can be used for training and testing of
- RGB (or grayscale image) based depth prediction
- sparse depth based depth prediction
- RGBd (i.e., both RGB and sparse depth) based depth prediction



## Contents
0. [Summary](#summary)
0. [Dependencies](#dependencies)
0. [Training](#training)
0. [Testing](#testing)
0. [Models](#models)
0. [Metrics](#metrics)
0. [Results](#results)
0. [References](#references)

## Summary
The project considers using sparse depth samples along with RGB images to generate dense depth map as shown in this [paper](https://arxiv.org/abs/1709.07492). The model has been trained on the kitti odometry dataset, which contains 22 sequences. We have replaced feature extractor from RESNET-18 to VGGNet. Additionally, we have used nearest neighbour upsampling instead of bilinear interpolation on the output of the decoder unit. Uniform Random Sampling has been done with depth points limited to 20,000.Moreover, certain improvements have been used over the proposed model such as self supervised depth completion neural network framework for getting better prediction from training data and Plug-and-play module to generate better results from the existing model on the test data.   
## Dependencies
This code was tested with Python 3 and PyTorch 0.4.0.
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

For instance, run the following command to train a network with ResNet50 as the encoder, deconvolutions of kernel size 3 as the decoder, and both RGB and 100 random sparse depth samples as the input to the network.
```bash
python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100 --data kitti
```

Training results will be saved under the `results` folder. To resume a previous training, run
```bash
python3 main.py --resume [path_to_previous_model]
```

## Testing
To test the performance of a trained model without training, simply run main.py with the `-e` option. For instance,
```bash
python3 main.py --evaluate [path_to_trained_model]
```

## Models
A number of trained models is available [here](https://drive.google.com/drive/folders/19IoDXg-lS6gPHgZh4m_63zrEdaTA0aQ_?usp=sharing). 

## Metrics
- Error metrics on KITTI dataset:

	| MODEL     |  RMSE(in mm)  |  ABSREL  | DELTA1 | DELTA2 |
	|-----------|:-------------:|:--------:|:------:|:------:|
	| VGG_RGB   | 4780.1        | 0.118    | 84.9   | 95.38  |
	| VGG_RGBD  | 3729.031      | 0.0712   | 93.00  | 97.34  |
	| RESNET_RGB| 4858.7        | 0.1205   | 84.51  | 95.22  |
	| RESNET_RGBD| 3798.221     | 0.0712   | 92.79  | 97.18  |
	| SELF_VGG_RGBD| 2486.115   | 0.058    | 96.15  | 98.13  |
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
Top Row showing the result on rgbd modality
<br>
Bottom Row showing the result on rgb modality


## References
We have used below sources for the purpose of this project and acknowledge the use of code from these sources:

	Fangchang Ma, et al. "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image." (2017).

	Ma, Fangchang et al. "Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera". arXiv preprint arXiv:1807.00275. (2018).

	Tsun-Hsuan Wang, et al. "Plug-and-Play: Improve Depth Estimation via Sparse Data Propagation." (2018).
