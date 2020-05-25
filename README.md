sparse-to-dense
============================
This repo can be used for training and testing of
- RGB (or grayscale image) based depth prediction
- sparse depth based depth prediction
- RGBd (i.e., both RGB and sparse depth) based depth prediction



## Contents
0. [Summary](#Summary)
0. [Dependencies](#Dependencies)
0. [Training](#training)
0. [Testing](#testing)
0. [Benchmark](#benchmark)
0. [Results](#Results)
0. [References](#References)

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

## Trained Models
A number of trained models is available [here](https://drive.google.com/drive/folders/19IoDXg-lS6gPHgZh4m_63zrEdaTA0aQ_?usp=sharing). 

## Benchmark
- Error metrics on KITTI dataset:

	| RGB     |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Make3D](http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) | 8.734 | 0.280 | 60.1 | 82.0 | 92.6 |
	| [Mancini et al](https://arxiv.org/pdf/1607.06349.pdf) (_IROS 2016_)  | 7.508 | - | 31.8 | 61.7 | 81.3 |
	| [Eigen et al](http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) (_NIPS 2014_)  | 7.156 | **0.190** | **69.2** | 89.9 | **96.7** |
	| Ours-RGB             | **6.266** | 0.208 | 59.1 | **90.0** | 96.2 |

	| RGBd-#samples   |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Cadena et al](https://pdfs.semanticscholar.org/18d5/f0747a23706a344f1d15b032ea22795324fa.pdf) (_RSS 2016_)-650 | 7.14 | 0.179 | 70.9 | 88.8 | 95.6 |
	| Ours-50 | 4.884 | 0.109 | 87.1 | 95.2 | 97.9 |
	| [Liao et al](https://arxiv.org/abs/1611.02174) (_ICRA 2017_)-225 | 4.50 | 0.113 | 87.4 | 96.0 | 98.4 |
	| Ours-100 | 4.303 | 0.095 | 90.0 | 96.3 | 98.3 |
	| Ours-200 | 3.851 | 0.083 | 91.9 | 97.0 | 98.6 |
	| Ours-500| **3.378** | **0.073** | **93.5** | **97.6** | **98.9** |
	Image to be inserted here.
	<!-- <img src="http://www.mit.edu/~fcma/images/ICRA18/acc_vs_samples_kitti.png" alt="photo not available" width="50%" height="50%"> -->

	<!-- Note: our networks are trained on the KITTI odometry dataset, using only sparse labels from laser measurements. -->

## Results

## References
We have used below sources for the purpose of this project and acknowledge the use of code from these sources:

	Fangchang Ma, et al. "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image." (2017).

	Ma, Fangchang et al. "Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera". arXiv preprint arXiv:1807.00275. (2018).

	Tsun-Hsuan Wang, et al. "Plug-and-Play: Improve Depth Estimation via Sparse Data Propagation." (2018).
