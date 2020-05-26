Self Supervised Learning Framework
==============================================================
This branch contains code for the Self-Supervised Learning Framework.
For the code containing preliminary task requirements, kindly refer to the master branch.

Run the following command to check for all the options:
```bash
python3 main.py --help
```
We have trained on the following model (self-supervised + the VGG based model as expected in the problem statement on rgbd data):
```bash
!python3 main.py --pretrained --data kitti_small
```

To evaluate the model use:
```bash
!python main.py --evaluate [path-to-model] --pnp no
```
To evaluate using pnp option (works only for models trained on rgbd data) use:
```bash
!python main.py --evaluate [path-to-model] --pnp yes
```
