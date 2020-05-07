import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = K[
        0,
        2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = K[
        1,
        2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K

oheight, owidth = 352, 1216

class KITTIDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(KITTIDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (210, 840)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth/s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Crop(130, 10, 220, 1200),
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])

        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255	#Why do this??
        # Scipy affine_transform produced RuntimeError when the depth map was
        # given as a 'numpy.ndarray'
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)
        
        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Crop(130, 10, 220, 1200),
            transforms.CenterCrop(self.output_size)
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255      #Why do this??
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        return rgb_np, depth_np
