import os
import os.path
import glob
import fnmatch
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo, RandomSampling
from dataloaders.pose_estimator import get_pose_pnp
import h5py
input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

IMG_EXTENSIONS = ['.h5', '.png']

def is_image_file(filename):
    return any(os.path.splitext(filename)[-1] == extension for extension in IMG_EXTENSIONS)

def make_dataset_h5(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return np.array(images)

def make_dataset_png(dir):
    images = []
    dir = os.path.expanduser(dir)
    dir_depth = os.path.join(dir,'depth')
    for root, _, fnames in sorted(os.walk(dir_depth)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                depth_path = os.path.join(root, fname)
                rgb_path = os.path.join(root.replace('depth', 'rgb'), fname)
                item = (rgb_path, depth_path)
                images.append(item)
    return np.array(images)

def png_loader(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path) if rgb_path is not None else None
    depth = cv2.imread(depth_path, 0) if depth_path is not None else None
    return rgb, depth

def h5_loader(path):

    if path is None:
        return None, None

    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0)) #HXWXC
    depth = np.array(h5f['depth'])
    return rgb, depth

def load_calib():
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]
    K[0, 2] = K[
        0,
        2] - 13
    K[1, 2] = K[
        1,
        2] - 11.5
    return K

oheight, owidth = 352, 1216


def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth


def train_transform(rgb, sparse, rgb_near, args):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    s = np.random.uniform(1.0,1.5)
    depth_np = sparse/s
    angle = np.random.uniform(-5.0,5.0)
    do_flip = np.random.uniform(0.0,1.0)<0.5

    transform = transforms.Compose([
      transforms.Crop(130,10,220,1200),
      transforms.Rotate(angle),
      transforms.Resize(s),
      transforms.CenterCrop(size=(210,840)),
      transforms.HorizontalFlip(do_flip)
    ])
    rgb_np = transform(rgb)
    rgb_np = color_jitter(rgb_np) # random color jittering
    if(rgb_near is not None):
      rgb_near = transform(rgb_near)
      rgb_near = color_jitter(rgb_near)

    depth_np = np.asfarray(sparse, dtype='float32') /255
    depth_np = transform(depth_np)
    return rgb_np , depth_np, rgb_near


def val_transform(rgb, sparse, rgb_near, args):
    transform = transforms.Compose([
        transforms.Crop(130,10,220,1200),
        transforms.CenterCrop(size=(210,840))
    ])
    if rgb is not None:
        rgb = transform(rgb)
        #rgb = np.asfarray(rgb,dtype='float')/255
    if sparse is not None:
        sparse = np.asfarray(sparse,dtype='float')/ 255
        sparse = transform(sparse)
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
        #rgb_near = np.asfarray(rgb,dtype='float')/255
    return rgb, sparse, rgb_near


def no_transform(rgb, sparse, rgb_near, args):
    return rgb, sparse, rgb_near


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

def get_rgb_near(imgs,index, args):

    max_frame_diff = 3
    candidates = [
        i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]

    if args.loader == png_loader:
        head1,_ = os.path.split(imgs[index][0])

    if args.loader == h5_loader:
        head1,_ = os.path.split(imgs[index])

    while True:
        random_offset = choice(candidates)
        if( (index + random_offset) >= len(imgs) ):
            continue

        path_near = imgs[index + random_offset]

        if args.loader == png_loader:
            head2,_ = os.path.split(imgs[index][0])

        if args.loader == h5_loader:
            head2,_ = os.path.split(imgs[index])

        if os.path.exists(path_near) and head2 == head1:
            break


    return path_near

class KittiDepth(data.Dataset):

    """A data loader for the Kitti dataset
    """
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
    def __init__(self,split,args):
        if(args.loader == png_loader):
            imgs = make_dataset_png(args.root)
        if(args.loader == h5_loader):
            imgs = make_dataset_h5(args.root)


        assert len(imgs)>0, "Found 0 images in subfolders of: " + args.root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), args.root))
        self.imgs = imgs

        if split == 'train':
            self.transform = train_transform
        elif split == 'val':
            self.transform = val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = args.loader

        self.modality = args.input
        self.sparsifier = args.sparsifier
        self.args = args
        self.split = split
        self.K = load_calib()
        self.threshold_translation = 0.1
        self.output_size = (210, 840)

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2) #Use the tensor version of expand_dims...
        return rgbd

    def __getraw__(self, index):

        if self.loader == png_loader:
            rgb_path, depth_path = self.imgs[index]
            rgb, depth = self.loader(rgb_path, depth_path)
            rgb_near_path = get_rgb_near(self.imgs,index,self.args) if\
                self.split == 'train' and self.args.use_pose else None
            rgb_near,_ = self.loader(rgb_near_path,None)
        if self.loader == h5_loader:
            path = self.imgs[index]
            rgb, depth = self.loader(path)
            rgb_near_path = get_rgb_near(self.imgs, index,self.args) if\
                self.split == 'train' and self.args.use_pose else None
            rgb_near,_ = self.loader(rgb_near_path)
        return rgb, depth, rgb_near

    def __getitem__(self, index):
        rgb, sparse,  rgb_near = self.__getraw__(index)
        rgb, sparse,  rgb_near = self.transform(rgb, sparse, rgb_near, self.args)
        r_mat, t_vec = None, None
        if self.split == 'train' and self.args.use_pose:
            success, r_vec, t_vec = get_pose_pnp(rgb, rgb_near, sparse, self.K)
            # discard if translation is too small
            success = success and LA.norm(t_vec) > self.threshold_translation
            if success:
                r_mat, _ = cv2.Rodrigues(r_vec)
            else:
                rgb_near = rgb
                t_vec = np.zeros((3, 1))
                r_mat = np.eye(3)

        gray = None

        candidates = {"rgb":rgb, "d":sparse, \
            "g":gray, "r_mat":r_mat, "t_vec":t_vec, "rgb_near":rgb_near,"rgbd":self.create_rgbd(rgb,sparse)}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.imgs)
