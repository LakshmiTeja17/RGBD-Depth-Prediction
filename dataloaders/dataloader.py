import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py #https://www.h5py.org/
import dataloaders.transforms as transforms
import cv2
from random import choice
from PIL import Image
from pose_estimator import get_pose_pnp
from numpy import linalg as LA
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

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

def get_rgb_near(path):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 3
    candidates = [
        i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]
    while True:
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number + random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path)

    return rgb_read(path_near)



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
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, 0)
    #rgb = np.transpose(rgb, (1, 2, 0))  #HXWXC
    return rgb, depth

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0)) #HXWXC
    depth = np.array(h5f['depth'])
    return rgb, depth

to_tensor = transforms.ToTensor()

class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader):
        
        if(loader == png_loader):
            imgs = make_dataset_png(root)
        if(loader == h5_loader):
            imgs = make_dataset_h5(root)
            
#         if(type == val and len(imgs) > 3200):
#             np.random.shuffle(imgs)
#             imgs = imgs[:3200]
            
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.threshold_translation = 0.1
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

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
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        if self.loader == png_loader:
            rgb_path, depth_path = self.imgs[index]
            count = 0
            max_frame_diff = 3
            candidates = [
                i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
                if i - max_frame_diff != 0
            ]
            while True:
                random_offset = choice(candidates)
                path_near = self.imgs[index+random_offset]
                if os.path.exists(path_near):
                    break
                assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(rgb_path)
            rgb_near = cv2.imread(path_near)
            depth_near = cv2.imread(path_near)
            rgb, depth = self.loader(rgb_path, depth_path)
        if self.loader == h5_loader:
            path = self.imgs[index]
            count = 0
            max_frame_diff = 3
            candidates = [
                i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
                if i - max_frame_diff != 0
            ]
            while True:
                random_offset = choice(candidates)
                path_near = self.imgs[index+random_offset]
                if os.path.exists(path_near):
                    break
                assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path)
            rgb, depth = self.loader(path)
            rgb_near,depth_near = self.loader(path_near)
        return rgb, depth , rgb_near,depth_near

    def __getitem__(self, index):
        rgb, depth,rgb_near,depth_near= self.__getraw__(index)

        if self.transform is not None:
            rgb_np = rgb
            depth_np = depth
            rgb_near_np = rgb_near
            # depth_near_np = depth_near
            rgb_np, depth_np = self.transform(rgb, depth)
            rgb_near_np, _ = self.transform(rgb_near, depth_near)
        else:
            raise(RuntimeError("transform not defined"))

        r_mat,t_vec = None,None
        if self.split == 'train' and self.args.use_pose:
            success, r_vec, t_vec = get_pose_pnp(rgb, rgb_near, depth, self.K)
            # discard if translation is too small
            success = success and LA.norm(t_vec) > self.threshold_translation
            if success:
                r_mat, _ = cv2.Rodrigues(r_vec)
            else:
                # return the same image and no motion when PnP fails
                rgb_near = rgb
                t_vec = np.zeros((3, 1))
                r_mat = np.eye(3)

        # rgb, gray = handle_gray(rgb, self.args)
        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)

        input_tensor = to_tensor(input_np)
        rgb_near_tensor = to_tensor(rgb_near_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)
        r_mat_tensor = to_tensor(r_mat)
        t_vec_tensor = to_tensor(t_vec)
        return input_tensor, depth_tensor,rgb_near_tensor,r_mat_tensor,t_vec_tensor

    def __len__(self):
        return len(self.imgs)
