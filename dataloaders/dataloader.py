import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py #https://www.h5py.org/
import dataloaders.transforms as transforms
import cv2

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
            rgb, depth = self.loader(rgb_path, depth_path)
        if self.loader == h5_loader:
            path = self.imgs[index]
            rgb, depth = self.loader(path)
            
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)

        if self.transform is not None:
            rgb_np = rgb
            depth_np = depth
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))


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
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)
