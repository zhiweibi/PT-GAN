from collections import defaultdict
from time import time
from data import BaseDataset
from data.utils import create_modal_mask, permute_modal_names
from data.nii_data_loader import nii_slides_loader, load_set, normalize_nii
import os
import os.path
import numpy as np
import cv2
import torch
import pickle

class BratsDataset(BaseDataset):
    def __init__(self, opt):

        # 1. load form nii file
        if opt.isTrain:
            data_root = opt.dataroot
        else:
            data_root = opt.test_dataroot
        transform = normalize_nii
        loader = nii_slides_loader
        choose_slice_num = 78
        resize = 256

        flair_path = os.path.join(data_root, 'flair')
        t1_path = os.path.join(data_root, 't1')
        t1ce_path = os.path.join(data_root, 't1ce')
        t2_path = os.path.join(data_root, 't2')

        self.flair_set = load_set(flair_path)
        self.t1_set = load_set(t1_path)
        self.t1ce_set = load_set(t1ce_path)
        self.t2_set = load_set(t2_path)

        self.n_data = len(self.flair_set)

        # 2. create modal mask
        modal_names = self.get_modal_names()
        n_modal = len(modal_names)
        self.n_modal = n_modal
        self.modal_mask_dict = create_modal_mask(modal_names)
        self.modal_permutations = permute_modal_names(modal_names)

        # 3. load_all modal into memory
        print('Loading BraTS Dataset...')
        start = time()
        cache_path = os.path.join(data_root, 'cache.pkl')
        if os.path.exists(cache_path):
            print('load data cache from: ', cache_path)
            with open(cache_path, 'rb') as fin:
                self.data_dict = pickle.load(fin)
        else:
            print('load data from raw')
            self.data_dict = defaultdict(list)
            for index in range(self.n_data):
                for modal in ['t1', 't1ce', 't2', 'flair']:
                    modal_path, _ = getattr(self, modal+'_set')[index]
                    modal_img = loader(modal_path, num=choose_slice_num, transform=transform) # np.ndarray, shape=[224,224]
                    modal_img = cv2.resize(modal_img, (resize, resize))
                    self.data_dict[modal].append(modal_img)
            with open(cache_path, 'wb') as fin:
                pickle.dump(self.data_dict, fin)
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        modal_order = self.modal_permutations[index % len(self.modal_permutations)]
        input_modal_names = modal_order[:-1]
        target_modal_name = modal_order[-1]
        # get mask of tartget modal
        target_mask = self.modal_mask_dict[target_modal_name]
        # append target modal mask to every input modal image array
        A = []
        for modal_name in input_modal_names:
            modal_numpy = self.data_dict[modal_name][index // len(self.modal_permutations)]
            modal_with_mask = np.concatenate([modal_numpy[None], target_mask])
            A.append(torch.tensor(modal_with_mask, dtype=torch.float))
        # get ith target modal image array
        target_modal_numpy = self.data_dict[target_modal_name][index // len(self.modal_permutations)]
        input = {
            'A': torch.cat(A),
            'B': torch.tensor(target_modal_numpy[None], dtype=torch.float),
            'modal_names': modal_order
        }

        return input

    def __len__(self):
        return self.n_data * len(self.modal_permutations)

    def get_modal_names(self):
        return ['t1', 't1ce', 't2', 'flair']
