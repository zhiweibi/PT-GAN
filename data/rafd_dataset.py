import os
from collections import defaultdict
from time import time
from data.multi_modal_dataset import MultiModalDataset
from PIL import Image


class RafdDataset(MultiModalDataset):

    def __init__(self, opt):
        
        super(RafdDataset, self).__init__(opt)

    def get_modal_names(self):
        return ['sad', 'neutral', 'angry', 'contemptuous', 'disgusted', 'surprised', 'fearful', 'happy']

    def load_img_paths(self):

        key2name = {}
        for img_name in os.listdir(self.dataroot):
            if img_name[-4:] != '.jpg':
                continue
            camangle, pid, race, gender, motion, gaze = img_name.split('.')[0].split('_')
            key2name[pid] = '_'.join([camangle, pid, race, gender])

        prange = list(key2name.keys())
        targetp = self.opt.targetp
        if self.isTrain:
            for p in targetp:
                prange.remove(p)
        else:
            prange = targetp

        datapath_dict = defaultdict(list)
        for pid in prange:
            for gaze in ['frontal', 'left', 'right']:
                for modal in self.get_modal_names():
                    img_prefix = key2name[pid]
                    img_path = os.path.join(self.dataroot, img_prefix + '_{}_{}.jpg'.format(modal, gaze))
                    datapath_dict[modal].append(img_path)

        return datapath_dict

    def load_data(self):
        datapath_dict = self.load_img_paths()
        self.n_data = len(datapath_dict[self.modal_names[0]])
        print('Loading {} Dataset...'.format(self.dataset))
        start = time()
        print('load data from raw')
        data_dict = defaultdict(list)
        for index in range(self.n_data):
            for modal in self.modal_names:
                img = Image.open(datapath_dict[modal][index]).convert('RGB')
                data_dict[modal].append(img)
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))
        return data_dict
