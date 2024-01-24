import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import torch.nn.functional as F
import codecs
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset, DataLoader
from typing import Optional

class LSTM(data.Dataset):
    """`LSTM_generated_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    raw_folder = '/gpfs/data1/lianggp/lir/forcast/'
    processed_folder = 'data/'
    train_test_file = 'lstm_generated_refdeduct_h15v04_train_test_data_earthformer.npz'

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None,
                 post_transform=None, post_target_transform=None):
        raw_folder = '/gpfs/data1/lianggp/lir/forcast/'
        processed_folder = 'data/'
        train_test_file = 'lstm_generated_refdeduct_h15v04_train_test_data.npz'
        self.root = raw_folder
        self.transform = transform
        self.target_transform = target_transform
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.split = split
        self.train = train  # training set or test set
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data_X = torch.from_numpy(np.load(
                os.path.join(self.root, self.processed_folder, self. train_test_file))['arr_0'])
            self.train_data_y = torch.from_numpy(np.load(
                os.path.join(self.root, self.processed_folder, self. train_test_file))['arr_2'])
        
        else:
            self.test_data_X = torch.from_numpy(np.load(
                os.path.join(self.root, self.processed_folder, self. train_test_file))['arr_1'])
            self.test_data_y = torch.from_numpy(np.load(
                os.path.join(self.root, self.processed_folder, self. train_test_file))['arr_3'])
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (seq, target) where sampled sequences are split into a seq
                    and target part
        """  
        if self.train:
            seq, target = self.train_data_X[index, :8], self.train_data_y[index, 8:]
        else:
            seq, target = self.test_data_X[index, :8], self.test_data_y[index, 8:]

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data_X)
        else:
            return len(self.test_data_X)
        
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self. train_test_file))
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

class LSTMDataModule():
    raw_folder = '/gpfs/data1/lianggp/lir/forcast/'
    processed_folder = 'data/'
    def __init__(self,
                 root: str = None,
                 val_ratio=0.1, seed=123, batch_size: int = 10):
        """
        Parameters
        ----------
        root
        val_ratio
        batch_size
        rescale_input_shape
            For the purpose of testing. Rescale the inputs
        rescale_target_shape
            For the purpose of testing. Rescale the targets
        """
        super().__init__()
        if root is None:
            root = '/gpfs/data1/lianggp/lir/forcast/'
        self.root = root
        self.val_ratio = val_ratio
        self.seed = seed
        self.batch_size = batch_size
        
    def prepare_data(self):
        LSTM(self.root, train=True)
        LSTM(self.root, train=False)
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_val_data = LSTM(self.root, train=True)
            all_indices = range(len(train_val_data))
            train_indices, val_indices = train_test_split(all_indices, test_size=self.val_ratio, random_state=self.seed)
            self.lstm_train = Subset(train_val_data, train_indices)
            self.lstm_val = Subset(train_val_data, val_indices)

        if stage == "test" or stage is None:
            self.lstm_test = LSTM(self.root, train=False)

        if stage == "predict" or stage is None:
            self.lstm_predict = LSTM(self.root, train=False)
            
    def train_dataloader(self):
        return DataLoader(self.lstm_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.lstm_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.lstm_test, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.lstm_predict, batch_size=self.batch_size, shuffle=False, num_workers=4)

    @property
    def num_train_samples(self):
        return len(self.lstm_train)

    @property
    def num_val_samples(self):
        return len(self.lstm_val)

    @property
    def num_test_samples(self):
        return len(self.lstm_test)

    @property
    def num_predict_samples(self):
        return len(self.lstm_predict)