import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import pywt
from skimage.transform import resize
from sktime.transformations.panel.rocket import Rocket
import joblib
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None,rescale_size=64,wt_name='morl',projected_space=256,channel_token_mixing=0,no_rocket=1,half_rocket=0,variation=64):
        self.root_path = root_path
        self.no_rocket=no_rocket
        self.half_rocket=half_rocket
        self.variation=variation

        if os.path.exists(self.root_path+'/'+flag+'_feature_df.csv') and os.path.exists(self.root_path+'/'+flag+'_labels_df.csv'):
            self.feature_df=pd.read_csv(self.root_path+'/'+flag+'_feature_df.csv',index_col=0)
            self.labels_df=pd.read_csv(self.root_path+'/'+flag+'_labels_df.csv',index_col=0)
            self.all_IDs=self.feature_df.index.unique()
            if 'JapaneseVowels' in self.root_path:
                self.max_seq_len=29
            else:
                self.max_seq_len=np.load(self.root_path+'/'+'TRAIN_max_seq_len.npy')
            self.class_names=np.load(self.root_path+'/'+'class_names.npy',allow_pickle=True)
            print("Classes: ",self.class_names)


        else:
            self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
            self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

            if limit_size is not None:
                if limit_size > 1:
                    limit_size = int(limit_size)
                else:  # interpret as proportion if in (0, 1]
                    limit_size = int(limit_size * len(self.all_IDs))
                self.all_IDs = self.all_IDs[:limit_size]
                self.all_df = self.all_df.loc[self.all_IDs]

            # use all features
            self.feature_names = self.all_df.columns
            self.feature_df = self.all_df  
          
            self.feature_df.to_csv(self.root_path+'/'+flag+'_feature_df.csv',index=True)
            self.labels_df.to_csv(self.root_path+'/'+flag+'_labels_df.csv',index=True)

             
        if os.path.exists(f"{self.root_path}/{flag}_{str(self.variation)+'_' if self.variation!=64 else ''}{str(rescale_size)}_.npy") == False:
            self.X_cwt = np.ndarray(shape=(len(self.all_IDs), self.feature_df.shape[1], rescale_size, rescale_size), dtype = 'float32')
            for sample in range(len(self.all_IDs)):
                series=np.array(self.feature_df.loc[self.all_IDs[sample]].values)#L,D
                for signal in range(self.feature_df.shape[1]):
                    coeffs, freqs = pywt.cwt(series[:, signal], self.variation, wt_name)
                    rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode = 'constant')
                    self.X_cwt[sample,signal,:,:] = rescale_coeffs
            np.save(f"{self.root_path}/{flag}_{str(self.variation)+'_' if self.variation!=64 else ''}{str(rescale_size)}_.npy",self.X_cwt)
            # print(self.X_cwt.shape)
        else:
            self.X_cwt=np.load(f"{self.root_path}/{flag}_{str(self.variation)+'_' if self.variation!=64 else ''}{str(rescale_size)}_.npy")
            # print(self.X_cwt.shape)
        # Data normalization
        
        print("Max sequence length: ",self.max_seq_len)
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        self.X_cwt=(self.X_cwt-np.min(self.X_cwt))/(np.max(self.X_cwt)-np.min(self.X_cwt))
        print("Min CWT:",np.min(self.X_cwt))
        print("Max CWT: ",np.max(self.X_cwt))
        print(flag+" :Dataset shape: ",self.feature_df.shape)

        if no_rocket==0:
            if os.path.exists(self.root_path+'/'+flag+'_x_all.npy')==False:
                X_all_np = np.zeros(shape=(len(self.all_IDs), self.feature_df.shape[1], self.max_seq_len), dtype='float32')
                for sample in range(len(self.all_IDs)):
                    series=np.array(self.feature_df.loc[self.all_IDs[sample]].values)#L,D

                    series=np.transpose(series,(1,0))
                    # if series.shape[1] < X_all_np[sample].shape[1]:
                    #     padded_series = np.zeros((series.shape[0], X_all_np[sample].shape[1]))
                    #     padded_series[:, :series.shape[1]] = series
                    #     series = padded_series
                    X_all_np[sample,:,:series.shape[1]] = series

                np.save(self.root_path+'/'+flag+'_x_all.npy',X_all_np)
            else:
                X_all_np=np.load(self.root_path+'/'+flag+'_x_all.npy')
            print("Min raw features:",np.min(X_all_np))
            print("Max raw features: ",np.max(X_all_np))
            # X_all_np=(X_all_np-np.min(X_all_np))/(np.max(X_all_np)-np.min(X_all_np))
            self.rocket_features=np.zeros(shape=(len(self.all_IDs), self.feature_df.shape[1], projected_space), dtype = 'float32')
            os.makedirs(self.root_path+'/rocket'+'_'+str(projected_space), exist_ok=True)
            if channel_token_mixing==0:
                if flag=='TRAIN':
                    existing_models = glob.glob(os.path.join(self.root_path+'/rocket'+'_'+str(projected_space)+'/', f'{flag}_rocket_transformer_*.pkl'))
                    if existing_models:
                        for i,cur_model in enumerate(existing_models):
                            trf=joblib.load(cur_model) 
                            self.rocket_features[:,i,:]=trf.transform(X_all_np)

                    else:
                        random_seeds=np.random.choice(range(10000), X_all_np.shape[1], replace=False)
                        print("Total rocket features: ",random_seeds.shape)
                        for i,curr_random in enumerate(random_seeds):
                            trf = Rocket(num_kernels=projected_space//2, normalise=True,random_state=curr_random)
                            trf.fit(X_all_np)
                            self.rocket_features[:,i,:]=trf.transform(X_all_np)
                            joblib.dump(trf,self.root_path+'/rocket'+'_'+str(projected_space)+'/'+flag+'_rocket_transformer_'+str(curr_random)+'.pkl')
                    np.save(self.root_path+'/'+flag+'_max_of_rocket.npy', np.max(self.rocket_features))
                    np.save(self.root_path+'/'+flag+'_min_of_rocket.npy', np.min(self.rocket_features))
                
                if flag=='TEST':
                    existing_models = glob.glob(os.path.join(self.root_path+'/rocket'+'_'+str(projected_space)+'/', 'TRAIN_rocket_transformer_*.pkl'))
                    if existing_models:
                        print("Found trained rocket transformer for Test")
                        for i,cur_model in enumerate(existing_models):
                            # print("Loading from disk")
                            trf=joblib.load(cur_model) 
                            self.rocket_features[:,i,:]=trf.transform(X_all_np)
                self.rocket_features = np.nan_to_num(self.rocket_features, 
                                                    nan= np.load(self.root_path+'/'+'TRAIN_min_of_rocket.npy'), 
                                                    posinf=np.load(self.root_path+'/'+'TRAIN_max_of_rocket.npy'))

            else:
                print("--Channel token mixing rocket features--")
                if flag=='TRAIN':
                    existing_models_D = glob.glob(os.path.join(self.root_path + '/rocket' + '_' + str(projected_space) + '/', f'{flag}_rocket_transformer_D_*.pkl'))
                    existing_models_X = glob.glob(os.path.join(self.root_path + '/rocket' + '_' + str(projected_space) + '/', f'{flag}_rocket_transformer_X_*.pkl'))

                    half_X = projected_space // 2
                    # Prepare empty arrays for half features
                    rocket_features_D = np.empty((X_all_np.shape[0], X_all_np.shape[1], half_X))
                    rocket_features_X = np.empty((X_all_np.shape[0], X_all_np.shape[1], half_X))


                    if existing_models_X:
                        for i, cur_model in enumerate(existing_models_X):
                            trf = joblib.load(cur_model)
                            rocket_features_X[:, i,:] = trf.transform(X_all_np)
                    else:
                        random_seeds = np.random.choice(range(10000),  X_all_np.shape[1], replace=False)
                        for i, curr_random in enumerate(random_seeds):
                            trf = Rocket(num_kernels=projected_space//4, normalise=True, random_state=curr_random)
                            trf.fit(X_all_np)
                            rocket_features_X[:, i,:] = trf.transform(X_all_np)
                            joblib.dump(trf, os.path.join(self.root_path+'/rocket'+'_'+str(projected_space), f'{flag}_rocket_transformer_X_{curr_random}.pkl'))

                    # Process for Transposed Dimension X
                    X_all_np_transposed = np.transpose(X_all_np, (0, 2, 1))  # Transpose to (B, X, D)
                    if existing_models_D:
                        for i, cur_model in enumerate(existing_models_D):
                            trf = joblib.load(cur_model)
                            rocket_features_D[:, i,:] = trf.transform(X_all_np_transposed)
                    else:
                        for i, curr_random in enumerate(random_seeds):
                            trf = Rocket(num_kernels=projected_space//4, normalise=True, random_state=curr_random)
                            trf.fit(X_all_np_transposed)
                            rocket_features_D[:, i,:] = trf.transform(X_all_np_transposed)
                            joblib.dump(trf, os.path.join(self.root_path+'/rocket'+'_'+str(projected_space), f'{flag}_rocket_transformer_D_{curr_random}.pkl'))

                    # Concatenate along the third dimension
                    self.rocket_features = np.concatenate((rocket_features_D, rocket_features_X), axis=-1)

                    # Save max and min for normalization
                    np.save(os.path.join(self.root_path, f'{flag}_max_of_rocket.npy'), np.max(self.rocket_features))
                    np.save(os.path.join(self.root_path, f'{flag}_min_of_rocket.npy'), np.min(self.rocket_features))           
                if flag=='TEST':
                    half_X = projected_space // 2
                    existing_models_D = glob.glob(os.path.join(self.root_path + '/rocket' + '_' + str(projected_space) + '/', 'TRAIN_rocket_transformer_D_*.pkl'))
                    existing_models_X = glob.glob(os.path.join(self.root_path + '/rocket' + '_' + str(projected_space) + '/', 'TRAIN_rocket_transformer_X_*.pkl'))
                    
                    rocket_features_D = np.empty((X_all_np.shape[0], X_all_np.shape[1], half_X))
                    rocket_features_X = np.empty((X_all_np.shape[0], X_all_np.shape[1], half_X))


                    for i, cur_model in enumerate(existing_models_X):
                        trf = joblib.load(cur_model)
                        rocket_features_D[:, i,:] = trf.transform(X_all_np)

                
                    X_all_np_transposed = np.transpose(X_all_np, (0, 2, 1))
                    for i, cur_model in enumerate(existing_models_D):
                        trf = joblib.load(cur_model)
                        rocket_features_X[:, i,:] = trf.transform(X_all_np_transposed)

                    # Concatenate along the third dimension
                    self.rocket_features = np.concatenate((rocket_features_D, rocket_features_X), axis=2)
                    
                    # Apply normalization using saved max and min
                    self.rocket_features = np.nan_to_num(self.rocket_features,
                                                        nan=np.load(os.path.join(self.root_path, 'TRAIN_min_of_rocket.npy')),
                                                        posinf=np.load(os.path.join(self.root_path, 'TRAIN_max_of_rocket.npy')))          
            print("Min Rocket before normalization:",np.min(self.rocket_features))
            print("Max Rocket before normalization: ",np.max(self.rocket_features))
            self.rocket_features=(self.rocket_features-np.min(self.rocket_features))/(np.max(self.rocket_features)-np.min(self.rocket_features))
            print("Min Rocket:",np.min(self.rocket_features))
            print("Max Rocket: ",np.max(self.rocket_features))
        

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0],flag=flag)  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath,flag='TRAIN'):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        np.save(self.root_path+'/'+'class_names.npy', self.class_names)

        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]
        np.save(self.root_path+'/'+flag+'_max_seq_len.npy', self.max_seq_len)

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        if self.no_rocket==0:
            if self.half_rocket==0:
                return torch.from_numpy(self.X_cwt[ind]),\
                    torch.from_numpy(self.rocket_features[ind]),\
                    torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)
            else:   
                return  torch.from_numpy(self.X_cwt[ind]),\
                        torch.from_numpy(self.rocket_features[ind]), \
                        self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)),\
                        torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)
        elif self.no_rocket==1:
            return  torch.from_numpy(self.X_cwt[ind]),\
                    self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)),\
                    torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)
       

    def __len__(self):
        return len(self.all_IDs)
