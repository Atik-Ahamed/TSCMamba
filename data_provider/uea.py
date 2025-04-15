import os
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore') 

def collate_fn(data, max_len=None,no_rocket=0,half_rocket=0):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    if no_rocket==0:
        # If there are rocket features
        if half_rocket==0:
            # Full rocket features
            batch_size = len(data)
            x_cwt,x_rocket, labels = zip(*data)

        
            # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)

            XCWT = torch.zeros(batch_size, x_cwt[0].shape[0], x_cwt[0].shape[1], x_cwt[0].shape[1])  # (batch_size,feat_dim,resize_scale,resize_scale)
            XROCKET = torch.zeros(batch_size, x_rocket[0].shape[0],x_rocket[0].shape[1])  # (batch_size,feat_dim,projected_dim)
            
            for i in range(batch_size):
                XCWT[i,:,:,:]=x_cwt[i]
            for i in range(batch_size):
                XROCKET[i,:,:]=x_rocket[i]

            targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

            return XCWT,XROCKET, targets
        else:
            # Half-rocket Half-MLP
            batch_size = len(data)
            x_cwt,x_rocket,features, labels = zip(*data)

        
            # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)

            XCWT = torch.zeros(batch_size, x_cwt[0].shape[0], x_cwt[0].shape[1], x_cwt[0].shape[1])  # (batch_size,feat_dim,resize_scale,resize_scale)
            XROCKET = torch.zeros(batch_size, x_rocket[0].shape[0],x_rocket[0].shape[1])  # (batch_size,feat_dim,projected_dim)
            
            for i in range(batch_size):
                XCWT[i,:,:,:]=x_cwt[i]
            for i in range(batch_size):
                XROCKET[i,:,:]=x_rocket[i]
            lengths = [X.shape[0] for X in features]  # original sequence length for each time series
            if max_len is None:
                    max_len = max(lengths)
            X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
            for i in range(batch_size):
                end = min(lengths[i], max_len)
                X[i, :end, :] = features[i][:end, :]
            X= torch.permute(X,(0,2,1)) #B,D,L
            # print("Raw features in collate fn: ",X.shape)
            targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)
            Rocket_and_RAW=torch.zeros(batch_size, x_rocket[0].shape[0],x_rocket[0].shape[1]+X.shape[2])#B,D,Projected_space+L
            Rocket_and_RAW=torch.cat([XROCKET,X],dim=2)

            return XCWT,Rocket_and_RAW, targets            
    elif no_rocket==1:
        batch_size = len(data)
        x_cwt,features, labels = zip(*data)
        XCWT = torch.zeros(batch_size, x_cwt[0].shape[0], x_cwt[0].shape[1], x_cwt[0].shape[1])  # (batch_size,feat_dim,resize_scale,resize_scale)
        for i in range(batch_size):
            XCWT[i,:,:,:]=x_cwt[i]
        targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)
        lengths = [X.shape[0] for X in features]  # original sequence length for each time series
        if max_len is None:
                max_len = max(lengths)
        X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            X[i, :end, :] = features[i][:end, :]
        X= torch.permute(X,(0,2,1)) #B,D,L
        return XCWT,X,targets


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='minmax', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
