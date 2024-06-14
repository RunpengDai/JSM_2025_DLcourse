import torch
import numpy as np
from torch.utils.data import Dataset
import logging


class NYCDataset(Dataset):
    def __init__(self, cfg, is_train):
        '''
            expect:
            X = (sample, timestep, map_height * map_width, map_height, map_width)
            Y = (sample, map_height * map_width, map_height, map_width)
            weather = (sample, timestep, ?)
            meta = (sample, timestep, ?)

            The meta data is not used in this work, but we can explore its effect in future works.
        '''
        # parameters
        self.seq_len = cfg.DATA.TIMESTEP
        self.horizon = cfg.DATA.HORIZON
        T = cfg.DATA.T
        days_test = cfg.DATA.DAYS_TEST
        enc_data = np.load(cfg.enc_data_path).astype(np.float32)
        dec_data = np.load(cfg.dec_data_path).astype(np.float32)
        len_test = 2*T * days_test

        nsparse_points = np.load(cfg.nsparse_points_path)
        sparse_points = np.load(cfg.sparse_points_path)
        new_idx = np.load(cfg.new_idx_path)
        #enc_poi = np.load(cfg.enc_poi_path)
        #dec_poi = np.load(cfg.dec_poi_path)
        dec_mask = np.zeros((cfg.DATA.NUM_NODE, cfg.DATA.NUM_NODE))
        enc_mask = np.zeros((cfg.MODEL.ENC_NUM_NODE, cfg.MODEL.ENC_NUM_NODE))
        trans_mat = np.zeros((cfg.DATA.NUM_NODE, cfg.MODEL.ENC_NUM_NODE)).astype(np.float32)
        time_stamp = np.array([ i%24 for i in range(enc_data.shape[0])])
        for idx, value in enumerate(new_idx):
            trans_mat[idx, int(value)]=1
        self.trans_mat = trans_mat

        for i in nsparse_points:
            dec_mask[i, nsparse_points] = 1
        self.dec_mask = torch.from_numpy(dec_mask).bool()

        for i in sparse_points:
            #enc_mask[int(new_idx[i]), :] = 1
            enc_mask[int(new_idx[i]), int(new_idx[i])] = 1
        self.enc_mask = torch.from_numpy(enc_mask).bool()

        if is_train:
            self.enc_data = enc_data[:-len_test] # (768, 116, 116)
            self.dec_data = dec_data[:-len_test] # (768, 632, 632)
            self.time_stamp = time_stamp[:-len_test] # 768,
        else:
            self.enc_data = enc_data[-(len_test//2 ):]
            self.dec_data = dec_data[-(len_test//2 ):]
            self.time_stamp = time_stamp[-(len_test//2 ):]

        self.indices = range(self.enc_data.shape[0] - self.seq_len - self.horizon + 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        img_idx = idx
        x_begin = idx
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.horizon

        enc_x = self.enc_data[x_begin : x_end]
        enc_time = self.time_stamp[x_begin : x_end]
        enc_y = self.enc_data[y_begin : y_end]
        dec_time = self.time_stamp[y_begin : y_end]

        dec_x = self.dec_data[x_begin : x_end]
        dec_y = self.dec_data[y_begin : y_end]
        dec_mask = self.dec_mask
        enc_mask = self.enc_mask
        trans_mat = self.trans_mat
        return {"enc_x": enc_x, "dec_y": dec_y, "dec_x": dec_x ,"enc_y":enc_y, 'dec_mask': dec_mask, 
                "enc_mask": enc_mask, "enc_time":enc_time, "dec_time": dec_time, 'trans_mat':trans_mat, "img_idx": img_idx}


def Add_Window_Horizon(data, window=3, horizon=1):
    length = data.shape[0]
    end_index = length - horizon - window + 1
    X = []
    Y = []
    index = 0
    while index < end_index:
        X.append(data[index:index + window, :, :])
        Y.append(data[index + window:index + window + horizon, :, :])
        index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def ffill(arr):
    out = arr.copy()
    mask = arr==0
    idx = np.where(~mask, np.array([i * np.ones((arr.shape[1], arr.shape[2])) for i in range(arr.shape[0])]))
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = out[idx.astype(int), np.arange(idx.shape[1])[:, None], np.arange(idx.shape[2])[:, None]]
    return out


def dis_mx(num_node, len_row, len_column):
    dis_mx = np.zeros((num_node, num_node))
    dirs = [[0, 1, 1], [1, 0, 1], [-1, 0, 1], [0, -1, 1], [1, 1, 2], [1, -1, 2], [-1, 1, 2], [-1, -1, 2]]
    # [方向，方向，距离]
    for i in range(len_row):
        for j in range(len_column):
            index = i * len_column + j  # grid_id
            for d in dirs:
                nei_i = i + d[0]
                nei_j = j + d[1]
                if 0 <= nei_i < len_row and 0 <= nei_j < len_column:
                    nei_index = nei_i * len_column + nei_j  # neighbor_grid_id
                    dis_mx[index][nei_index] = d[2]
                    dis_mx[nei_index][index] = d[2]
    return dis_mx

def normalize(data):
    mmin = data.min()
    mmax = data.max()
    data = (data - mmin) / (mmax - mmin)
    return data
