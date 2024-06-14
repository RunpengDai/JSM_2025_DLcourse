import os.path as osp
from easydict import EasyDict as edict


class Config:
    root_dir = osp.abspath(osp.dirname(__file__))
    output_dir = osp.join(root_dir, 'logs-w-agg-query')
    checkpoint_dir = osp.join(output_dir, "checkpoint")
    enc_data_path = 'data/od.npy'
    #enc_data_path = '/nfsshare/home/TJXY09/yangrun/data/kdd/sz/sz_12_od.npy'
    #dec_data_path = '/nfsshare/home/TJXY09/Runpeng/tmp/original_od/sz_od_data.npy'
    dec_data_path = 'data/chegndu_4ring_13_order_od.npy'

    #enc_poi_path = '/nfsshare/home/TJXY09/yangrun/data/kdd/POI/cd_new_POI.npy'
    #dec_poi_path = '/nfsshare/home/TJXY09/yangrun/data/kdd/POI/cd_old_POI.npy'

    nsparse_points_path = 'data/nsparse_points.npy'
    sparse_points_path = 'data/sparse_points.npy'
    new_idx_path = 'data/new_ids.npy'
    # device
    DEVICE = "mps"
#    DEVICE = "cpu"
    # tensor
    # data
    DATA = edict()
    DATA.NORMALIZE = False
    DATA.T = 48
    #DATA.T = 17
    DATA.NUM_NODE = 632
    DATA.DAYS_TEST=7

    DATA.TIMESTEP = 6
    DATA.HORIZON = 1

    # dataloader
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 4
    DATALOADER.BATCH_SIZE = 16

    # model
    MODEL = edict()
    #MODEL.NUM_NODE = 632
    MODEL.ENC_NUM_NODE =116
    MODEL.DEC_NUM_NODE = 632
    MODEL.NUM_HEADS = 4
    MODEL.EMBED_DIM = 64
    MODEL.DROP_PORB = 0.5
    MODEL.NUM_QUERY = 64
    MODEL.POIDIM = 632

    # solver
    SOLVER = edict()
    SOLVER.LR = 0.001
    SOLVER.CHECKPOINT_PERIOD = 5
    SOLVER.MAX_EPOCH = 30


config = Config()
