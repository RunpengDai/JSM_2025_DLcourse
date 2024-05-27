from easydict import EasyDict as edict

DeepSurv_config = edict()
DeepSurv_config.epochs = 500
DeepSurv_config.learning_rate = 0.067
DeepSurv_config.lr_decay_rate = 6.494e-4
DeepSurv_config.optimizer = 'Adam'
DeepSurv_config.drop = 0.147
DeepSurv_config.norm = True
DeepSurv_config.dims = [6, 48, 48, 1]
DeepSurv_config.activation = 'ReLU'
DeepSurv_config.l2_reg = 0
DeepSurv_config.model_path = "model/deepsurv.pth"