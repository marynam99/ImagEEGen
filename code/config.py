import os
import numpy as np

# class Config_EEG_finetune(Config_MBM_finetune):
#     def __init__(self):
        
#         # Project setting
#         self.root_path = './'
#         # self.root_path = '.'
#         self.output_path = './exps/'

#         self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
#         self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')

#         self.dataset = 'EEG' 
#         self.pretrain_mbm_path = './pretrains/eeg_pretrain/checkpoint.pth' 

#         self.include_nonavg_test = True


#         # Training Parameters
#         self.lr = 5.3e-5
#         self.weight_decay = 0.05
#         self.num_epoch = 15
#         self.batch_size = 16 if self.dataset == 'GOD' else 4 
#         self.mask_ratio = 0.5
#         self.accum_iter = 1
#         self.clip_grad = 0.8
#         self.warmup_epochs = 2
#         self.min_lr = 0.
#         self.use_nature_img_loss = False
#         self.img_recon_weight = 0.5
#         self.focus_range = None # [0, 1500] # None to disable it
#         self.focus_rate = 0.6

#         # distributed training
#         self.local_rank = 0
        
class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = './'
        self.output_path = './exps/'

        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = None

        self.img_size = 256

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        # self.lr = 5.3e-5
        self.lr = 5e-4
        self.num_epoch = 500
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True #False
        self.cls_tune = False
        self.subject = 4
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 1
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None 

