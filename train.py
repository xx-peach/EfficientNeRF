import os, sys

from torch._C import device, dtype
from opt import get_opts
import torch
import torch.nn.functional as F
from collections import defaultdict
from torchvision import transforms

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from spherical_harmonic import eval_sh_torch
from models.sh import eval_sh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import time
import imageio.v2 as imageio
import glob

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

class NerfTree_Pytorch(object):  # This is only based on Pytorch implementation
    def __init__(self, xyz_min, xyz_max, grid_coarse, grid_fine, deg, sigma_init, sigma_default, device):
        ''' this is only based on Pytorch implementation
        arguments:
            xyz_min       - list (3,) or (1, 3), default = [-coord_scope, -coord_scope, -coord_scope] <- [-3.0, -3.0, -3.0]
            xyz_max       - list (3,) or (1, 3), default = [ coord_scope,  coord_scope,  coord_scope] <- [ 3.0,  3.0,  3.0]
            grid_coarse   - int, the grid resolution of coarse density Voxels Vc, default = 384
            grid_fine     - int, the additional grid resolution of fine density Voxels Vf inside each coarse density voxel, default = 3
            deg           - 
            sigma_init    - float, the initial volume density of all voxels inside the coarse density Voxels Vc, default = 30.0
            sigma_default - float, the initial volume density of all voxels inside the fine density Voxels Vf, default = -20.0 
        '''
        super().__init__()
        self.sigma_init = sigma_init        # initial volume density for coarse density voxels, default = 30.0
        self.sigma_default = sigma_default  # initial volume density for fine density voxels, default = -20.0

        # (Dc, Dc, Dc) == (384, 384, 384) of (1,), the coarse density Voxels Vc which stores only the lateset density inside
        self.sigma_voxels_coarse = torch.full((grid_coarse,grid_coarse,grid_coarse), self.sigma_init, device=device)        # (Dc, Dc, Dc)
        self.index_voxels_coarse = torch.full((grid_coarse,grid_coarse,grid_coarse), 0, dtype=torch.long, device=device)    # (Dc, Dc, Dc)
        self.voxels_fine = None

        self.xyz_min = xyz_min[0]                           # default = -3.0
        self.xyz_max = xyz_max[0]                           # default =  3.0
        self.xyz_scope = self.xyz_max - self.xyz_min        # default =  6.0

        self.grid_coarse = grid_coarse              # default = 384
        self.grid_fine = grid_fine                  # default = 3
        self.res_coarse = grid_coarse               # 384, 即对于 coarse density Voxels Vc 来说，它把整个 [xyz_min, xyz_max] 场景分成了 [384, 384, 384] 份
        self.res_fine = grid_coarse * grid_fine     # 1152，而 fine density Voxels Vf 来说，它是将 coarse density Voxels Vc 的每个 voxel 又分成了 [3, 3, 3] 份
        
        self.dim_sh = 3 * (deg + 1)**2
        self.device = device
    
    def calc_index_coarse(self, xyz):
        """ compute the coarse grid coordinate of each sampled point in world coordinate
        arguments:
            xyz - (N, 3), world coordinates of all the sampled points
        returns:
            ijk_coarse - (N, 3), grid coarse coordinates of all the sampled points
        """
        ijk_coarse = ((xyz - self.xyz_min) / self.xyz_scope * self.grid_coarse).long().clamp(min=0, max=self.grid_coarse-1)
        # return index_coarse[:, 0] * (self.grid_coarse**2) + index_coarse[:, 1] * self.grid_coarse + index_coarse[:, 2]
        return ijk_coarse

    def update_coarse(self, xyz, sigma, beta):
        ''' update the stored volume density inside the coarse density Voxels Vc using the valid density just predicted by coarse MLP
        arguments:
            xyz   - (Nv, 3), world coordinates of all the sampled points
            sigma - (Nv,), densities that just predicted by the coarse MLP of all the sampled points
        returns:
            self.sigma_voxels_coarse - lateset updated coarse density Voxels Vc
        '''
        # compute the coarse grid coordinate of each sampled point
        ijk_coarse = self.calc_index_coarse(xyz)
        # update the coarse density Voxels Vc using beta momumtum mathod
        self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] \
                    = (1 - beta) * self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] + \
                        beta * sigma
    
    def create_voxels_fine(self):
        """ create fine density Voxels Vf to those coarse voxels with its sigma(density) > 0 at each training step
            每一个 training iteration 开始时都会调用该函数，根据 latest coarse Voxels Vc 来初始化一个新的 fine density Voxels Vf
        """
        # https://pytorch.org/docs/stable/generated/torch.logical_and.html
        # (Nv, 3), indices of Nv valid sample points, namely indices of those coarse voxels with sigma > 0
        ijk_coarse = torch.logical_and(self.sigma_voxels_coarse > 0, self.sigma_voxels_coarse != self.sigma_init).nonzero().squeeze(1)  # (Nv, 3)

        # indexing the valid sample points from 1 to Nv
        num_valid = ijk_coarse.shape[0] + 1                                                     # Nv + 1
        index = torch.arange(1, num_valid, dtype=torch.long, device=ijk_coarse.device)          # (Nv,), namely tensor([1, 2, ..., Nv])
        self.index_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] = index  # (Dc, Dc, Dc)

        # (Nv, Df, Df, Df) of (28,), the fine density Voxels Vf which stores the volume density, rgb color and so on
        self.voxels_fine = torch.zeros(num_valid, self.grid_fine, self.grid_fine, self.grid_fine, self.dim_sh+1, device=self.device)    # (Nv, Dc, Dc, Dc, 28)
        self.voxels_fine[...,  0] = self.sigma_default      # first dimension stores the volume density, initilize by sigma_default
        self.voxels_fine[..., 1:] = 0.0                     # remaining dimensions store the color rgb and so on

    def calc_index_fine(self, xyz):
        """ compute the fine grid coordinate(已知 coarse index 的情况下求 coarse voxel 内的 fine index) of each sampled point
        arguments:
            xyz - (N, 3), world coordinates of all the sampled points
        returns:
            index_fine - (N, 3), grid fine coordinates of all the sampled points inside the coarse voxels
        """
        xyz_norm = (xyz - self.xyz_min) / self.xyz_scope    # (N, 3)
        xyz_fine = (xyz_norm * self.res_fine).long()        # (N, 3)
        index_fine = xyz_fine % self.grid_fine              # (N, 3)
        return index_fine
        
    def update_fine(self, xyz, sigma, sh):
        ''' update the fine density Voxels Vf by simply assignment
        arguments:
            xyz   - (N, 3), world coordinates of all the sampled points
            sigma - (N, 1), densities that just predicted by the fine MLP of all the sampled points
            sh    - (N, F), 
        returns:
            self.voxels_fine - lateset updated fine density Voxels Vf
        '''
        # calculate `ijk_coarse` and use it to get the valid flag of Vc voxels corresponding to xyz
        index_coarse = self.query_coarse(xyz, 'index')                  # (N,) each belongs to the range(0, 1, ..., Nv)
        # 
        nonzero_index_coarse = torch.nonzero(index_coarse).squeeze(1)   # (Nv,), index of valid sampled point in xyz, xyz[i] valid if Vc[i]'s index_voxels_coarse > 0
        index_coarse = index_coarse[nonzero_index_coarse]

        # calculate `index_fine` of each valid sample points
        ijk_fine = self.calc_index_fine(xyz[nonzero_index_coarse])

        # concatenate the fine MLP predicted sigma and sh together
        feat = torch.cat([sigma, sh], dim=-1)

        self.voxels_fine[index_coarse, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]] = feat[nonzero_index_coarse]
    
    def query_coarse(self, xyz, type='sigma'):
        '''
            xyz: (N, 3)
        '''
        ijk_coarse = self.calc_index_coarse(xyz)

        if type == 'sigma':
            out = self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]    # (N,)
        else:
            out = self.index_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]    # (N,)
        return out

    def query_fine(self, xyz):
        '''
            x: (N, 3)
        '''
        # calc index_coarse
        index_coarse = self.query_coarse(xyz, 'index')

        # calc index_fine
        ijk_fine = self.calc_index_fine(xyz)

        return self.voxels_fine[index_coarse, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]]


class EfficientNeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(EfficientNeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        # instantiate the loss function
        self.loss = loss_dict[hparams.loss_type]()

        # instantiate the positional embedder for xyz and view directions
        self.embedding_xyz = Embedding(3, 10)       # 10 is the default number
        self.embedding_dir = Embedding(3, 4)        # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.deg = 2
        self.dim_sh = 3 * (self.deg + 1)**2

        # instantiate the coarse NeRF model and the fine NeRF model
        self.nerf_coarse = NeRF(D=4, W=128, in_channels_xyz=63, in_channels_dir=27, skips=[2], deg=self.deg)
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], deg=self.deg)
            self.models += [self.nerf_fine]

        # instantiate the NeRFTree
        self.sigma_init = hparams.sigma_init
        self.sigma_default = hparams.sigma_default
        coord_scope = hparams.coord_scope
        self.nerf_tree = NerfTree_Pytorch(xyz_min=[-coord_scope, -coord_scope, -coord_scope], 
                                          xyz_max=[coord_scope, coord_scope, coord_scope], 
                                          grid_coarse=384, 
                                          grid_fine=3,
                                          deg=self.deg, 
                                          sigma_init=self.sigma_init, 
                                          sigma_default=self.sigma_default,
                                          device='cuda')
        os.makedirs(f'logs/{self.hparams.exp_name}/ckpts', exist_ok=True)
        self.nerftree_path = os.path.join(f'logs/{self.hparams.exp_name}/ckpts', 'nerftree.pt')
        if self.hparams.ckpt_path != None and os.path.exists(self.nerftree_path):
            voxels_dict = torch.load(self.nerftree_path)
            self.nerf_tree.sigma_voxels_coarse = voxels_dict['sigma_voxels_coarse']
        
        self.xyz_min = self.nerf_tree.xyz_min
        self.xyz_max = self.nerf_tree.xyz_max
        self.xyz_scope = self.nerf_tree.xyz_scope
        self.grid_coarse = self.nerf_tree.grid_coarse
        self.grid_fine = self.nerf_tree.grid_fine
        self.res_coarse = self.nerf_tree.res_coarse
        self.res_fine = self.nerf_tree.res_fine
        
    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs
    
    def sigma2weights(self, deltas, sigmas):
        """ compute the weights from volume density using equation (2)
        arguments:
            deltas - (N_rays, N_samples_coarse), interval distance between every two coarse sampled points
            sigmas - (N_rays, N_samples_coarse), last step updated volume density stored in the coarse density Voxels Vc of each coarse sampled point
        returns:
            weights - (N_rays, N_samples_coarse), compute the alpha_i*T_i of each coarse sampled point
            alphas  - (N_rays, N_samples_coarse), compute the 1-exp(-sigma_i*delta_i) of each coarse sampled point
        """
        # add some noise to the volume density
        noise = torch.randn(sigmas.shape, device=sigmas.device)
        sigmas = sigmas + noise
        # use equation (2) to compute the weight using sigma(volume density)
        alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas))       # (N_rays, N_samples_coarse)
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1)    # [1, a1, a2, ...]
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]    # (N_rays, N_samples_coarse)
        return weights, alphas
    
    def render_rays(self, 
                models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                noise_std=0.0,
                N_importance=0,
                chunk=1024*32,
                white_back=False
                ):

        def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, idx_render):
            N_samples_ = xyz_.shape[1]
            # Embed directions
            xyz_ = xyz_[idx_render[:, 0], idx_render[:, 1]].view(-1, 3) # (N_rays*N_samples_, 3)
            view_dir = dir_.unsqueeze(1).expand(-1, N_samples_, -1)
            view_dir = view_dir[idx_render[:, 0], idx_render[:, 1]]
            # Perform model inference to get rgb and raw sigma
            B = xyz_.shape[0]
            out_chunks = []
            for i in range(0, B, chunk):
                out_chunks += [model(embedding_xyz(xyz_[i:i+chunk]), view_dir[i:i+chunk])]
            out = torch.cat(out_chunks, 0)
           
            out_rgb = torch.full((N_rays, N_samples_, 3), 1.0, device=device)
            out_sigma = torch.full((N_rays, N_samples_, 1), self.sigma_default, device=device)
            out_sh = torch.full((N_rays, N_samples_, self.dim_sh), 0.0, device=device)
            out_defaults = torch.cat([out_sigma, out_rgb, out_sh], dim=2)
            out_defaults[idx_render[:, 0], idx_render[:, 1]] = out
            out = out_defaults

            sigmas, rgbs, shs = torch.split(out, (1, 3, self.dim_sh), dim=-1)
            del out
            sigmas = sigmas.squeeze(-1)
                    
            # Convert these values using volume rendering (Section 4)
            deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
            delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
            deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
            
            weights, alphas = self.sigma2weights(deltas, sigmas)

            weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

            # compute final weighted outputs
            rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
            depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

            if white_back:
                rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

            return rgb_final, depth_final, weights, sigmas, shs
        
        #############################################################################
        # extract models, positional embedder and other params from the input lists #
        #############################################################################
        model_coarse = models[0]                # coarse NeRF model
        embedding_xyz = embeddings[0]           # positional embedder for xyz coordinate
        device = rays.device                    # traget device: one of ['cpu', 'gpu']
        is_training = model_coarse.training
        result = {}

        ############################################################################
        # decompose the raw inputs returned by dataloader to get rays_o and rays_d #
        ############################################################################
        N_rays = rays.shape[0]                          # number of rays per batch, default = 1024
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]     # both of (N_rays, 3) == (batch_size, 3) == (1024, 3)

        #########################
        # embed view directions #
        #########################
        dir_embedded = None

        #############################################################################
        # compute the coarse sample points coordinates using rays_o, rays_d, z_vals #
        #############################################################################
        N_samples_coarse = self.N_samples_coarse
        z_vals_coarse = self.z_vals_coarse.clone().expand(N_rays, -1)   # (N_rays, N_samples_coarse)
        # add some uniform perturb to the coarse sample points when training
        if is_training:
            delta_z_vals = torch.empty(N_rays, 1, device=device).uniform_(0.0, self.distance/N_samples_coarse)
            z_vals_coarse = z_vals_coarse + delta_z_vals                # (N_rays, N_samples_coarse)
        # compute the world coordinates of all the coarse sample points
        xyz_sampled_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_coarse.unsqueeze(2)         # (N_rays, N_samples_coarse, 3)
        # reshape the `xyz_sampled_coarse` to 2 dimensions 
        xyz_coarse = xyz_sampled_coarse.reshape(-1, 3)                  # (N_rays * N_samples_coarse, 3)

        ##########################################################################
        # perform valid sampling and compute the output of the coarse NeRF model #
        #*only perform this during the training epoch_{0} and epoch_{num_epochs} #
        ##########################################################################
        sigmas = self.nerf_tree.query_coarse(xyz_coarse, type='sigma').reshape(N_rays, N_samples_coarse)    # (N_rays, N_samples_coarse)
        # update density voxel during coarse training
        if is_training and self.nerf_tree.voxels_fine == None:
            #############################################################################
            # compute the indices of all the valid coarse sample points we're gonna use #
            #############################################################################
            with torch.no_grad():
                # introduce uniform sampling, not necessary, 这是直接随机把部分 rays 所有的 sample points 都变成 valid
                sigmas[torch.rand_like(sigmas[:, 0]) < self.hparams.uniform_ratio] = self.sigma_init 
                # generate the indices of the valid sample points
                if self.hparams.warmup_step > 0 and self.trainer.global_step <= self.hparams.warmup_step:
                    # during warmup, treat all points as valid samples
                    idx_render_coarse = torch.nonzero(sigmas >= -1e10).detach()     # (Nv, 2)
                else:
                    # or else, treat points whose density > 0 as valid samples
                    idx_render_coarse = torch.nonzero(sigmas > 0.0).detach()        # (Nv, 2)
            #################################################
            # compute the output of the coarse NeRF network #
            #################################################
            rgb_coarse, depth_coarse, weights_coarse, sigmas_coarse, _ = \
                inference(model_coarse, embedding_xyz, xyz_sampled_coarse, rays_d, dir_embedded, z_vals_coarse, idx_render_coarse)
            # add coarse NeRF's result to the return dict()
            result['num_samples_coarse'] = torch.FloatTensor([idx_render_coarse.shape[0] / N_rays])   
            result['rgb_coarse']         = rgb_coarse
            result['z_vals_coarse']      = self.z_vals_coarse
            result['depth_coarse']       = depth_coarse
            result['sigma_coarse']       = sigmas_coarse
            result['weight_coarse']      = weights_coarse
            result['opacity_coarse']     = weights_coarse.sum(1)
            #################################################################################################    
            # update the valid coarse density Voxels Vc using the density just predicted by the coarse NeRF #
            #################################################################################################
            xyz_coarse_ = xyz_sampled_coarse[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]          # (Nv, 3)
            sigmas_coarse_ = sigmas_coarse.detach()[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]   # (Nv,)
            self.nerf_tree.update_coarse(xyz_coarse_, sigmas_coarse_, self.hparams.beta)
        
        ####################################################################################################
        # re-compute the z value's distance between every two coarse sample points and compute the weights #
        ####################################################################################################
        with torch.no_grad():
            # 一开始 prepare_data() 里面算的 deltas 其实没啥用，因为对于 training 来说是加了 perturb 的
            deltas_coarse = z_vals_coarse[:, 1:] - z_vals_coarse[:, :-1]    # (N_rays, N_samples_coarse-1)
            delta_inf = 1e10 * torch.ones_like(deltas_coarse[:, :1])        # (N_rays, 1) the last delta is infinity
            deltas_coarse = torch.cat([deltas_coarse, delta_inf], -1)       # (N_rays, N_samples_coarse)
            # 这里用的 sigmas 是从上一 step 的 coarse density Voxels Vc 的结果，而不是当前 step 更新过后的结果
            weights_coarse, _ = self.sigma2weights(deltas_coarse, sigmas)   # (N_rays, N_samples_coarse) == (1024, 128)
            weights_coarse = weights_coarse.detach()

        ###############################################################################################
        # find pivotal sample points(indice) with weight > epsilon among all the coarse sample points #
        ###############################################################################################
        idx_render = torch.nonzero(weights_coarse >= min(self.hparams.weight_threashold, weights_coarse.max().item()))  # (Np, 2), Np \in [0, N_rays * N_samples_coarse]
        scale = N_importance        # default = 128, sample N_importance fine points near each pivotal points
        #########################################################################################
        # compute the specific indices of fine sample points near all the pivotal sample points #
        #########################################################################################
        z_vals_fine = self.z_vals_fine.clone()                          # (1, N_samples_coarse * N_importance) == (1, 128*5)
        # z_vals_fine 是 prepare_data() 的时候就预先准备好的，假设 each ray 上的所有 coarse sample points 都是 pivotal 后采样的 fine points 的 z values
        if is_training: z_vals_fine = z_vals_fine + delta_z_vals        # (N_rays, N_samples_coarse * N_importance) == (1024, 128*5)
        # find the valid pivotal fine sample points 
        idx_render = idx_render.unsqueeze(1).expand(-1, scale, -1)      # (Np, N_importance, 2), Np ∈ [0, 1024*128]
        # idx_render_fine[..., 0] represents the index of all the rays, idx_render_fine[..., 1] represents the index of all the coarse sample points along one ray
        idx_render_fine = idx_render.clone()                            # (Np, N_importance, 2), Np ∈ [0, 1024*128]
        # 编号是按照 coarse_{0} 的 [0, ..., scale-1], coarse_{1} 的 [scale, 2*scale-1], ..., 以及 coarse_{N_samples} 的 [(N_samples-1)*scale, N_samples*scale-1]
        idx_render_fine[..., 1] = idx_render[..., 1] * scale + (torch.arange(scale, device=device)).reshape(1, scale)
        idx_render_fine = idx_render_fine.reshape(-1, 2)                # (Np * N_importance, 2), Np*N_importance ∈ [0, 1024*128*5]
        # sample maximum N_rays * 64 == 1024 * 64 points at the fine stage
        if idx_render_fine.shape[0] > N_rays * 64:
            indices = torch.randperm(idx_render_fine.shape[0])[:N_rays * 64]
            idx_render_fine = idx_render_fine[indices]
        ###############################################################
        # compute the world coordinates of all the fine sample points #
        ###############################################################
        xyz_sampled_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine.unsqueeze(2)     # (N_rays, N_samples*N_importance, 3)
        #############################################
        # compute the output of the fine NeRF model #
        #############################################
        model_fine = models[1]
        rgb_fine, depth_fine, _, sigmas_fine, shs_fine = \
            inference(model_fine, embedding_xyz, xyz_sampled_fine, rays_d, dir_embedded, z_vals_fine, idx_render_fine)
        #############################################################################################    
        # update the valid fine density Voxels Vf using the results just predicted by the fine NeRF #
        #*only perform this update during the last epoch, namely epoch_{num_epoch} for caching use. #
        #############################################################################################
        if is_training and self.nerf_tree.voxels_fine != None:
            with torch.no_grad():
                xyz_fine_ = xyz_sampled_fine[idx_render_fine[:, 0], idx_render_fine[:, 1]]
                sigmas_fine_ = sigmas_fine.detach()[idx_render_fine[:, 0], idx_render_fine[:, 1]].unsqueeze(-1)
                shs_fine_ = shs_fine.detach()[idx_render_fine[:, 0], idx_render_fine[:, 1]]
                self.nerf_tree.update_fine(xyz_fine_, sigmas_fine_, shs_fine_)

        # add fine NeRF's result to the return dict()
        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['num_samples_fine'] = torch.FloatTensor([idx_render_fine.shape[0] / N_rays])

        return result

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        # if self.nerf_tree.voxels_fine == None or self.models[0].training:
        #     chunk = self.hparams.chunk
        # else:
        #     chunk = B // 8
        chunk = self.hparams.chunk
        for i in range(0, B, chunk):
            rendered_ray_chunks = \
                self.render_rays(self.models,
                            self.embeddings,
                            rays[i:i+chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back
                                )
                            
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results
    
    def optimizer_step(self, epoch=None, 
                    batch_idx=None, 
                    optimizer=None, 
                    optimizer_idx=None, 
                    optimizer_closure=None, 
                    on_tpu=None, 
                    using_native_amp=None, 
                    using_lbfgs=None):
        if self.hparams.warmup_step > 0 and self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.warmup_step))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def prepare_data(self):
        """ https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=prepare_data#prepare-data
            called right after the __init__() function? in my comprehension
        """
        ####################################################################
        # create the specifiy training dataset and validation/test dataset #
        ####################################################################
        dataset = dataset_dict[self.hparams.dataset_name]   # default = BlenderDataset
        # prepare kwargs for instantiating the dataset object
        kwargs = {'root_dir': self.hparams.root_dir, 'img_wh': tuple(self.hparams.img_wh)}
        # create the training dataset
        self.train_dataset = dataset(split='train', **kwargs)
        # create the validation or test(for blender type) dataset
        if self.hparams.dataset_name == 'blender': self.val_dataset = dataset(split='test', **kwargs)
        else: self.val_dataset = dataset(split='val', **kwargs)
        
        ###########################################################################
        # fetch the common bounds for all scenes in blender(or other data types?) #
        ###########################################################################
        self.near = self.train_dataset.near                 # default = 2.0
        self.far = self.train_dataset.far                   # default = 6.0
        self.distance = self.far - self.near                # default = 4.0
        # transfer the float bounds to torch.Tensor()
        near = torch.full((1,), self.near, dtype=torch.float32, device='cuda')      # (1,), namely tensor([2.0,])
        far = torch.full((1,), self.far, dtype=torch.float32, device='cuda')        # (1,), namely tensor([6.0,])

        ###############################################################################
        # generate the z values of all the linear coarse sample points along each ray #
        ###############################################################################
        self.N_samples_coarse = self.hparams.N_samples                              # default = 128
        # compute z values of all the coarse sample points linearly
        z_vals_coarse = torch.linspace(0, 1, self.N_samples_coarse, device='cuda')  # (N_samples_coarse,)
        if not self.hparams.use_disp:
            # use linear sampling in depth space
            z_vals_coarse = near * (1-z_vals_coarse) + far * z_vals_coarse          # (N_samples_coarse,)
        else:
            # use linear sampling in disparity space
            z_vals_coarse = 1/(1/near * (1-z_vals_coarse) + 1/far * z_vals_coarse)  # (N_samples_coarse,)
        # add first dimension to the `z_vals_coarse`
        self.z_vals_coarse = z_vals_coarse.unsqueeze(0)     # (1, N_samples_coarse)

        #############################################################################
        # generate the z values of all the linear fine sample points along each ray #
        #############################################################################
        self.N_samples_fine = self.hparams.N_samples * self.hparams.N_importance    # default = 128 * 5 = 640
        # compute z values of all the fine sample points linearly
        z_vals_fine = torch.linspace(0, 1, self.N_samples_fine, device='cuda')      # (N_samples_coarse,)
        if not self.hparams.use_disp:
            # use linear sampling in depth space
            z_vals_fine = near * (1-z_vals_fine) + far * z_vals_fine                # (N_samples_coarse,)
        else:
            # use linear sampling in disparity space
            z_vals_fine = 1/(1/near * (1-z_vals_fine) + 1/far * z_vals_fine)        # (N_samples_coarse,)
        # add first dimension to the `z_vals_coarse`
        self.z_vals_fine = z_vals_fine.unsqueeze(0)         # (1, N_samples_coarse)

        ################################################################################################
        # compute the z value's distance between every two sample points, coarse and fine respectively #
        ################################################################################################
        # distance for coarse sample points
        deltas = self.z_vals_coarse[:, 1:] - self.z_vals_coarse[:, :-1]     # (1, N_samples_coarse-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])                   # (1, 1) the last delta is infinity
        self.deltas_coarse = torch.cat([deltas, delta_inf], -1)             # (1, N_samples_coarse)
        # distance for fine sample points
        deltas = self.z_vals_fine[:, 1:] - self.z_vals_fine[:, :-1]     # (1, N_samples_fine-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])               # (1, 1) the last delta is infinity
        self.deltas_fine = torch.cat([deltas, delta_inf], -1)           # (1, N_samples_fine)
        

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams.batch_size,   # default = 1024
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,                         # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_idx):
        self.log('train/lr', get_learning_rate(self.optimizer), on_step=True, prog_bar=True)
        rays, rgbs = self.decode_batch(batch)

        # create the fine density Voxels Vf only at the last epoch_{num_epoch}
        extract_time = self.current_epoch >= (self.hparams.num_epochs - 1)
        if extract_time and self.nerf_tree.voxels_fine == None:
            self.nerf_tree.create_voxels_fine()
    
        results = self(rays)

        loss_total = loss_rgb = self.loss(results, rgbs)
        self.log('train/loss_rgb', loss_rgb, on_step=True)

        # if self.hparams.weight_tv > 0.0:
        #     alphas_coarse = results['alpha_coarse']
        #     loss_tv = self.hparams.weight_tv * (alphas_coarse[:, 1:] - alphas_coarse[:, :-1]).pow(2).mean()
        #     self.log('train/loss_tv', loss_tv, on_step=True)
        #     loss_total += loss_tv

        self.log('train/loss_total', loss_total, on_step=True)

        if 'num_samples_coarse' in results:
            self.log('train/num_samples_coarse', results['num_samples_coarse'].mean(), on_step=True)

        if 'num_samples_fine' in results:
            self.log('train/num_samples_fine', results['num_samples_fine'].mean(), on_step=True)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_idx % 1000 == 0 and self.nerf_tree.voxels_fine == None:
            fig = plt.figure()
            depths = results['z_vals_coarse'][0].detach().cpu().numpy()
            sigmas = torch.nn.ReLU()(results['sigma_coarse'][0]).detach().cpu().numpy()
            weights = results['weight_coarse'][0].detach().cpu().numpy()
            near = self.near - (self.far - self.near) * 0.1
            far = self.far + (self.far - self.near) * 0.1
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
            ax[0].scatter(x=depths, y=sigmas)
            ax[0].set_xlabel('Depth', fontsize=16)
            ax[0].set_ylabel('Density', fontsize=16)
            ax[0].set_title('Density Distribution of a Ray', fontsize=16)
            ax[0].set_xlim([near, far])

            ax[1].scatter(x=depths, y=weights)
            ax[1].set_xlabel('Depth', fontsize=16)
            ax[1].set_ylabel('Weight', fontsize=16)
            ax[1].set_title('Weight Distribution of a Ray', fontsize=16)
            ax[1].set_xlim([near, far])

            self.logger.experiment.add_figure('train/distribution',
                                               fig, self.global_step)
            plt.close()

        feats = {}
        with torch.no_grad():
            psnr_fine = psnr(results[f'rgb_{typ}'], rgbs)
            self.log('train/psnr_fine', psnr_fine, on_step=True, prog_bar=True)

            if 'rgb_coarse' in results:
                psnr_coarse = psnr(results['rgb_coarse'], rgbs)
                self.log('train/psnr_coarse', psnr_coarse, on_step=True)

        if batch_idx % 1000 == 0:
            torch.cuda.empty_cache()
        return loss_total

    def validation_step(self, batch, batch_idx):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)

        results = self(rays)
        log = {}
        log['val_loss'] = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        
        W, H = self.hparams.img_wh
        img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
        img = img.permute(2, 0, 1) # (3, H, W)
        img_path = os.path.join(f'logs/{hparams.exp_name}/video', "%06d.png" % batch_idx)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        transforms.ToPILImage()(img).convert("RGB").save(img_path)
        
        idx_selected = 0
        if batch_idx == idx_selected:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            stack = torch.stack([img_gt, img]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/gt_pred',
                                               stack, self.global_step)
            
            img_path = os.path.join(f'logs/{hparams.exp_name}', f'epoch_{self.current_epoch}.png')
            transforms.ToPILImage()(img).convert("RGB").save(img_path)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        torch.cuda.empty_cache()
        return log

    def validation_epoch_end(self, outputs):
        log = {}
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        num_voxels_coarse = torch.logical_and(self.nerf_tree.sigma_voxels_coarse > 0, self.nerf_tree.sigma_voxels_coarse != self.sigma_init).nonzero().shape[0]
        self.log('val/loss', mean_loss, on_epoch=True)
        self.log('val/psnr', mean_psnr, on_epoch=True, prog_bar=True)
        self.log('val/num_voxels_coarse', num_voxels_coarse, on_epoch=True)

        # save sparse voxels
        sigma_voxels_coarse_clean = self.nerf_tree.sigma_voxels_coarse.clone()
        sigma_voxels_coarse_clean[sigma_voxels_coarse_clean == self.sigma_init] = self.sigma_default
        voxels_dict = {
            'sigma_voxels_coarse': sigma_voxels_coarse_clean,
            'index_voxels_coarse': self.nerf_tree.index_voxels_coarse,
            'voxels_fine': self.nerf_tree.voxels_fine
        }
        torch.save(voxels_dict, self.nerftree_path)

        img_paths = glob.glob(f'logs/{hparams.exp_name}/video/*.png')
        writer = imageio.get_writer(f'logs/{hparams.exp_name}/video/video_{self.current_epoch}.mp4', fps=40)
        for im in img_paths:
            writer.append_data(imageio.imread(im))
        writer.close()


if __name__ == '__main__':
    hparams = get_opts()
    system = EfficientNeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'logs/{hparams.exp_name}/ckpts', '{epoch:d}'),
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=5,)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      gpus=hparams.num_gpus,
                      strategy='ddp' if hparams.num_gpus>1 else None,
                      benchmark=True)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)