import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .blocks import *
from torch.distributions import Normal, Independent, kl
from utils import *

class IEncoder(nn.Module):
    def __init__(self, n_channels, device, filters=None, training=True):
        super(IEncoder, self).__init__()
        self.n_channels = n_channels
        self.training = training
        self.device = device
        if filters is None:
            filters = [32, 64, 128, 192, 192]
        
        
        self.init_conv = nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1)
        self.down = TransitionDown()
#        # 32 * 128 * 128
#        self.block1 = DenseBlock(filters[0], filters[0], self.training)
#        # 64 * 64 * 64
#        self.block2 = DenseBlock(filters[0], filters[1], self.training)
#        # 128 * 32 * 32
#        self.block3 = DenseBlock(filters[1], filters[2], self.training)
#        # 192 * 16 * 16
#        self.block4 = DenseBlock(filters[2], filters[3], self.training)
#        # 192 * 8 * 8
#        self.block5 = DenseBlock(filters[3], filters[4], self.training)
        
        self.blocks = nn.ModuleList([
                DenseBlock(filters[0], filters[0], self.training)] + [
                DenseBlock(filters[i], filters[i+1], self.training) for i in range(len(filters)-1)
            ])
        

    def forward(self, x):
        x = self.init_conv(x)
        x_f = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            x_f.append(x)
            x = self.down(x)
        return x_f
            
#        x1 = self.block1(x)
#        x2 = self.block2(self.down(x1))
#        x3 = self.block3(self.down(x2))
#        x4 = self.block4(self.down(x3))
#        x5 = self.block5(self.down(x4))
        #return x1, x2, x3, x4, x5

class MEncoder(nn.Module):
    def __init__(self, n_channels, device, filters=None, training=True, fuse=False):
        super(MEncoder, self).__init__()
        self.n_channels = n_channels
        self.training = training
        self.fuse = fuse
        self.device = device
        if filters is None:
            filters = [32, 64, 128, 192, 192]
        
        self.init_conv = nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1)
        self.down = TransitionDown()
#        # 32 * 128 * 128
#        self.block1 = DenseBlock(filters[0], filters[0], self.training)
#        # 64 * 64 * 64
#        self.block2 = DenseBlock(filters[0], filters[1], self.training)
#        # 128 * 32 * 32
#        if self.fuse:
#            self.block3 = DenseBlock(2 * filters[1], filters[2], self.training)
#        else:
#            self.block3 = DenseBlock(filters[1], filters[2], self.training)
#        # 192 * 16 * 16
#        self.block4 = DenseBlock(filters[2], filters[3], self.training)
#        # 192 * 8 * 8
#        self.block5 = DenseBlock(filters[3], filters[4], self.training)

        self.blocks = nn.ModuleList([
                DenseBlock(filters[0], filters[0], self.training)] + [
                DenseBlock(filters[i], filters[i+1], self.training) for i in range(len(filters)-1)
            ])
        if fuse:
            self.blocks[2] = DenseBlock(2 * filters[1], filters[2], self.training)
        

    def forward(self, x, x_=None):
        x = self.init_conv(x)
        x_f = []
        for i in range(len(self.blocks)):
            if i == 2 and self.fuse:
                x = torch.cat([x, x_], dim=1)
            x = self.blocks[i](x)
            x_f.append(x)
            x = self.down(x)
        return x_f
        
#        x = self.init_conv(x)
#        x1 = self.block1(x)
#        x2 = self.block2(self.down(x1))
#        x3 = self.block3(self.down(x2))
#        if self.fuse:
#            x4 = self.block4(self.down(torch.cat([x3, x_], dim=1)))
#        else:
#            x4 = self.block4(self.down(x3))
#        x5 = self.block5(self.down(x4))
#        return x1, x2, x3, x4, x5

class SampleLatent(nn.Module):
    def __init__(self, device, filters, n_sample=1, latent_dim=8, share=False,
                       image_size=128, flow_levels=None):
        super(SampleLatent, self).__init__()
        Finnal_dim = filters[-1]
        self.Finnal_dim = Finnal_dim
        self.filters = filters
        self.latent_dim = latent_dim
        self.n_sample = n_sample
        self.share = share
        self.device = device
        
        self.image_size = image_size
        flow_levels = list(range(len(filters))) if flow_levels is None else flow_levels
        assert type(flow_levels) == list
        self.flow_levels = flow_levels

        self.resblock_x = nn.Sequential(
            nn.BatchNorm2d(Finnal_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            ResBlock(in_channels=Finnal_dim, out_channels=Finnal_dim, mid_channels=Finnal_dim//2,
                     post=False)
        )
        self.latent_x = self.latent_conv(Finnal_dim, 2 * latent_dim)

        self.resblock_m = nn.Sequential(
            nn.BatchNorm2d(Finnal_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            ResBlock(in_channels=Finnal_dim, out_channels=Finnal_dim, mid_channels=Finnal_dim//2,
                     post=False)
        )
        self.latent_m = self.latent_conv(Finnal_dim, 2 * latent_dim)
        
        size = self.image_size // (2 ** (len(filters) - 1))
        self.norm_conv = self.dist_norm_conv(2 * latent_dim)
        self.norm_conv_init = nn.Parameter(torch.randn(1, filters[-1], size, size))
        
    def latent_conv(self, Finnal_dim, out_channels):
        module = nn.ModuleList()
        sub_module = nn.Sequential(
                            SingleDenseBlock(Finnal_dim, out_channels, kernel_size=1, padding=0),
                            nn.AdaptiveAvgPool2d((1, 1))
                     )
        module.append(sub_module)
        for i in reversed(range(len(self.filters))):
            sub_module = nn.ModuleList([SingleDenseBlock(self.filters[i] + self.latent_dim, 
                                        self.filters[i])] + [nn.Sequential(
                            SingleDenseBlock(self.filters[i], out_channels, kernel_size=1, padding=0),
                            nn.AdaptiveAvgPool2d((1, 1))
                        )])
            module.append(sub_module)
        return module
    
    def dist_norm_conv(self, out_channels):
        module = nn.ModuleList()
        for i in reversed(range(1, len(self.filters))):
            sub_module = nn.ModuleList([SingleDenseBlock(self.filters[i] + self.latent_dim, 
                                        self.filters[i-1])] + [nn.Sequential(
                            SingleDenseBlock(self.filters[i-1], 
                                             out_channels, kernel_size=1, padding=0),
                            nn.AdaptiveAvgPool2d((1, 1))
                        )])
            module.append(sub_module)
        return module

    def sample_dist(self, dist, mode, PlanarFlow=None, InvertFlow=None):
        assert type(dist) == list
        assert len(dist) == len(self.filters)
        assert not (PlanarFlow is not None and InvertFlow is not None)
        samples_hierarchy = []
        zk = []
        log_det_hierarchy = []
        for i in range(len(dist)):
            level = len(dist) - i - 1
            size = self.image_size // (2**(level))
            if (PlanarFlow is None and InvertFlow is None) or (level not in self.flow_levels):
                samples = sample_latent(self.device, dist[level], self.latent_dim, 1, size, mode)
                zk.insert(0, -1)
                log_det_hierarchy.insert(0, -1)
            elif PlanarFlow is not None and InvertFlow is None:
                samples, log_det = PlanarFlow(dist[level], mode, level)
                zk.insert(0, samples)
                log_det_hierarchy.insert(0, log_det)
                samples = tile(self.device, samples, 2, size)
                samples = tile(self.device, samples, 3, size)
            elif PlanarFlow is None and InvertFlow is not None:
                samples = InvertFlow.sample(dist[level], mode, level)
                samples = tile(self.device, samples, 2, size)
                samples = tile(self.device, samples, 3, size)
            samples_hierarchy.insert(0, samples)
        return samples_hierarchy, zk, log_det_hierarchy

    def sample_dist_infer(self, dist, mode, n_samples=32, PlanarFlow=None, InvertFlow=None):
        assert type(dist) == list
        assert len(dist) == len(self.filters)
        assert not (PlanarFlow is not None and InvertFlow is not None)
        def _sample_latent(device, dist, latent_dim, n_sample, size, mode):
            if mode == 'rsample':
                samples = dist.rsample([n_sample])
            elif mode == 'sample':
                samples = dist.sample([n_sample])
            samples = samples.reshape(n_sample, latent_dim, samples.size(3), samples.size(4))
            
            samples = tile(device, samples, 2, size)
            samples = tile(device, samples, 3, size)
            
            return samples
        samples_hierarchy = []
        for i in range(len(dist)):
            level = len(dist) - i - 1
            size = self.image_size // (2**(level))
            if (PlanarFlow is None and InvertFlow is None) or (level not in self.flow_levels):
                samples = _sample_latent(self.device, dist[level], self.latent_dim, n_samples,
                                         size, mode)
            elif PlanarFlow is None and InvertFlow is not None:
                samples = InvertFlow.sample(dist[level], mode, level, n_samples=n_samples)
                samples = tile(self.device, samples, 2, size)
                samples = tile(self.device, samples, 3, size)
            samples_hierarchy.insert(0, samples)
        return samples_hierarchy
        
    def get_dist(self, latent):
        mu = latent[:, :self.latent_dim, ...]
        log_sigma = latent[:, self.latent_dim:, ...]
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),3)
        return dist
    def get_distributions(self, x, resblock, latent_conv, mode, PlanarFlow=None,
                          InvertFlow=None):
        assert type(x) == list
        assert (len(x) + 1) == len(self.latent_x) == len(self.latent_m)
        assert not (PlanarFlow is not None and InvertFlow is not None)
        resout = resblock(x[-1])
        dist = []
        latent = latent_conv[0](resout)
        for i in range(1, len(latent_conv)):
            dist.insert(0, self.get_dist(latent))
            level = len(x) - i
            size = self.image_size // (2**(level))
            if (PlanarFlow is None and InvertFlow is None) or (level not in self.flow_levels):
                samples = sample_latent(self.device, dist[0], self.latent_dim, 1, size, mode)
            elif PlanarFlow is not None and InvertFlow is None:
                samples, _ = PlanarFlow(dist[0], mode, level)
                samples = tile(self.device, samples, 2, size)
                samples = tile(self.device, samples, 3, size)
            elif PlanarFlow is None and InvertFlow is not None:
                samples = InvertFlow.sample(dist[0], mode, level)
                samples = tile(self.device, samples, 2, size)
                samples = tile(self.device, samples, 3, size)
            feature = latent_conv[i][0](torch.cat([x[level], samples], dim=1))
            latent = latent_conv[i][1](feature)
        return dist, resout
    def get_normal_distributions(self, x, norm_conv, norm_conv_init, mode):
        B, C, H, W = x.size()
        mu = torch.zeros(B, self.latent_dim, 1, 1, dtype=x.dtype).to(self.device)
        sigma = torch.log(torch.ones(B, self.latent_dim, 1, 1, dtype=x.dtype)).to(self.device)
        dist = []
        latent = torch.cat([mu, sigma], dim=1)
        feature = norm_conv_init
        feature = feature.repeat(B, 1, 1, 1)
        for i in range(len(norm_conv)):
            dist.insert(0, self.get_dist(latent))
            level = len(norm_conv) - i - 1
            size = self.image_size // (2**(level))
            samples = sample_latent(self.device, dist[0], self.latent_dim, 1, size, mode)
            feature = F.interpolate(feature, size=(size, size), mode='bilinear') 
            feature = norm_conv[i][0](torch.cat([feature, samples], dim=1))
            latent = norm_conv[i][1](feature)
        dist.insert(0, self.get_dist(latent))
        return dist
    
    def forward(self, x, partten, mode, PlanarFlow=None, InvertFlow=None):
        if partten == 'I' or self.share:
            return self.get_distributions(x, self.resblock_x, self.latent_x, mode,
                                          InvertFlow=InvertFlow)
        elif partten == 'M':
            return self.get_distributions(x, self.resblock_m, self.latent_m, mode,
                                          PlanarFlow=PlanarFlow)
        elif partten == 'N':
            return self.get_normal_distributions(x, self.norm_conv, self.norm_conv_init, mode)

class PlanarFlows(nn.Module):
    def __init__(self, device, flow_length=4, latent_dim=8, hierarchy=5):
        super(PlanarFlows, self).__init__()
        self.device = device
        self.flow_length = flow_length
        self.latent_dim = latent_dim
        self.hierarchy = hierarchy
        
        flows = [nn.Sequential(*[PlanarTransform(latent_dim) for _ in range(flow_length)])
                 for _ in range(hierarchy)]
        self.flows = nn.ModuleList(flows)
        
    def forward(self, base_dist, mode, level):
        flow = self.flows[level]
        z = None
        if mode == 'rsample':
            z = base_dist.rsample()
        elif mode == 'sample':
            z = base_dist.sample()
        # pass through flow:
        # 1. compute expected log_prob of data under base dist 
        #    -- nothing tied to parameters here so irrelevant to grads
        base_log_prob = base_dist.log_prob(z)
        # 2. compute sum of log_abs_det_jacobian through the flow
        z = z.view(-1, self.latent_dim)  # (batch, latent_dim)
        zk, sum_log_abs_det_jacobians = flow(z)
        # 3. compute expected log_prob of z_k the target dist
        #p_log_prob = target_dist.log_prob(zk)
        
        #kl_loss = base_log_prob - sum_log_abs_det_jacobians - p_log_prob
        
        zk = zk.view(-1, self.latent_dim, 1, 1)
        
        return zk, base_log_prob - sum_log_abs_det_jacobians
    
#    def kl_loss_zk(self, zk, log_det, target_dist, InvertFlow=None):
#        assert len(zk) == len(log_det) == len(target_dist) == self.hierarchy
#        hierarchy_loss = []
#        for i in range(len(zk)):
#            if InvertFlow is None:
#                p_log_prob = target_dist[i].log_prob(zk[i])
#            else:
#                _, p_log_prob = InvertFlow(zk[i], target_dist[i], i)
#            loss = log_det[i] - p_log_prob
#            hierarchy_loss.append(loss)
#        return torch.stack(hierarchy_loss).mean()
            

#    def kl_loss_dist(self, base_dist, target_dist, mode):
#        assert type(base_dist) == type(target_dist) == list
#        assert len(base_dist) == len(target_dist) == self.hierarchy
#        hierarchy_zk = []
#        hierarchy_loss = []
#        for i in range(len(base_dist)):
#            flow = self.flows[i]
#            z = None
#            if mode == 'rsample':
#                z = base_dist[i].rsample()
#            elif mode == 'sample':
#                z = base_dist[i].sample()
#            # pass through flow:
#            # 1. compute expected log_prob of data under base dist 
#            #    -- nothing tied to parameters here so irrelevant to grads
#            base_log_prob = base_dist[i].log_prob(z)
#            # 2. compute sum of log_abs_det_jacobian through the flow
#            z = z.view(-1, self.latent_dim)  # (batch, latent_dim)
#            zk, sum_log_abs_det_jacobians = flow(z)
#            # 3. compute expected log_prob of z_k the target dist
#            zk = zk.view(-1, self.latent_dim, 1, 1)
#            p_log_prob = target_dist[i].log_prob(zk)
#            
#            loss = base_log_prob - sum_log_abs_det_jacobians - p_log_prob
#            hierarchy_zk.append(zk)
#            hierarchy_loss.append(loss)
#        return hierarchy_zk, torch.stack(hierarchy_loss).mean()

class NICEFlows(nn.Module):
    def __init__(self, device, data_dim, hierarchy=5, hidden_dim=30, hidden_layer=3,
                 num_coupling_layers=4):
        super().__init__()

        self.data_dim = data_dim
        self.device = device
        
        self.hierarchy = hierarchy

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(data_dim, orientation=(i % 2 == 0))
                                                for i in range(num_coupling_layers)]
    
        coupling_layers = [nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                    hidden_dim=hidden_dim,
                                    mask=masks[i], num_layers=hidden_layer)
                                  for i in range(num_coupling_layers)])
                           for _ in range(hierarchy)
                           ]
        self.coupling_layers = nn.ModuleList(coupling_layers)
    
        scaling_layer = [ScalingLayer(data_dim=data_dim) for _ in range(hierarchy)]
        self.scaling_layer = nn.ModuleList(scaling_layer)

    def forward(self, base_dist, mode, level):
        z = None
        if mode == 'rsample':
            z = base_dist.rsample()
        elif mode == 'sample':
            z = base_dist.sample()
        
        base_log_prob = base_dist.log_prob(z)

        z = z.view(-1, self.data_dim)  # (batch, latent_dim)
        zk, sum_log_abs_det_jacobians = self.f(z, level)

        zk = zk.view(-1, self.data_dim, 1, 1)
        
        return zk, base_log_prob - sum_log_abs_det_jacobians

    def f(self, x, level):
        coupling_layers = self.coupling_layers[level]
        scaling_layer = self.scaling_layer[level]
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        z, log_det_jacobian = scaling_layer(z, log_det_jacobian)
        return z, log_det_jacobian

#    def kl_loss_zk(self, zk, log_det, target_dist, InvertFlow=None):
#        assert len(zk) == len(log_det) == len(target_dist) == self.hierarchy
#        hierarchy_loss = []
#        for i in range(len(zk)):
#            if InvertFlow is None:
#                p_log_prob = target_dist[i].log_prob(zk[i])
#            else:
#                _, p_log_prob = InvertFlow(zk[i], target_dist[i], i)
#            loss = log_det[i] - p_log_prob
#            hierarchy_loss.append(loss)
#        return torch.stack(hierarchy_loss).mean()

    def _get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask     # flip mask orientation
        mask = torch.tensor(mask).to(self.device)
        return mask.float()
        

class InvertFlows(nn.Module):
    def __init__(self, device, data_dim, hierarchy=5, hidden_dim=30, hidden_layer=3,
                 num_coupling_layers=4):
        super().__init__()

        self.data_dim = data_dim
        self.device = device

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(data_dim, orientation=(i % 2 == 0))
                                                for i in range(num_coupling_layers)]
    
        coupling_layers = [nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                    hidden_dim=hidden_dim,
                                    mask=masks[i], num_layers=hidden_layer)
                                  for i in range(num_coupling_layers)])
                           for _ in range(hierarchy)
                           ]
        self.coupling_layers = nn.ModuleList(coupling_layers)
    
        scaling_layer = [ScalingLayer(data_dim=data_dim) for _ in range(hierarchy)]
        self.scaling_layer = nn.ModuleList(scaling_layer)
    

    def forward(self, x, prior, level, invert=False):
        if not invert:
            x = x.view(-1, self.data_dim)
            z, log_det_jacobian = self.f(x, level)
            z = z.view(-1, self.data_dim, 1, 1)
            log_likelihood = prior.log_prob(z) + log_det_jacobian
            return z, log_likelihood
    
        zk = self.f_inverse(z, level)
        zk = zk.view(-1, self.data_dim, 1, 1)
        return zk

    def f(self, x, level):
        coupling_layers = self.coupling_layers[level]
        scaling_layer = self.scaling_layer[level]
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        z, log_det_jacobian = scaling_layer(z, log_det_jacobian)
        return z, log_det_jacobian

    def f_inverse(self, z, level):
        coupling_layers = self.coupling_layers[level]
        scaling_layer = self.scaling_layer[level]
        x = z
        x, _ = scaling_layer(x, 0, invert=True)
        for i, coupling_layer in reversed(list(enumerate(coupling_layers))):
            x, _ = coupling_layer(x, 0, invert=True)
        return x

    def sample(self, prior, mode, level, n_samples=1):
        #z = self.prior.sample([num_samples, self.data_dim]).view(self.samples, self.data_dim)
        z = None
        if mode == 'rsample':
            z = prior.rsample([n_samples])
        elif mode == 'sample':
            z = prior.sample([n_samples])
        z = z.view(-1, self.data_dim)
        
        zk = self.f_inverse(z, level)
        zk = zk.view(-1, self.data_dim, 1, 1)
        return zk

    def _get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask     # flip mask orientation
        mask = torch.tensor(mask).to(self.device)
        return mask.float()
        
            

class IDecoder(nn.Module):
    def __init__(self, device, latent_dim, filters=None, training=True,
                       image_size=128, add_dropout=False):
        super(IDecoder, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.training = training
        if filters is None:
            filters = [32, 64, 128, 192, 192]
        self.filters = filters
        self.image_size = image_size
        
        self.up = TransitionUp()
#        # (192 + latent_dim) * 8 * 8  -->  (192) * 8 * 8
#        self.block5 = DenseBlock(filters[4] + latent_dim, filters[3], self.training)
#        # (192 + latent_dim) * 16 * 16  -->  (128) * 16 * 16
#        self.block4 = DenseBlock(filters[3] + latent_dim, filters[2], self.training)
#        # (128 + latent_dim) * 32 * 32  -->  (64) * 32 * 32
#        self.block3 = DenseBlock(filters[2] + latent_dim, filters[1], self.training)
#        # (64 + latent_dim) * 64 * 64  -->  (32) * 64 * 64
#        self.block2 = DenseBlock(filters[1] + latent_dim, filters[0], self.training)
#        # (32 + latent_dim) * 128 * 128  -->  (32) * 128 * 128
#        self.block1 = DenseBlock(filters[0] + latent_dim, filters[0], self.training)

        self.blocks = nn.ModuleList([
                DenseBlock(filters[i]  + latent_dim, filters[i-1], self.training,
                           add_dropout=add_dropout) for i in reversed(range(1, len(filters)))
            ] + [DenseBlock(filters[0]  + latent_dim, filters[0], self.training,
                            add_dropout=add_dropout)
        ])
        
        self.out_conv = nn.Conv2d(filters[0], 1, kernel_size=3, padding=1)
    
    def forward(self, x, samples):
        assert len(samples) == len(self.filters)
        
        for i in range(len(self.blocks)):
            level = len(samples) - 1 - i
            size = self.image_size // (2**level)
            if i == 0:
                x = F.interpolate(x, size=(size, size), mode='bilinear') 
            x = self.blocks[i](torch.cat([x, samples[level]], dim=1))
            if i != len(self.blocks) - 1:
                x = self.up(x)
        rec = self.out_conv(x)
        return rec, x
        
        
#        samples = self.sample_latent(dist[4], 1, 8, mode)
#        x5_ = self.block5(torch.cat([x, samples], dim=1))
#        samples = self.sample_latent(dist[3], 1, 16, mode)
#        x4_ = self.block4(torch.cat([self.up(x5_), samples], dim=1))
#        samples = self.sample_latent(dist[2], 1, 32, mode)
#        x3_ = self.block3(torch.cat([self.up(x4_), samples], dim=1))
#        samples = self.sample_latent(dist[1], 1, 64, mode)
#        x2_ = self.block2(torch.cat([self.up(x3_), samples], dim=1))
#        samples = self.sample_latent(dist[0], 1, 128, mode)
#        x1_ = self.block1(torch.cat([self.up(x2_), samples], dim=1))
#        
#        rec = self.out_conv(x1_)
#        return rec

class MDecoder(nn.Module):
    def __init__(self, device, latent_dim, filters=None, training=True,
                       image_size=128, add_dropout=False):
        super(MDecoder, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.training = training
        if filters is None:
            filters = [32, 64, 128, 192, 192]
        self.filters = filters
        self.image_size = image_size
        
        self.up = TransitionUp()
#        # (192 + latent_dim + 192) * 8 * 8  -->  (192) * 8 * 8
#        self.block5 = DenseBlock(2 * filters[4] + latent_dim, filters[3], self.training)
#        # (192 + latent_dim + 192) * 16 * 16  -->  (128) * 16 * 16
#        self.block4 = DenseBlock(2 * filters[3] + latent_dim, filters[2], self.training)
#        # (128 + latent_dim + 128) * 32 * 32  -->  (64) * 32 * 32
#        self.block3 = DenseBlock(2 * filters[2] + latent_dim, filters[1], self.training)
#        # (64 + latent_dim + 64) * 64 * 64  -->  (32) * 64 * 64
#        self.block2 = DenseBlock(2 * filters[1] + latent_dim, filters[0], self.training)
#        # (32 + latent_dim + 32) * 128 * 128  -->  (32) * 128 * 128
#        self.block1 = DenseBlock(2 * filters[0] + latent_dim, filters[0], self.training)

        self.blocks = nn.ModuleList([
                DenseBlock(2 * filters[i] + latent_dim, filters[i-1], self.training,
                           add_dropout=add_dropout) for i in reversed(range(1, len(filters)))
            ] + [DenseBlock(2 * filters[0] + latent_dim, filters[0], self.training,
                            add_dropout=add_dropout)
        ])
        
        self.out_conv = nn.Conv2d(2 * filters[0], 1, kernel_size=3, padding=1)
    
    def forward(self, x, x_, samples, x_nec):
        assert type(x_) == list
        assert len(self.filters) == len(samples) == len(x_)
        
        for i in range(len(self.blocks)):
            level = len(samples) - 1 - i
            size = self.image_size // (2**level)
            if i == 0:
                x = F.interpolate(x, size=(size, size), mode='bilinear')
            x = self.blocks[i](torch.cat([x, samples[level], x_[level]], dim=1))
            if i != len(self.blocks) - 1:
                x = self.up(x)
        x = torch.cat([x, x_nec], dim=1)
        rec = self.out_conv(x)
        return rec
        
        
#        samples = self.sample_latent(dist[4], 1, 8, mode)
#        x5_ = self.block5(torch.cat([x, samples, x_[4]], dim=1))
#        samples = self.sample_latent(dist[3], 1, 16, mode)
#        x4_ = self.block4(torch.cat([self.up(x5_), samples, x_[3]], dim=1))
#        samples = self.sample_latent(dist[2], 1, 32, mode)
#        x3_ = self.block3(torch.cat([self.up(x4_), samples, x_[2]], dim=1))
#        samples = self.sample_latent(dist[1], 1, 64, mode)
#        x2_ = self.block2(torch.cat([self.up(x3_), samples, x_[1]], dim=1))
#        samples = self.sample_latent(dist[0], 1, 128, mode)
#        x1_ = self.block1(torch.cat([self.up(x2_), samples, x_[0]], dim=1))
#        
#        rec = self.out_conv(x1_)
#        return rec
    
class Discriminator(nn.Module):
    def __init__(self, device, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
    
        self.net = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.LeakyReLU(0.2, True),
                nn.Linear(50, 50),
                nn.LeakyReLU(0.2, True),
                nn.Linear(50, 50),
                nn.LeakyReLU(0.2, True),
                nn.Linear(50, 2)
            )
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        output = self.net(x)
        return output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    