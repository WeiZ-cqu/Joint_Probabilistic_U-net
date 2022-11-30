import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from .blocks import *
from .net_parts import *
from torch.distributions import Normal, Independent, kl

class Net(nn.Module):
    def __init__(self, n_channels, device, batch_size=5, n_sample=1, latent_dim=8, filters=None,
                       training=True, share=False, fuse=False, use_normal=False,
                       use_PlanarFlow=True, use_InvertFlow=True, use_mKL_V=False):
        super(Net, self).__init__()
        self.n_channels = n_channels
        self.n_sample = n_sample
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.device = device
        self.image_size = 128
        self.use_normal = use_normal
        self.use_PlanarFlow = use_PlanarFlow  # only work for posterior
        self.use_InvertFlow = use_InvertFlow  # only work for prior
        self.use_mKL_V = use_mKL_V
        self.fuse = fuse
        self.share = share
        if filters is None:
            filters = [32, 64, 128, 192, 192]
        self.filters = filters
        
        self.flow_levels = list(range(len(filters)))[:]

        self.Iencoder = IEncoder(n_channels, device, filters, training).to(device)
        self.Mencoder = MEncoder(n_channels if fuse else n_channels + 1, 
                                 device, filters, training, fuse=fuse).to(device)
        self.SampleLatent = SampleLatent(device, filters, n_sample=n_sample, 
                                         latent_dim=latent_dim, share=share,
                                         image_size=self.image_size,
                                         flow_levels=self.flow_levels).to(device)
        self.Idecoder = IDecoder(device, latent_dim, filters, training,
                                 image_size=self.image_size, add_dropout=use_mKL_V).to(device)
        self.Mdecoder = MDecoder(device, latent_dim, filters, training,
                                 image_size=self.image_size, add_dropout=use_mKL_V).to(device)
        self.PlanarFlow = None
        if use_PlanarFlow:
#            self.PlanarFlow = PlanarFlows(device, flow_length=4, latent_dim=8,
#                                         hierarchy=len(filters)).to(device)
            self.PlanarFlow = NICEFlows(device, self.latent_dim,
                                          hierarchy=len(filters)).to(device)
        self.InvertFlow = None
        if use_InvertFlow:
            self.InvertFlow = InvertFlows(device, self.latent_dim,
                                          hierarchy=len(filters)).to(device)
        
        #self.Discriminator = nn.ModuleList([Discriminator(device, self.latent_dim).to(device) 
        #                                    for _ in range(len(self.flow_levels))])
        
    def forward(self, x, m):
        x1_I, x2_I, x3_I, x4_I, x5_I = self.Iencoder(x)
        x1_M, x2_M, x3_M, x4_M, x5_M = self.Mencoder(m if self.fuse 
                                                       else torch.cat([x, m], dim=1), x2_I)
        dist_M, encode_M = self.SampleLatent([x1_M, x2_M, x3_M, x4_M, x5_M], 'M',
                                             mode='rsample', PlanarFlow=self.PlanarFlow)
        dist_I, encode_I = self.SampleLatent([x1_I, x2_I, x3_I, x4_I, x5_I], 'I',
                                             mode='rsample', InvertFlow=self.InvertFlow)
        
        self.normal_latent = self.SampleLatent(x1_I, 'N', mode='rsample')
        samples_hierarchy_I, _, _ = self.SampleLatent.sample_dist(dist_I, mode='rsample',
                                                                  InvertFlow=self.InvertFlow)
        samples_hierarchy_M, self.zk, self.log_det = self.SampleLatent.sample_dist(dist_M,
                                                                          mode='rsample',
                                                                          PlanarFlow=self.PlanarFlow)
        rec_I_with_I, _ = self.Idecoder(encode_I, samples_hierarchy_I)
        rec_I_with_M, x_nec = self.Idecoder(encode_I, samples_hierarchy_M)
        
        rec_M = self.Mdecoder(encode_I, [x1_I, x2_I, x3_I, x4_I, x5_I], samples_hierarchy_M,
                              x_nec)
        #rec_M = self.Mdecoder(encode_I, x_f_, samples_hierarchy_M)
        
        return dist_I, dist_M, rec_I_with_I, rec_I_with_M, rec_M
    
    def save_tensor_for_sample(self, x):
        self.x1_I, self.x2_I, self.x3_I, self.x4_I, self.x5_I = self.Iencoder(x)
    
    def sample(self, x, n_samples):
        if self.use_normal:
            normal_latent = self.SampleLatent(self.x1_I, 'N', mode='rsample')
            samples_hierarchy_N = self.SampleLatent.sample_dist_infer(normal_latent,
                                                                      mode='sample',
                                                                      n_samples=n_samples)
            encode_I = self.sampleLatent.resblock_x(self.x5_I)
        elif self.use_InvertFlow:
            dist_I, encode_I = self.SampleLatent(
                                [self.x1_I, self.x2_I, self.x3_I, self.x4_I, self.x5_I], 'I',
                                                                  mode='sample',
                                                                  InvertFlow=self.InvertFlow)
            samples_hierarchy_I = self.SampleLatent.sample_dist_infer(dist_I, mode='sample',
                                                                      n_samples=n_samples,
                                                                      InvertFlow=self.InvertFlow)
        else:
            dist_I, encode_I = self.SampleLatent(
                                [self.x1_I, self.x2_I, self.x3_I, self.x4_I, self.x5_I], 'I',
                                                                  mode='sample')
            samples_hierarchy_I = self.SampleLatent.sample_dist_infer(dist_I, mode='sample',
                                                                      n_samples=n_samples)
            
        encode_I = encode_I.repeat(n_samples, 1, 1, 1)
        rec_I_with_I, x_nec = self.Idecoder(encode_I, samples_hierarchy_I)
        x1_I, x2_I = self.x1_I.repeat(n_samples, 1, 1, 1), self.x2_I.repeat(n_samples, 1, 1, 1)
        x3_I, x4_I = self.x3_I.repeat(n_samples, 1, 1, 1), self.x4_I.repeat(n_samples, 1, 1, 1)
        x5_I = self.x5_I.repeat(n_samples, 1, 1, 1)
        pre_mask = self.Mdecoder(encode_I, [x1_I, x2_I, x3_I, x4_I, x5_I],
                                 samples_hierarchy_I, x_nec)
#        pre_mask = self.Mdecoder(encode_I, x_f_,
#                                 samples_hierarchy_I)
        
        return pre_mask

    def elbo_rec_I(self, rec_I, image, dist_I, dist_M):
        criterion = nn.L1Loss(size_average=False, reduce=False, reduction=None)
        reconstruction_loss = criterion(input=rec_I, target=image)
        sum_reconstruction_loss = torch.sum(reconstruction_loss) / self.batch_size
        mean_reconstruction_loss = torch.mean(reconstruction_loss)

        mu = torch.zeros(self.batch_size, self.latent_dim, 1, 1, dtype=rec_I.dtype).to(self.device)
        sigma = torch.ones(self.batch_size,self.latent_dim, 1, 1, dtype=rec_I.dtype).to(self.device)
        normal = Independent(Normal(loc=mu, scale=sigma),3)
        mut_kl = 0
        for i in range(len(dist_I)):
            mut_kl += torch.mean(kl.kl_divergence(dist_I[i], self.normal_latent[i]))
        
        return -(sum_reconstruction_loss + 10 * mut_kl)

    def elbo_rec_M(self, rec_I, image, rec_M, mask, dist_I, dist_M):
        criterion_I = nn.L1Loss(size_average=False, reduce=False, reduction=None)
        criterion_M = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)
        reconstruction_loss = 1 * criterion_I(input=rec_I, target=image) + \
                              criterion_M(input=rec_M, target=mask)
        sum_reconstruction_loss = torch.sum(reconstruction_loss) / self.batch_size
        mean_reconstruction_loss = torch.mean(reconstruction_loss)
        
        mu = torch.zeros(self.batch_size, self.latent_dim, 1, 1, dtype=rec_I.dtype).to(self.device)
        sigma = torch.ones(self.batch_size,self.latent_dim, 1, 1, dtype=rec_M.dtype).to(self.device)
        normal = Independent(Normal(loc=mu, scale=sigma),3)
        mut_kl = 0
        for i in range(len(dist_I)):
            if self.use_normal:
                mut_kl += torch.mean(kl.kl_divergence(dist_M[i], self.normal_latent[i]))
                #mut_kl += torch.mean(kl.kl_divergence(dist_M[i], dist_I[i]))
            elif self.use_PlanarFlow and not self.use_InvertFlow and i in self.flow_levels:
                mut_kl += self.kl_loss_zk(self.zk[i], self.log_det[i], dist_I[i])
            elif self.use_PlanarFlow and self.use_InvertFlow  and i in self.flow_levels:
                mut_kl += self.kl_loss_zk(self.zk[i], self.log_det[i], dist_I[i],
                                                     i, self.InvertFlow)
            else:
                mut_kl += torch.mean(kl.kl_divergence(dist_M[i], dist_I[i]))

#        kl_w = torch.abs(mut_kl.detach())/len(dist_I)
#        rec_weight = 1 / (2 * (torch.exp(-1/kl_w)**2))
#        sum_reconstruction_loss = torch.sum(rec_weight * reconstruction_loss) / self.batch_size
#        N = Normal(0, 0.5)
#        #slack = torch.exp(N.log_prob(torch.mean(reconstruction_loss.detach())))
#        slack = torch.exp(N.log_prob(kl_w))
#        
#        return sum_reconstruction_loss, torch.abs((mut_kl / len(dist_I)) - slack * len(dist_I))
        if self.use_mKL_V:
            return sum_reconstruction_loss, torch.abs(mut_kl / len(dist_I) - 0.5)
        else:
            return sum_reconstruction_loss, mut_kl / len(dist_I)

    def kl_loss_zk(self, zk, log_det, target_dist, i=None, InvertFlow=None):
        if InvertFlow is None:
            p_log_prob = target_dist.log_prob(zk)
        else:
            _, p_log_prob = InvertFlow(zk, target_dist, i)
        loss = log_det - p_log_prob
        return loss.mean()
    
    def GAN_KL_Loss(self, dist_I, dist_M, level):
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        if self.PlannarFlow is not None:
            posterior_vec, _ = self.PlanarFlow(dist_M, 'sample', level)
        else:
            posterior_vec = dist_M.rsample()
        if self.InvertFlow is not None:
            prior_vec = self.InvertFlow.sample(dist_I, 'sample', level)
        else:
            prior_vec = dist_I.rsample()
        
        posterior_prob = self.Discriminator(posterior_vec)
        KL = (posterior_prob[:, :1] - posterior_prob[:, 1:]).mean()
        
        prior_prob = self.Discriminator(prior_vec)
        
        d_loss = 0.5*(F.cross_entropy(posterior_prob, zeros) + F.cross_entropy(prior_prob, ones))
        
        return KL, d_loss
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        