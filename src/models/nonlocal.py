import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import math
import functools


class _NonLocalNd_bn(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out, out_bn, sync_bn, whiten_type, temperature, 
                 with_gc, value_split, gc_beta):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        #assert whiten_type in ['channel', 'spatial']
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(_NonLocalNd_bn, self).__init__()
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if use_out:
            if with_gc and value_split:
                self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
                self.conv_gc_value = conv_nd(inplanes, planes, kernel_size=1)
                self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
            else:
                self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
                self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            if with_gc and value_split:
                self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
                self.conv_gc_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
                self.conv_out = None
            else:
                self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
                self.conv_out = None
        if out_bn:
            if sync_bn:
                self.out_bn = SyncNorm2d(inplanes)
            else:
                self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
        if 'bn_affine' in whiten_type:
            self.key_bn_affine = nn.BatchNorm1d(planes)
            self.query_bn_affine = nn.BatchNorm1d(planes)
        if 'bn' in whiten_type:
            self.key_bn = nn.BatchNorm1d(planes, affine=False)
            self.query_bn = nn.BatchNorm1d(planes, affine=False)
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        if gc_beta:
            self.beta = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.temperature = temperature
        self.with_gc = with_gc
        self.value_split = value_split
        self.gc_beta = gc_beta
        
        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        #init.constant_(self.norm.weight, 0)
        #init.constant_(self.norm.bias, 0)
        #self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]
        query = self.conv_query(x)
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)
        
        if self.with_gc and self.value_split:
            value_gc = self.conv_gc_value(input_x)
            value_gc = value_gc.view(value_gc.size(0), value_gc.size(1), -1)
            
        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)
        
        if 'channel' in self.whiten_type :
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean
        if 'spatial' in self.whiten_type :
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type :
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)
        if 'ln_nostd' in self.whiten_type :
            key_mean = key.mean(1).mean(1).view(key.size(0), 1, 1)
            query_mean = query.mean(1).mean(1).view(query.size(0), 1, 1)
            key -= key_mean
            query -= query_mean
        if 'bn_nostd' in self.whiten_type:
            key_mean = key.mean(2).mean(0).view(1, key.size(1), 1)
            query_mean = query.mean(2).mean(0).view(1, query.size(1), 1)
            key -= key_mean
            query -= query_mean

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map/self.scale
        sim_map = sim_map/self.temperature
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        # if self.norm is not None:
        #     out = self.norm(out)
        out_sim = self.gamma * out_sim
        
        #out = residual + out_sim
        
        if self.with_gc:
            # [N, 1, H', W']
            mask = self.conv_mask(input_x)
            # [N, 1, H'x W']
            mask = mask.view(mask.size(0), mask.size(1), -1)
            mask = self.softmax(mask)
            # [N, C', 1, 1]
            if self.value_split:
                out_gc = torch.bmm(value_gc, mask.permute(0,2,1)).unsqueeze(-1)
            else:
                out_gc = torch.bmm(value, mask.permute(0,2,1)).unsqueeze(-1)
            if self.gc_beta:
                out_gc = self.beta * out_gc
            out_sim = out_sim+out_gc
            
        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)
            
        out = out_sim + residual

        return out

class _NonLocalNd(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out, out_bn):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(_NonLocalNd, self).__init__()
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
            
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        #init.constant_(self.norm.weight, 0)
        #init.constant_(self.norm.bias, 0)
        #self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]
        query = self.conv_query(x)
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        # [N, C', T x H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', T x H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map/self.scale
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out = out.transpose(1, 2)
        # [N, C', T,  H, W]
        out = out.view(out.size(0), out.size(1), *x.size()[2:])
        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out = self.conv_out(out)
        # if self.norm is not None:
        #     out = self.norm(out)
        out = self.gamma * out
        if self.out_bn:
            out = self.out_bn(out)
        out = residual + out
        return out

class NonLocal2d_bn(_NonLocalNd_bn):

    def __init__(self, inplanes, planes, downsample=True, use_gn=False, lr_mult=None, use_out=False, out_bn=False, sync_bn=False, whiten_type=['channel'], temperature=1.0, with_gc=False, value_split=False, gc_beta=False):
        super(NonLocal2d_bn, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample, use_gn=use_gn, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn, sync_bn=sync_bn, whiten_type=whiten_type, temperature=temperature, with_gc=with_gc, value_split=value_split, gc_beta=gc_beta)
        
class NonLocal2d(_NonLocalNd):

    def __init__(self, inplanes, planes, downsample=False, use_gn=False, lr_mult=None, use_out=False, out_bn=False):
        super(NonLocal2d, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample, use_gn=use_gn, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn)
