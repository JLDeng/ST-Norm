import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from scipy.linalg import block_diag
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class DilateConv(nn.Module):
    def __init__(self, num_nodes, n_his, n_inputs, n_outputs, kernel_size, stride, dilation, padding, stnorm_bool, tnorm_bool, snorm_bool):
        super(DilateConv, self).__init__()
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        self.stnorm_bool = stnorm_bool
        if tnorm_bool:
            self.tn = nn.BatchNorm1d(num_nodes * n_inputs, track_running_stats=False, affine=True)
        if snorm_bool:
            self.sn = nn.InstanceNorm1d(n_inputs, track_running_stats=False, affine=True)
        if stnorm_bool:
            self.stn = nn.InstanceNorm2d(n_inputs, track_running_stats=False, affine=True)
        num = int(tnorm_bool) + int(snorm_bool) + int(stnorm_bool) + 1

        self.conv = weight_norm(nn.Conv2d(n_inputs * num, n_outputs, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation= dilation))


    def forward(self, x):
        b, c, n, t = x.shape

        x_list = [x]
        if self.tnorm_bool:
            x_tnorm = self.tn(x.reshape(b, c * n, t)).view(b, c, n, t)
            x_list.append(x_tnorm)
        if self.snorm_bool:
            x_snorm = self.sn(x.permute(0, 3, 1, 2).reshape(b * t, c, n)).view(b, t, c, n).permute(0, 2, 3, 1)
            x_list.append(x_snorm)
        if self.stnorm_bool:
            x_stnorm = self.stn(x)
            x_list.append(x_stnorm)
        x = torch.cat(x_list, dim=1)
        out = self.conv(x)
        return out



class TemporalBlock(nn.Module):
    def __init__(self, num_nodes, n_his, n_inputs, n_outputs, kernel_size, stride, dilation, padding, stnorm_bool, tnorm_bool, snorm_bool, dropout=0):
        super(TemporalBlock, self).__init__()
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
        # change covn type
        self.conv1 = DilateConv(num_nodes, n_his, n_inputs, n_outputs, kernel_size, stride, dilation, padding, stnorm_bool=stnorm_bool, tnorm_bool=tnorm_bool, snorm_bool=snorm_bool)
        self.conv2 = DilateConv(num_nodes, n_his, n_outputs, n_outputs, kernel_size, stride, dilation, padding, stnorm_bool=stnorm_bool, tnorm_bool=tnorm_bool, snorm_bool=snorm_bool)

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net1 = nn.Sequential(self.chomp1, self.relu1, self.dropout1)
        self.net2 = nn.Sequential(self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        b, n, ic, t = x.shape
        #x = self.bn(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = self.conv1(x)
        out = self.net1(out)
        out = self.conv2(out)
        out = self.net2(out)
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)
        out_np = out.data.cpu().numpy()
        #np.save(f"rep_{self.layer_id}", out_np)

        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_nodes, in_channels, n_his, n_pred, hidden_channels, n_layers, stnorm_bool, snorm_bool, tnorm_bool, kernel_size=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        decode_layers = []
        channels = [in_channels] + [hidden_channels] * n_layers
        for i in range(n_layers):
            dilation_size = 2 ** i
            in_channels = channels[i]
            out_channels = channels[i + 1]
            layers += [TemporalBlock(num_nodes, n_his, in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, snorm_bool=snorm_bool, tnorm_bool=tnorm_bool, stnorm_bool=stnorm_bool)]
        self.layers = nn.ModuleList(layers)
        self.out_conv = nn.Conv2d(hidden_channels, n_pred, 1)

    def forward(self, x):
        b, t, n, c = x.size()
        x = x.permute(0, 3, 2, 1)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        out = out[..., -1:]
        out = self.out_conv(out)
        return out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)

