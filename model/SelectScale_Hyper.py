import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from graph.hypergraph import *
from .basic_ST_unit import TCN_GCN_unit as TCN_GCN_unit_origin
from .pos_encoding import PositionalEncoding

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, num_frame, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)

        self.G_part = nn.Parameter(torch.from_numpy(G_part.astype(np.float32)))
        self.G_body = nn.Parameter(torch.from_numpy(G_body.astype(np.float32)))
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.GloAvgMat = torch.ones(25, 25) / 25
        self.num_subset = num_subset
        self.num_point = num_point

        self.conv_joint = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_joint.append(nn.Conv2d(in_channels, out_channels, 1))
        
        self.PE = PositionalEncoding(in_channels, num_point, num_frame, 'spatial')
        self.conv_theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_phi = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_nonlocal = nn.Conv2d(in_channels, out_channels, 1)
        
        self.conv_part = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_body = nn.Conv2d(in_channels, out_channels, 1)
        '''self.conv_avg = nn.Conv2d(in_channels, out_channels, 1)'''

        # selective-scale.
        self.num_branch = 3
        d = int(out_channels / 2)
        self.fc = nn.Linear(out_channels, d)
        self.fc_branch = nn.ModuleList([])
        for i in range(3):
            self.fc_branch.append(
                nn.Linear(d, out_channels)
            )
        self.softmax = nn.Softmax(dim=1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_joint[i], self.num_subset)

    def norm(self, A):
        D_list = torch.sum(A, 0).view(1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = torch.eye(self.num_point).to(device=A.device) * D_list_12
        A = torch.matmul(A, D_12)
        return A

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        G_part = self.G_part.cuda(x.get_device())
        G_body = self.G_body.cuda(x.get_device())
        Avg = self.GloAvgMat.cuda(x.get_device())

        x_joint = None
        for i in range(self.num_subset):
            x_temp = x.view(N, C * T, V)
            x_temp = self.conv_joint[i](torch.matmul(x_temp, self.norm(A[i])).view(N, C, T, V))
            x_joint = x_temp + x_joint if x_joint is not None else x_temp

        x_withPE = self.PE(x)
        theta = self.conv_theta(x_withPE).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
        phi = self.conv_phi(x_withPE).view(N, self.inter_c * T, V)
        att_map = self.soft(torch.matmul(theta, phi) / theta.size(-1))  # N V V
        x_nonlocal = x.view(N, C * T, V)
        x_nonlocal = torch.matmul(x_nonlocal, att_map).view(N, C, T, V)
        x_nonlocal = self.conv_nonlocal(x_nonlocal)

        x_part = x.view(N, C * T, V)
        x_part = torch.matmul(x_part, self.norm(G_part)).view(N, C, T, V)
        x_part = self.conv_part(x_part)
        x_body = x.view(N, C * T, V)
        x_body = torch.matmul(x_body, self.norm(G_body)).view(N, C, T, V)
        x_body = self.conv_body(x_body)
        """
        x_avg = x.view(N, C * T, V)
        x_avg = torch.matmul(x_avg, Avg).view(N, C, T, V)
        x_avg = self.conv_avg(x_avg)
        y = x_joint + x_part + x_body + x_nonlocal + x_avg
        """

        # selective-scale.
        x_joint = x_joint.unsqueeze_(dim=1)
        x_part = x_part.unsqueeze_(dim=1)
        x_body = x_body.unsqueeze_(dim=1)
        
        x_local = torch.cat([x_joint, x_part, x_body], dim=1)
        x_sum = torch.sum(x_local, dim=1)
        glo_avg = x_sum.mean(-1).mean(-1)
        feature_z = self.fc(glo_avg)

        attention_vectors = None
        for i, fc in enumerate(self.fc_branch):
            vector = fc(feature_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        x_local_selected = (x_local * attention_vectors).sum(dim=1)
        
        y = x_local_selected + x_nonlocal
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, num_frame, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, G_part, G_body, num_point, num_frame)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()
        
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        G_part = generate_G_part()
        G_body = generate_G_body()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit_origin(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A, G_part, G_body, num_point, 300)
        self.l3 = TCN_GCN_unit(64, 64, A, G_part, G_body, num_point, 300)
        self.l4 = TCN_GCN_unit(64, 64, A, G_part, G_body, num_point, 300)
        self.l5 = TCN_GCN_unit(64, 128, A, G_part, G_body, num_point, 300, stride=2, pool_channels=[128, 300])
        self.l6 = TCN_GCN_unit(128, 128, A, G_part, G_body, num_point, 150)
        self.l7 = TCN_GCN_unit(128, 128, A, G_part, G_body, num_point, 150)
        self.l8 = TCN_GCN_unit(128, 256, A, G_part, G_body, num_point, 150, stride=2, pool_channels=[256, 150])
        self.l9 = TCN_GCN_unit(256, 256, A, G_part, G_body, num_point, 75)
        self.l10 = TCN_GCN_unit_origin(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
