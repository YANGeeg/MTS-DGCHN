
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import scipy.io as sio


def load_adj():
    path = Path('../myTGCN/data/Ahat')
    file_codedata = 'Ahat_sparse09s1' + '.mat'
    # file_codedata = 'Ahat' + '.mat'
    filedata = path / file_codedata
    adjData = sio.loadmat(filedata)
    adj = adjData['Ahat_sparse']
    A = torch.from_numpy(adj).float()
    A_hat = A
    # print(A_hat.size())
    return A_hat

###Q K V
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v):
        # q x k^T
        # attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  ### H*W  * W*H
        # # dim=-1表示对最后一维softmax
        # attn = self.dropout(F.softmax(attn, dim=-1))
        # output = torch.matmul(attn, v)  #  H*H  *  H*W  得到H个权重，这和我的时间片加权不符合，加到了30通道上，所以改成下面的
        attn = torch.matmul(q.transpose(2, 3) / self.temperature, k)  ### W*H  * H*W
        # dim=-1表示对最后一维softmax
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(v,attn)  #   H*W * W*W  得到W个权重
        return output

##used
class Tblock3(nn.Module):   ###256采样
    def __init__(self,  input_size, num_T):
        # input_size: EEG channel x datapoint
        super(Tblock3, self).__init__()  # 子类把父类的__init__()放到自己的__init__()当中

        self.Tception = nn.Sequential(
            nn.Conv2d(input_size[0], input_size[0], kernel_size=(1, 7), stride=(1, 4), padding=0),
            nn.ReLU(), )
        size = self.get_size(input_size)
        self.attention = ScaledDotProductAttention(temperature=size[3] ** 0.5)
        #self.layer_norm = nn.LayerNorm(size[3], eps=1e-6)

        self.Tception1 = nn.Sequential(
            nn.Conv2d(input_size[0], num_T, kernel_size=(1, 7), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(input_size[0], num_T, kernel_size=(1, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(input_size[0], num_T, kernel_size=(1, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))

        # self.Tception4 = nn.Sequential(
        #     nn.Conv2d(num_T, num_T, kernel_size=(1, 3), stride=1, padding=0),
        #     nn.ReLU(),
        #     )

        self.BN_t = nn.BatchNorm2d(num_T)  # 进行数据的归一化处理

    #  1*num_T*30*t
    def forward(self, x):

        input = self.Tception(x)
        q,k,v=input,input,input
        out=self.attention(q,k,v)
        input = out+input
        #input = self.layer_norm(input)

        #########

        y = self.Tception1(input)
        out = y
        y = self.Tception2(input)
        out = torch.cat((out, y), dim=3)  # 行连接
        y = self.Tception3(input)
        out = torch.cat((out, y), dim=3)
        # out=self.Tception4(out)
        out = self.BN_t(out)

        return out

    def get_size(self, input_size):  ##加权个数
        data = torch.ones((1, input_size[0], input_size[1], input_size[2]))
        y = self.Tception(data)
        out = y
        #print(out.size())
        return out.size()

class GCNBlock(nn.Module):      ##  input_size, num_T

    def __init__(self, GCN_in,GCN_out ):

        super(GCNBlock, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(GCN_in,GCN_out))  ###out_channels应该是时间特征数
        self.reset_parameters1()
        A_hat=load_adj()
        self.A_hat = nn.Parameter(A_hat,requires_grad=True)

    def reset_parameters1(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X):

        lfs = torch.einsum("ij,jklm->kilm", [self.A_hat, X.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        return t2




#used
class DGCN(nn.Module):

    def __init__(self, num_classes, input_size, num_T, num_S,GCN_hiden1,GCN_hiden2,GCN_hiden3,
                 hiden,dropout_rate):

        super(DGCN, self).__init__()

        self.GCN1=GCNBlock(GCN_in=input_size[2],GCN_out=GCN_hiden1)
        self.GCN2 = GCNBlock(GCN_in=GCN_hiden1, GCN_out=GCN_hiden2)
        self.GCN3 = GCNBlock(GCN_in=GCN_hiden2, GCN_out=GCN_hiden3)

        self.batch_norm = nn.BatchNorm2d(input_size[1])  ##input_size[1]=30

        self.BN_s = nn.BatchNorm2d(num_S)

        self.Sception1 = nn.Sequential(
            nn.Conv2d(input_size[0], num_S, kernel_size=(30, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)))
        # self.Sception2 = nn.Sequential(
        #     nn.Conv2d(input_size[0], num_S, kernel_size=(15, 1), stride=(15, 1), padding=0),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)))
        # self.Sception3 = nn.Sequential(
        #     nn.Conv2d(input_size[0], num_S, kernel_size=(10, 1), stride=(10, 1), padding=0),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)))

        self.fc1 = nn.Sequential(
            nn.Linear(1*num_S*GCN_hiden3, hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())


    def forward(self,X):

        out = X  ### B*9*30*Tfetures
        out = out.permute(0, 2, 1, 3)#  B*30*9*gcn
        out=self.GCN1(out)  #  B*30*9*gcn
        self.feature =out.view(out.size()[0], -1)
        out=self.GCN2(out)
        out = self.GCN3(out)

        out = self.batch_norm(out)

        out = out.permute(0, 2, 1, 3)  ###  B*30*9*gcn---->B*9*30*gcn
        z = self.Sception1(out)
        out_final = z
        # z = self.Sception2(out)
        # out_final = torch.cat((out_final, z), dim=2)
        # z = self.Sception3(out)
        # out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)

        out = out.view(out.size()[0], -1)
        self.feature=out
        out= self.fc1(out)
        out=self.fc2(out)
        return out

class TCNmodel2(nn.Module):

    def __init__(self, input_size, num_T, num_S,
                 ):

        super(TCNmodel2, self).__init__()

        self.Get_timefeatures = Tblock3(input_size=input_size, num_T=num_T, )  ###为了得到时间特征的个数 Tsize

        self.BN_s = nn.BatchNorm2d(num_S)

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(30, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(15, 1), stride=(15, 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.conv = nn.Sequential(
            nn.Conv2d(num_S, num_S, kernel_size=(3, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
            )

        # size = self.get_size(input_size)
        #
        # self.fc1 = nn.Sequential(
        #     nn.Linear(size[1], hiden),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate))
        # self.fc2 = nn.Sequential(
        #     nn.Linear(hiden, num_classes),
        #     nn.LogSoftmax())


    def forward(self,X ):

        out = self.Get_timefeatures(X)  ### B*9*30*Tfetures
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out_final=self.conv(out_final)

        out = self.BN_s(out_final)
        # out = out.view(out.size()[0], -1)
        # out= self.fc1(out)
        # out=self.fc2(out)
        return out


    # def get_size(self, input_size):
    #     data = torch.ones((1, input_size[0], input_size[1], input_size[2]))
    #     out=self.Get_timefeatures(data)
    #     z = self.Sception1(out)
    #     out_final = z
    #     out = self.BN_s(out_final)
    #     out = out.view(out.size()[0], -1)
    #     return out.size()
class GCNmodel3(nn.Module):

    def __init__(self,  input_size, num_S,GCN_hiden1,GCN_hiden2,GCN_hiden3,
                 ):

        super(GCNmodel3, self).__init__()

        self.GCN1=GCNBlock(GCN_in=input_size[2],GCN_out=GCN_hiden1)
        self.GCN2 = GCNBlock(GCN_in=GCN_hiden1, GCN_out=GCN_hiden2)
        self.GCN3 = GCNBlock(GCN_in=GCN_hiden2, GCN_out=GCN_hiden3)

        self.batch_norm = nn.BatchNorm2d(input_size[1])  ##input_size[1]=30

        self.BN_s = nn.BatchNorm2d(num_S)

        self.Sception1 = nn.Sequential(
            nn.Conv2d(input_size[0], num_S, kernel_size=(30, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)))


    def forward(self,X ):

        out = X  ### B*9*30*Tfetures
        out = out.permute(0, 2, 1, 3)#  B*30*1*gcn
        out=self.GCN1(out)  #  B*30*1*gcn
        out=self.GCN2(out)
        out = self.GCN3(out)

        out = self.batch_norm(out)

        out = out.permute(0, 2, 1, 3)  ###  B*30*1*gcn---->B*1*30*gcn
        # print(out.size())
        z = self.Sception1(out)
        out_final = z
        # z = self.Sception2(out)
        # out_final = torch.cat((out_final, z), dim=2)
        # z = self.Sception3(out)
        # out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)

        # out = out.view(out.size()[0], -1)
        # out= self.fc1(out)
        # out=self.fc2(out)
        return out

class MTS_DGCHN(nn.Module):

    def __init__(self, num_classes, input_size, num_T, num_S,GCN_hiden1,GCN_hiden2,GCN_hiden3,
                 hiden,dropout_rate):

        super(MTS_DGCHN, self).__init__()

        self.Get_T = TCNmodel2(input_size=input_size, num_T=num_T, num_S=num_S)  ###为了得到时间特征的个数 Tsize

        self.Get_S = GCNmodel3(input_size=input_size, num_S=num_S,
                              GCN_hiden1=GCN_hiden1,GCN_hiden2=GCN_hiden2,GCN_hiden3=GCN_hiden3,)



        self.conv = nn.Sequential(
            nn.Conv2d(2*num_S,num_S, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(),
            )

        self.BN = nn.BatchNorm2d(num_S)  # 进行数据的归一化处理
        size = self.get_size(input_size=input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())


    def forward(self,X ):
        # self.raw =X.view(X.size()[0], -1)
        out_t = self.Get_T(X)

        out_s=self.Get_S(X)
        out= torch.cat((out_t, out_s), dim=1)
        out=self.conv(out)
        out = self.BN(out)

        out = out.view(out.size()[0], -1)
        self.feature = out

        out= self.fc1(out)
        out=self.fc2(out)
        return out

    def get_size(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], input_size[2]))

        out_t = self.Get_T(data)
        out_s = self.Get_S(data)
        # print(out_s.size())
        out = torch.cat((out_t, out_s), dim=1)
        out = self.conv(out)
        out = self.BN(out)
        out = out.view(out.size()[0], -1)
        return out.size()



# if __name__ == "__main__":

#     model = MTS_DGCHN(3, (1, 30, 256), 12, 18, 192, 128,  43,    512, 0.2)
