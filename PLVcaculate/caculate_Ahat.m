clc,clear;

load('n_adj.mat')  %30*30
load('n_d.mat')

Ahat=n_d^(-0.5)*n_adj*n_d^(-0.5); %邻接矩阵归一化  与上方等价，度矩阵的-0.5 次方 没有 NAN 
save('Ahat.mat','Ahat')