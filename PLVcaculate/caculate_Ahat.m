clc,clear;

load('n_adj.mat')  %30*30
load('n_d.mat')

Ahat=n_d^(-0.5)*n_adj*n_d^(-0.5); %�ڽӾ����һ��  ���Ϸ��ȼۣ��Ⱦ����-0.5 �η� û�� NAN 
save('Ahat.mat','Ahat')