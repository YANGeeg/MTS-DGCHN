clc,clear;
load('Adj.mat')
load('D.mat')

% d=zeros(30,30);
% for i=1:1024  %%ȡһ��ʱ�̵ģ�������ȡƽ�� 
%     d=d+squeeze(D(i,:,:));
% end
% n_d=d/1024;%�Ⱦ���
% adj=zeros(30,30);
% for i=1:1024    
%     adj=adj+squeeze(Adj(i,:,:));
% end
% n_adj=adj/1024; %%ȡһ��ʱ�̵ģ�������ȡƽ��
n_adj=squeeze(Adj(640,:,:));
n_d=squeeze(D(640,:,:));
save('n_adj.mat','n_adj')  %%30*30
save('n_d.mat','n_d')    %%30*30