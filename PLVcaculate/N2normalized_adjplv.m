clc,clear;
load('Adj.mat')
load('D.mat')

% d=zeros(30,30);
% for i=1:1024  %%取一个时刻的，？还是取平均 
%     d=d+squeeze(D(i,:,:));
% end
% n_d=d/1024;%度矩阵
% adj=zeros(30,30);
% for i=1:1024    
%     adj=adj+squeeze(Adj(i,:,:));
% end
% n_adj=adj/1024; %%取一个时刻的，？还是取平均
n_adj=squeeze(Adj(640,:,:));
n_d=squeeze(D(640,:,:));
save('n_adj.mat','n_adj')  %%30*30
save('n_d.mat','n_d')    %%30*30