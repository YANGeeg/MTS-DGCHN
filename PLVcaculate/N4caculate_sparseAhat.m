clc,clear;

load('adj_sparse_ins9.mat')  %30*30
d_sparse=zeros(30,30);
for i=1:30
      %%  Djj= 每一行 Aj 的和
       d_sparse(i,i)=sum(adj_sparse(i,:)); %邻接矩阵的每一行和
end
save('d_sparse_ins9.mat','d_sparse')
Ahat_sparse=d_sparse^(-0.5)*adj_sparse*d_sparse^(-0.5); %邻接矩阵归一化  邻接矩阵每一行相加都是100% 
save('Ahat_sparse_ins9.mat','Ahat_sparse')

% Afinal=zeros(30,30);
% for i=1:30
%     Afinal(i,i)=sum(adj_sparse(i,:))+sum(adj_sparse(:,i)); 
% end
% Afinal=Afinal/30; %%随意取
%  save('Afinal.mat','Afinal')
 
