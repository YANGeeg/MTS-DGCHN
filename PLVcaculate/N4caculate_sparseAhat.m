clc,clear;

load('adj_sparse_ins9.mat')  %30*30
d_sparse=zeros(30,30);
for i=1:30
      %%  Djj= ÿһ�� Aj �ĺ�
       d_sparse(i,i)=sum(adj_sparse(i,:)); %�ڽӾ����ÿһ�к�
end
save('d_sparse_ins9.mat','d_sparse')
Ahat_sparse=d_sparse^(-0.5)*adj_sparse*d_sparse^(-0.5); %�ڽӾ����һ��  �ڽӾ���ÿһ����Ӷ���100% 
save('Ahat_sparse_ins9.mat','Ahat_sparse')

% Afinal=zeros(30,30);
% for i=1:30
%     Afinal(i,i)=sum(adj_sparse(i,:))+sum(adj_sparse(:,i)); 
% end
% Afinal=Afinal/30; %%����ȡ
%  save('Afinal.mat','Afinal')
 
