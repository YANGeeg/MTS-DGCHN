clc,clear;
load('n_adj.mat')  %30*30
%a=30;

p=0.9;%%ϡ���
%del=a*p;
%%%%%����2
b=sort(n_adj(:));
c=size(b);
del=c(1)*p;
for i=1:30
    for j=1:30
        if(n_adj(i,j)<=b(del))
            n_adj(i,j)=0;
        end
    end
end
%%%%%����1            
% ind=zeros(30,30);
% for i=1:30
%     [new,index]=sort(n_adj(i,:));%ÿһ�д�С��������
%     ind(i,:)=index;
% end
% for i=1:30
%     for j=1:del
%          n_adj(i,ind(i,j))=0;
%     end
% end
adj_sparse=n_adj;
save('adj_sparse_ins9.mat','adj_sparse')  %30*30

%��ͼ
x=adj_sparse;
XVarNames = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'};

matrixplot(x,'XVarNames',XVarNames,'YVarNames',XVarNames,'TextColor',[0.6,0.6,0.6],'ColorBar','on');
