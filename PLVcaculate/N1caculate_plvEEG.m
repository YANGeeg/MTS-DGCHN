clc;clear;
load('Data_independent.mat');
data=Data_independent;  %%%%%改动
Dataset=permute(data,[2,3,1]);%%%%432*30*4000   ----> 30*4000*432
data=Dataset;

%PLV计算需要输入数据格式 电极×采样点×试次
eegData = data; 
srate = 256; %Hz
 filtSpec.order = 120;
 filtSpec.range = [1 40]; %Hz
 dataSelectArr = rand(432, 1) >= 0.5; % attend trials
dataSelectArr(:, 2) = ~dataSelectArr(:, 1); % ignore trials
 [plv] = pn_eegPLV0(eegData, srate, filtSpec, dataSelectArr);
 figure; plot((0:size(eegData, 2)-1)/srate, squeeze(plv(:, 1, 16, :)));
  xlabel('Time (s)'); ylabel('Plase Locking Value');

[q,w,e,r]=size(plv);
plvdata=zeros(q,w,e);
Adj=zeros(q,30,30);
% D0=zeros(q,30,30);
D=zeros(q,30,30);
for i=1:q
   x=squeeze(plv(i,:,:,1));
   y=squeeze(plv(i,:,:,2));
   y=y';
   z=x+y;
   plvdata(i,:,:)=z;
   a=z+eye(30);
   Adj(i,:,:)=a;  %Adj矩阵  等于PLV  +单位矩阵
   for j=1:30
      % D0(i,j,j)=sum(z(j,:));%%  Djj= 每一行 Aj 的和
       D(i,j,j)=sum(a(j,:)); %有的论文里度矩阵等于加上单位矩阵的邻接矩阵的每一行和
   end
       
end
save('plvdata.mat','plvdata')%%4000*30*30  PLV矩阵
save('Adj.mat','Adj') %%%4000*30*30    Adj矩阵  等于PLV  +单位矩阵，自己和自己有关系
save('D.mat','D')   %%%%4000*30*30  度矩阵


