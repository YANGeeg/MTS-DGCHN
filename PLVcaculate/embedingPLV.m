clc,clear;
load('Data.mat') %%432*30*4000
load('Ahat.mat')
load('Ahat_sparse.mat')
load('adj_sparse.mat')
load('Ah.mat')

%  embeddata=zeros(432,30,4000);
%  for i=1:432
%      x=squeeze(Data(i,:,:));
%      %X=Ahat*x;
%      X=Ahat_sparse*x;
%      embeddata(i,:,:)=X;
%  end
%  save('embeddata.mat','embeddata')

  embeddata1=zeros(432,30,4000);
  for i=1:432
      x=squeeze(Data(i,:,:));
      %X=Ahat*x;
      X=adj_sparse*x;
      embeddata1(i,:,:)=X;
  end
  save('embeddata1.mat','embeddata1')

