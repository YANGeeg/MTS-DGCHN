clc,clear;
[num,txt]=xlsread('Adj.xlsx',1,'B2:AE31');%读入数据
Adj_location=num;
done=zeros(30,30);
for i=1:30
      %%  Djj= 每一行 Aj 的和
       done(i,i)=sum(Adj_location(i,:)); %邻接矩阵的每一行和
end
Adj_location=done^(-0.5)*Adj_location*done^(-0.5); %邻接矩阵归一化  邻接矩阵每一行相加都是100% 
save('Adj_location.mat','Adj_location')