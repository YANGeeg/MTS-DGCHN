clc,clear;
A=ones(30,30);
done=zeros(30,30);
for i=1:30
      %%  Djj= 每一行 Aj 的和
       done(i,i)=sum(A(i,:)); %邻接矩阵的每一行和
end
A=done^(-0.5)*A*done^(-0.5); %邻接矩阵归一化  邻接矩阵每一行相加都是100% 
save('A.mat','A')