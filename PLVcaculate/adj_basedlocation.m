clc,clear;
[num,txt]=xlsread('Adj.xlsx',1,'B2:AE31');%��������
Adj_location=num;
done=zeros(30,30);
for i=1:30
      %%  Djj= ÿһ�� Aj �ĺ�
       done(i,i)=sum(Adj_location(i,:)); %�ڽӾ����ÿһ�к�
end
Adj_location=done^(-0.5)*Adj_location*done^(-0.5); %�ڽӾ����һ��  �ڽӾ���ÿһ����Ӷ���100% 
save('Adj_location.mat','Adj_location')