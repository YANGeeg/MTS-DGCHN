clc,clear;
A=ones(30,30);
done=zeros(30,30);
for i=1:30
      %%  Djj= ÿһ�� Aj �ĺ�
       done(i,i)=sum(A(i,:)); %�ڽӾ����ÿһ�к�
end
A=done^(-0.5)*A*done^(-0.5); %�ڽӾ����һ��  �ڽӾ���ÿһ����Ӷ���100% 
save('A.mat','A')