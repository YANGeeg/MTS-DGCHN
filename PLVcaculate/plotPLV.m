clc,clear;
load('n_adj.mat')
x=n_adj;
XVarNames = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'};

matrixplot(x,'XVarNames',XVarNames,'YVarNames',XVarNames,'TextColor',[0.6,0.6,0.6],'ColorBar','on');
