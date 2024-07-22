function [ correctRate,retYYSvm,optC,optG] = classifymethod(train_x,train_y,test_x,test_y)
%UNTITLED �˴���ʾ�йش˺����ժҪ
%   �˴���ʾ��ϸ˵��
addpath('./Tools');

addpath('./svm');
[accSvm, retYYSvm, optC, optG] = svmc(train_x, train_y, test_x, test_y, 0, 0, 2, 1);
correctRate = accSvm;

end

