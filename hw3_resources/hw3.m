addpath('srcs', 'hw3_resources')
%set(gcf,'Visible','off')              % turns current figure "off"
%set(0,'DefaultFigureVisible','off');  % all subsequent figures "off"

%%% 3.1 Neuro networks %%%%
X = [1, 2; 4, 5; 6, 9; 8, 11; 11, 14; 15, 20];
Y = [1, 0; 0, 1; 0, 1; 1, 0; 1, 0; 1,0];
W1 = [1, 2; 3, 5; 5,1];
W2 = [2 1 1; 5 -1 3];

w1_0 = ones(size(W1));
w2_0 = ones(size(W2));

grad_desc_3(@ANN_loss, w1_0, w2_0, X, Y, 1, 0.001)


importdata('hw3_resources/data/toy_multiclass_1_train.csv')

