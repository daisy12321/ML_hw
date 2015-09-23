function [X,Y] = bishopCurveData()
%y = sin(2 pi x) + N(0,0.3),
data = importdata('alternate_curvefitting.txt');

X = data(1,:);
Y = data(2,:);

