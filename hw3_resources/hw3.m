addpath('srcs', 'hw3_resources')
%set(gcf,'Visible','off')              % turns current figure "off"
%set(0,'DefaultFigureVisible','off');  % all subsequent figures "off"

%%% 3.1 Neuro networks %%%%
X = [1, 2];
Y = [1, 0];
W1 = [1, 2; 3, 5; 5,1];
W2 = [2 1 1; 5 -1 3];
d = length(X);
[K, M] = size(W2);
[a1, Z, a2, h] = fwd_prop(X, W1, W2)%compute estimated y_k from x and w

dE2Log = zeros(K, M);
dE2Log_sub = zeros(K, M);

dE1Log = zeros(K, d);
dE1Log_sub = zeros(K, d);

dl_w2 = zeros(1, M);
dl_w1 = zeros(K, M);

dSig2 = zeros(1, K);
dSig1 = zeros(1, M);
for k = 1:K
    % Layer 2
    dSig2(k) = sigmoid(a2(k))*(1-sigmoid(a2(k)));
    dE2Log(k, :) = 1/h(k)*dSig2(k)*Z';
    dE2Log_sub(k, :) = -1/(1-h(k))*dSig2(k)*Z';
    dl_w2 = dl_w2 + -Y(k).* dE2Log(k,:) - (1-Y(k)) .* dE2Log_sub(k,:);
end
disp(dl_w2);

dSig1 = sigmoid(a1) .* (1-sigmoid(a1)); % in Mx1
dSig1Matrix = repmat(dSig1', K, 1);
for k = 1:K
    % layer 1
    dE1Log_tmp = 1/h(k)*dSig2(k)* W2 .* dSig1Matrix;
    dE1Log_sub_tmp = -1/(1-h(k))*dSig2(k)* W2 .* dSig1Matrix;
    dl_w1 = dl_w1 + -Y(k).* dE1Log_tmp - (1-Y(k)) .* dE1Log_sub_tmp;
end
disp(dl_w1);