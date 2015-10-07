addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 2.1 & 1.2 %%%%
% L2 regularized LR using gradient descent
% LR_train = load('data_stdev1_train.csv');
LR_train = load('data_stdev2_train.csv');
% LR_train = load('data_stdev4_train.csv');
% LR_train = load('data_nonsep_train.csv');
X_train = [ones(size(Y_train)), LR_train(:, 1:2)];
Y_train = LR_train(:, 3);
neg_idx = Y_train == -1;
pos_idx = Y_train == 1;

n = size(X_train,1);
p = size(X_train,2);

lambda = 0.5
LR_loss = @(w) lambda * norm(w(2:p), 2)^2 + sum(-log(sigmoid(Y_train .* (X_train * w'))))
LR_grad = @(w) 2 * lambda * w  + sum((-repmat(Y_train, 1, p) .* X_train) ...
                .* repmat(1 - sigmoid(Y_train .* (X_train * w')), 1, p))
[x, f, x_hist] = grad_desc(LR_loss, LR_grad, [0  1 1],1, 1e-6)
%[x, f] = grad_desc_2(LR_loss, [1  1 1], 3, 1e-6)

%%% compare to Matlab native function:
options = optimoptions(@fminunc,'Display','iter');
[z, f] = fminunc(LR_loss, [1, 1, 1], options)

% plot the contour and the optimum point
h = figure;
hold on;
plot(X_train(pos_idx, 2), X_train(pos_idx, 3), 'ro');
plot(X_train(neg_idx, 2), X_train(neg_idx, 3), 'bx');
plot([-5, 5], [(-x(1)+x(2)*5)/x(3), (-x(1)-x(2)*5)/x(3)], 'y-');
plot([-5, 5], [(-z(1)+z(2)*5)/z(3), (-z(1)-z(2)*5)/z(3)], 'g-');
hold off;
h_legend = legend('(+)', '(-)', 'In-house GD','fminunc')
    set(h_legend,'FontSize',14);
set(h,'Units','Inches');
xlabel('x1')
ylabel('x2') 
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h, 'hw2_1.pdf', '-dpdf', '-r0')


