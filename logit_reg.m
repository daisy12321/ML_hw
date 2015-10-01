%%% 1.Appendix %%%%
%%% stochastic gradient descent for LR %%%
% simulate data
X_1 = mvnrnd([1 1],eye(2), 10)'
X_2 = mvnrnd([-1 -1],eye(2), 10)'
X = [X_1, X_2]
Y = vertcat(ones(10,1),zeros(10,1))
figure;
hold on;
plot(X(1,1:10), X(2,1:10), 'r+')
plot(X(1,11:20), X(2,11:20), 'o')
hold off;

fun = @theta - (Y .* log(sigmoid(@theta * X)') + (1 - Y) .* log(1 - sigmoid(@theta * X)'));

