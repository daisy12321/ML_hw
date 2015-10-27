function [X_scaled, X_mean, X_sd] = scale_std(X)
    [n,p] = size(X);
    X_scaled = zeros(n,p);
    X_sd = zeros(1,p);
    X_mean = zeros(1,p);
    for j = 1:p
        X_sd(j) = std(X(:, j));
        X_mean(j) = mean(X(:, j));
        %max_tmp = max(X(:,j));
        %min_tmp = min(X(:,j));
%         if max_tmp == min_tmp 
%             denom = 1;
%         else
%             denom = max_tmp - min_tmp;
%         end

        X_scaled(:, j) = (X(:,j) - X_mean(j))/X_sd(j);
    end
end