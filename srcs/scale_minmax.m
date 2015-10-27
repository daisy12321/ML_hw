function [X_scaled, X_min, denom] = scale_minmax(X)
    [n,p] = size(X);
    X_scaled = zeros(n,p);
    X_min = zeros(1,p);
    X_max = zeros(1,p);
    denom = zeros(1,p);
    for j = 1:p
        X_min(j) = min(X(:,j));
        X_max(j) = max(X(:,j));
        if X_max(j) == X_min(j) 
            denom(j) = 1;
        else
            denom(j) = X_max(j) - X_min(j) ;
        end

        X_scaled(:, j) = (X(:,j) - X_min(j))/denom(j);
    end
end