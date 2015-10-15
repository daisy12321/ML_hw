function X_scaled = scale(X)
    [n,p] = size(X);
    X_scaled = zeros(n,p);
    for j = 1:p
        max_tmp = max(X(:,j));
        min_tmp = min(X(:,j));
        if max_tmp == min_tmp 
            denom = 1;
        else
            denom = max_tmp - min_tmp;
        end

        X_scaled(:, j) = (X(:,j) - min_tmp)/denom;
    end
end