function k = rbf(x, z, sigma2)
    k = exp(-(x-z)*(x-z)' / (2*sigma2));
end