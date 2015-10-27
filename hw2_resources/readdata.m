function [X_out, Y] = readdata(name, type, scale)

    data = importdata(strcat('data/data_',name,'_', type, '.csv'));
 
    [n, p1] = size(data);
    Xtmp = data(:,1:p1-1);
 
    if scale == true
        X_out = scale(Xtmp);
    else
        X_out = Xtmp;
    end
    
        
    Y = data(:,p1);