function plotDecisionBoundary3Class(X, Y, scoreFn, values, mytitle, filename)
% X is data matrix (each row is a data point)
% Y is desired output (1 or -1)
% scoreFn is a function of a data point
% values is a list of values to plot

% Plot the decision boundary. For that, we will asign a score to
% each point in the mesh [x_min, m_max]x[y_min, y_max].
    
mins=min(X)-1;
maxes=max(X)+1;
    
h = max((maxes(1)-mins(1))/200., (maxes(2)-mins(2))/200.);
    
[xx, yy] = meshgrid(mins(1):h:maxes(1), mins(2):h:maxes(2));
    
arr=[xx(:),yy(:)];
zz = zeros(length(arr),3);
for i=1:length(arr),
    zz(i,:) = scoreFn([1, arr(i,:)]); 
end
zz_1=reshape(zz(:,1),size(xx));
zz_2=reshape(zz(:,2),size(xx));
zz_3=reshape(zz(:,3),size(xx));
zz_1_best = zz_1 - max(zz_2, zz_3);
zz_2_best = zz_2 - max(zz_1, zz_3);
zz_3_best = zz_3 - max(zz_1, zz_2);


fig = figure;
hold on;
title(mytitle);
colormap cool
[C,h]=contour(xx, yy, zz_1_best, values);
[C,h]=contour(xx, yy, zz_2_best, values);
[C,h]=contour(xx, yy, zz_3_best, values);
set(h,'ShowText','on');
%Plot the training points
scatter(X(:,1),X(:,2),50,1-Y);

set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig, filename, '-dpdf', '-r0')



    

