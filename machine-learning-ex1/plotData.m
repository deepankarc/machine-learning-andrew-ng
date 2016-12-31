function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

plot(x,y,'rx','MarkerSize',5);
title('City Pop. vs Profits');
xlabel('Pop. of City in 10,000s');
ylabel('Revenue in $10,000s');

%figure; % open a new figure window

end
