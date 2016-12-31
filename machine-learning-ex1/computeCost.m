function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

for i = 1:m
    
    hypoFunc = theta(1)*X(i,1) + theta(2)*X(i,2);
    J = J + (hypoFunc - y(i))*(hypoFunc - y(i));
    
end;

J = J/(2*m);
