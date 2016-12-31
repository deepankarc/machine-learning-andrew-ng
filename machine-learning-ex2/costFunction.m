function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta))';

for row = 1:m
        hypoFunc = sigmoid(X*theta);      
        
        J = J + (-y(row))*log(hypoFunc(row))-(1-y(row))*log(1-hypoFunc(row));
        grad = grad + (hypoFunc(row) - y(row))*X(row,:);
end;

J = J/m;
grad = grad/m;

end
