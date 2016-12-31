function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

theta_dim = size(theta);
rows = theta_dim(1,1);
cols = theta_dim(1,2);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%

%% REGULARISED COST FUNCTION J OF LOGISTIC REGRESSION

%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%

z = X*theta;    %argument of sigmoid function
g = sigmoid(z);     %sigmoid function

%J = sum((-y.*log(g))-((1-y).*log(1-log(g))))/m;
J = (sum((-y).*log(g)-((1-y).*log(1-g))) ...
        + (lambda/2)*(theta(2:rows,1)'*theta(2:rows,1)))/m;    %cost function
        

%% REGULARISED GRADIENT OF COST FUNCTION

% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)

%calculate gradient of cost function
grad = ((g-y)'*X)/m + (lambda/m)*[0; theta(2:rows,1)]'; 

%X = X(1:size(X,1),2:size(X,2)); %reset size of X
% =============================================================

grad = grad(:);

end
