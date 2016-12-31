function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
regThetaSize = 2:size(theta,1);

hypoFunc = sigmoid(X*theta);
    
% You need to return the following variables correctly 
J = 0;
Jone = 0;
Jtwo = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = ((-y')*log(hypoFunc) - (1-y')*log(1-hypoFunc) ...
    + (lambda/2)*sum(theta(regThetaSize).^2))/m;

grad = (((hypoFunc - y)'*X)')/m + lambda*theta.*[0; ones(length(theta)-1,1)]/m;

% for row = 1:m
%     J = J + (-y(row))*log(hypoFunc(row))-(1-y(row))*log(1-hypoFunc(row));    
% end;

% for thetaIndex = 1:size(theta,1)
%     
%     for row = 1:m
%         
%         if(thetaIndex == 1)
%             grad(thetaIndex) = grad(thetaIndex) + (hypoFunc(row) - y(row))*X(row,thetaIndex);
%         else
%             grad(thetaIndex) = grad(thetaIndex) + (hypoFunc(row) - y(row))*X(row,thetaIndex) ...
%                 + lambda*theta(thetaIndex,1);
%         end;
%     end;
%     
% end;
    
% grad = grad/m;

end


