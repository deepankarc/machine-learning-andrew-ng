function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Uncomment for general implementation for N-layers
% for i = 1:num_layers
%     curr_layer_size = layer_size(i);
%     
%     if(i ~= 1)
%         prev_layer_size = layer_size(i-1);
%         Theta(i) = reshape(nn_params((i-1)*prev_layer_size(i)+1:i*curr_layer_size(i)));
%     else
%         Theta(1) = reshape(1:i*curr_layer_size(i));
%     end;
%     
% end;

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
y_vec = zeros(m,num_labels);
for i = 1:num_labels
    y_vec(:,i) = (y==i);
end;
delta_1 = zeros(size(Theta1,1),size(Theta1,2));
delta_2 = zeros(size(Theta2,1),size(Theta2,2));
         
% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

a1 = [ones(m,1) X];     % adds bias values

z2 = a1*Theta1';
a2 = [ones(m,1) sigmoid(z2)];    % computes value of hidden layer

z3 = a2*Theta2';
a3 = sigmoid(z3);    % computes value of output layer

J = sum(sum(-(y_vec).*log(a3)-(1-y_vec).*log(1-a3)))/m;

regTheta1 = sum(sum((Theta1(:,2:input_layer_size+1).*Theta1(:,2:input_layer_size+1))));
regTheta2 = sum(sum(Theta2(:,2:hidden_layer_size+1).*Theta2(:,2:hidden_layer_size+1)));

J = J + (regTheta1+regTheta2)*(lambda/(2*m));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%

% calculation of del values
del_3 = a3 - y_vec;
d_2 = (del_3*Theta2);
del_2 = d_2(:,2:end).*sigmoidGradient(z2);

% calculation of intermediate delta values

delta_1 = (delta_1 + del_2'*a1);
delta_2 = (delta_2 + del_3'*a2);

% gradient of the cost function wrt thetas

Theta1_grad = delta_1/m;
Theta2_grad = delta_2/m;

% regularization with the cost function and gradients

Theta1_grad = Theta1_grad + (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
