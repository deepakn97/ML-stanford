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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%theta1_size = hidden_layer_size * (input_layer_size+1)
%theta2 size = num_labels * (hidden_layer_size+1)


% Setup some useful variables
m = size(X, 1); %training set size
         
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
X = [ones(m,1) X]; %size = m*(input_layer_size+1)

a1 = sigmoid(X*Theta1'); % size = (m*(hidden_layer_size)) 
a1 = [ones(m,1) a1]; % size = (m*(hidden_layer_size+1)) 
a2 = sigmoid(a1*Theta2'); %size = (m*(num_labels))

lh = log(a2);
lh1 = log(1-a2);

Y = zeros(m,num_labels);

for i = 1:num_labels
  Y(:,i) = (y==i);
end

temp = (1/m).*((-Y.*lh) - ((1-Y).*lh1));
J = sum(temp(:));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for t = 1:m
  a1 = X(t,:)';
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  del_3 = a3-Y(t,:)';
  del_2 = Theta2'*del_3.*[1;sigmoidGradient(z2)];
  del_2 = del_2(2:end);
  delta2 = delta2 + del_3*a2';
  delta1 = delta1 + del_2*a1';
end
Theta1_grad = (1/m).*delta1;
Theta2_grad = (1/m).*delta2;
  
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);
reg_param = (lambda/(2*m)).*(sum((theta1.*theta1)(:)) + sum((theta2.*theta2)(:)));

J = J + reg_param;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m).*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m).*Theta2(:,2:end);



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
