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

% Setup some useful variables
m = size(X, 1);
         
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Add ones to the X data matrix
X = [ones(m, 1) X];

% to recode y
eye10 = eye(num_labels,num_labels);

% accumulate J
for i = 1 : m
    % a1 = input
    a1 = X(i,:);
    a1 = a1'; % make it a column vector

    % compute a2
    a2 = sigmoid(Theta1 * a1);

    % add bias entry a2
    a2 = [ones(1,1) a2']';

    % compute a3
    a3 = sigmoid(Theta2 * a2);
    
    % code y as vector
    recodedY = eye10(:, y(i));

    % cost for this example
    exampleCost = sum(-recodedY .* log(a3) - (1 - recodedY) .* log(1 - a3));

    J = J + exampleCost;
 
end

J = J / m;

% lop off first column cause it is the bias terms which we don't regularize
Theta1Trimmed = Theta1(:, 2:end);
Theta2Trimmed = Theta2(:, 2:end);

% compute regularization term
regTerm = (lambda / (2 * m)) * ( sum(sum( Theta1Trimmed .* Theta1Trimmed)) + sum(sum(Theta2Trimmed .* Theta2Trimmed)) );

% add it to cost
J = J + regTerm;



% back prop 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t = 1 : m
    % feed forward
    % a1 = input
    a1 = X(t,:);
    a1 = a1'; % make it a column vector

    % compute a2 (bias units already added to X) 
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);

    % add bias entry a2
    a2 = [ones(1,1) a2']';

    % compute a3
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
 
    % code y as vector
    recodedY = eye10(:, y(t));

    % feed back, first error term
    delta3 = (a3 - recodedY);

    % second error term, ignore bias unit
    delta2 = (Theta2' * delta3)(2:end,:) .* sigmoidGradient(z2);

    % accumulate grad
    Theta1_grad = Theta1_grad + delta2 * a1';
    Theta2_grad = Theta2_grad + delta3 * a2';

end

% no regularization on first bias terms
Theta1_grad(:,1) = Theta1_grad(:,1) / m;
Theta2_grad(:,1) = Theta2_grad(:,1) / m;

% regularization on following terms
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) / m + Theta1(:,2:end) * (lambda / m);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) / m + Theta2(:,2:end) * (lambda / m);;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
