for i = 1:m
    
    % feedforward NN
    a1 = [1 X(i,:)];
    
    z2 = a1*Theta1';
    a2 = [1 sigmoid(z2)];
    
    z3 = a2*Theta2';
    a3 = sigmoid(z3);

    J = J + sum(-y_vec(i,:).*log(a3)-(1-y_vec(i,:)).*(log(1-a3)));
    
    % calculate dels
    del3 = (a3 - y_vec(i,:));
    d2 = del3*Theta2;
    del2 = d2(1,2:size(d2,2)).*sigmoidGradient(z2);    % does not select 
                                                       % the bias unit term
    
    % calculate deltas
    delta_1 = delta_1 + del2'*a1;
    delta_2 = delta_2 + del3'*a2(1,2:size(a2,2));

end;

J = J/m;
Theta1_grad = delta_1/m;
Theta2_grad = delta_2/m;