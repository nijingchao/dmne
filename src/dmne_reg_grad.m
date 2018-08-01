%% regularization gradient

function grad = dmne_reg_grad(A, U, net, beta)

%% feed forward

hidlyrs = forwardpass(A, net);

%% delta

H = hidlyrs{end};
delta = 2 * beta * (H - U);

%% Backpropagation

grad = backwardpass(delta, hidlyrs, net);

end
