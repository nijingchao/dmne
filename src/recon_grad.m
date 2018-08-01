%% reconstruction gradient

function grad = recon_grad(A, net)

%% feed forward

hidlyrs = forwardpass(A, net);

%% delta

delta = 2 * (hidlyrs{end} - A);

%% backpropagation

grad = backwardpass(delta, hidlyrs, net);

end
