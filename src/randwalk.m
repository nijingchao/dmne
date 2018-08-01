%% extract structural context by random walk

function A = randwalk(G, maxstep, alpha)

num_nodes = length(G);
G = mat_row_norm(G);

P = eye(num_nodes, num_nodes);
P_new = P;
A = zeros(num_nodes, num_nodes);

for i = 1:maxstep
    P_new = alpha * P_new * G + (1 - alpha) * P;
    A = A + P_new;
end

end
