%% generate the adjacency matrix of a network

function [net_mat, nodes] = gen_net(edges_u, edges_v, weights)

nodes = [edges_u; edges_v];
nodes = unique(nodes);
n = length(nodes);

edges_u_idx = zeros(length(edges_u), 1);
for i = 1:length(nodes)
    idx = edges_u == nodes(i);
    edges_u_idx(idx) = i;
end

edges_v_idx = zeros(length(edges_v), 1);
for i = 1:length(nodes)
    idx = edges_v == nodes(i);
    edges_v_idx(idx) = i;
end

net_mat = sparse(edges_u_idx, edges_v_idx, weights, n, n);

end
