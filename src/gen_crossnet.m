%% generate the adjacency matrix of the cross-network relationships

function crossnet_mat = gen_crossnet(net_1_edges, net_2_edges, weights, net_1_nodes, net_2_nodes)

n_1 = length(net_1_nodes);
n_2 = length(net_2_nodes);

net_1_edges_idx = zeros(length(net_1_edges), 1);
for i = 1:length(net_1_nodes)
    idx = net_1_edges == net_1_nodes(i);
    net_1_edges_idx(idx) = i;
end

net_2_edges_idx = zeros(length(net_2_edges), 1);
for i = 1:length(net_2_nodes)
    idx = net_2_edges == net_2_nodes(i);
    net_2_edges_idx(idx) = i;
end

crossnet_mat = sparse(net_1_edges_idx, net_2_edges_idx, weights, n_1, n_2);

end
