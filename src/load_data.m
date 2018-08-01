%% load data

function [nets, crossnets, labels] = load_data(datadir, netidx, crossnetidx, labelidx)

%% initialization

g = length(netidx);
nets = cell(g, 1);
nodeids = cell(g, 1);
crossnets = cell(g, g);
labels = cell(g, 1);

%% load networks and labels

for i = 1:length(netidx)
    idx = netidx(i);
    datapath = [datadir 'net/net_' num2str(idx) '.txt'];
    fid = fopen(datapath, 'r');
    net_i = textscan(fid, '%f %f %f');
    fclose(fid);
    
    edges_u = net_i{1};
    edges_v = net_i{2};
    weights = net_i{3};
    
    [net_mat_i, nodes_i] = gen_net(edges_u, edges_v, weights);
    nets{i} = net_mat_i;
    nodeids{i} = nodes_i;
end

%% load labels

for i = 1:length(labelidx)
    idx = labelidx(i);
    datapath = [datadir 'label/label_' num2str(idx) '.txt'];
    fid = fopen(datapath, 'r');
    labels_i = textscan(fid, '%f %f');
    fclose(fid);
    
    label_nodes = labels_i{1};
    labels_i = labels_i{2};
    
    nodes_i = nodeids{i};
    [~, memidx] = ismember(nodes_i, label_nodes);
    if any(memidx == 0) == 0
        labels_new = labels_i(memidx);
    else
        error('label node ids do not match network node ids.');
    end
    
    labels_i = labels_new;
    labels{i} = labels_i;
end

%% load cross-network relationships

for i = 1:length(crossnetidx)
    idx = crossnetidx{i};
    idx_1 = idx(1);
    idx_2 = idx(2);
    datapath = [datadir 'crossnet/crossnet_' num2str(idx_1) '_' num2str(idx_2) '.txt'];
    fid = fopen(datapath, 'r');
    crossnet_i = textscan(fid, '%f %f %f');
    fclose(fid);
    
    net_1_edges = crossnet_i{1};
    net_2_edges = crossnet_i{2};
    weights = crossnet_i{3};
    
    net_1_nodes = nodeids{idx_1};
    net_2_nodes = nodeids{idx_2};
    
    crossnet_mat_i = gen_crossnet(net_1_edges, net_2_edges, weights, net_1_nodes, net_2_nodes);
    crossnets{idx_1, idx_2} = crossnet_mat_i;
end

end
