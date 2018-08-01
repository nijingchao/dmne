function [macrof1, microf1] = eval_fscore(V, V_idx, G_Labels)

%% Initialization

if isempty(V_idx)
    
    [Vals, V_idx] = max(V, [], 2);
    
end

%% TP, FP, FN

NonNoise_idx = G_Labels > 0;
G_Labels = G_Labels(NonNoise_idx);
V_idx = V_idx(NonNoise_idx);

V_Labels = unique(V_idx);
K = length(V_Labels);

TPs = zeros(K,1);
FPs = zeros(K,1);
FNs = zeros(K,1);

for i = 1:K
    
    nodes_i_gd = find(G_Labels == V_Labels(i));
    nodes_i_pd = find(V_idx == V_Labels(i));
    
    % TP
    
    TPi = intersect(nodes_i_gd, nodes_i_pd);
    TPs(i) = length(TPi);
    
    % FP
    
    FPi = setdiff(nodes_i_pd, nodes_i_gd);
    FPs(i) = length(FPi);
    
    % FN
    
    FNi = setdiff(nodes_i_gd, nodes_i_pd);
    FNs(i) = length(FNi);
    
end

%% Macro-F1

macrof1 = (2*TPs)./(2*TPs + FPs + FNs + eps);
macrof1 = sum(macrof1)/(K + eps);

%% Micro-F1

microf1 = (2*sum(TPs))/(2*sum(TPs) + sum(FPs) + sum(FNs) + eps);

end