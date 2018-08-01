%% evaluate classification accuracy

function [allmacfs, allmicfs] = eval_cls(embs, labels, tr_ratio)

g = length(embs);
allmacfs = [];
allmicfs = [];

addpath('libsvm/matlab');
% addpath('libsvm/matlab');

for i = 1:g
    emb_i = sparse(embs{i});
    label_i = labels{i};
    [macfs, micfs] = eval_cls_one_net(emb_i, label_i, tr_ratio);
    allmacfs = [allmacfs; macfs];
    allmicfs = [allmicfs; micfs];
end

end
