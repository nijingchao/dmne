%% evaluate classification accuracy of one network

function [macfs, micfs] = eval_cls_one_net(emb, label, tr_ratio)

%% initialization

runtime = 100;
macfs = zeros(runtime,1);
micfs = zeros(runtime,1);
samples = find(label>0);
n = length(samples);

%% evaluation

for i = 1:runtime
    
    % sample training data
    
    tr_len = ceil(n * tr_ratio);
    rp_idx = randperm(n);
    tr_idx = samples(rp_idx(1:tr_len));
    tr_idx = unique(tr_idx);
    tst_idx = setdiff(samples, tr_idx);
    tst_idx = unique(tst_idx);
    
    % train svm
    
    tr_model = train(label(tr_idx, :), emb(tr_idx, :), '-q');
    [predlabels, ~, ~] = predict(label(tst_idx, :), emb(tst_idx, :), tr_model, '-q');
    
    % accuracy
    
    [macf, micf] = eval_fscore([], predlabels, label(tst_idx, :));
    macfs(i) = macf;
    micfs(i) = micf;
    
end

end
