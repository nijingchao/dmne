%% run dmne

function rundmne(datadir, alpha, beta, lambda, batchsize, stepsize, momentum, ...
    maxepoch, decay, acttype, gpuidx, tr_ratio, vis, dmnetype)

% input:
%
% datadir: the directory of input data
% alpha, beta, lambda: model parameters
% batchsize: batch size
% stepsize: step size for gradient descent
% momentum: momentum for gradient descent
% maxepoch: the maximal number of epochs
% decay: decay factor of gradient descent
% acttype: the type of activation function
% gpuidx: the index of gpu
% tr_ratio: the ratio of training data for multi-label classification
% vis: indicator for visualization
% dmnetype: the type of dmne algorithm, pd or ed
%
% reference:
%
% Co-Regularized Deep Multi-Network Embedding
% Jingchao Ni, Shiyu Chang, Xiao Liu, Wei Cheng, Haifeng Chen, Dongkuan Xu and Xiang Zhang
% Proceedings of the International Conference on World Wide Web (WWW), 2018.
%
% copyright (c) 2018 Jingchao Ni
% contact: jingchaoni@psu.edu
%

%% parameters

if ~exist('datadir', 'var') || isempty(datadir)
    datadir = '../dataset/6ng/';
end
if ~exist('alpha', 'var') || isempty(alpha)
    alpha = 1;
end
if ~exist('beta', 'var') || isempty(beta)
    beta = 1;
end
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1e-4;
end
if ~exist('batchsize', 'var') || isempty(batchsize)
    batchsize = 200;
end
if ~exist('stepsize', 'var') || isempty(stepsize)
    stepsize = 0.1;
end
if ~exist('momentum', 'var') || isempty(momentum)
    momentum = 0.9;
end
if ~exist('maxepoch', 'var') || isempty(maxepoch)
    maxepoch = 200;
end
if ~exist('decay', 'var') || isempty(decay)
    decay = 1;
end
if ~exist('acttype', 'var') || isempty(acttype)
    acttype = 'sigmoid';
end
if ~exist('gpuidx', 'var') || isempty(gpuidx)
    gpuidx = 1;
end
if ~exist('tr_ratio', 'var') || isempty(tr_ratio)
    tr_ratio = 0.9;
end
if ~exist('vis', 'var') || isempty(vis)
    vis = 1;
end
if ~exist('dmnetype', 'var') || isempty(dmnetype)
    dmnetype = 'pd';
end

%% load dataset

netidx = 1:5;
crossnetidx = {[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]};
labelidx = 1:5;
[G_mats, S_mats, labels] = load_data(datadir, netidx, crossnetidx, labelidx);

g = length(G_mats);
O_mats = cell(g,g);
S_nm_mats = cell(g,g);

for i = 1:g
    for j = (i+1):g
        S = S_mats{i,j};
        O_mats{i,j} = diag(any(S,2));
        O_mats{j,i} = diag(any(S,1)');
        S_nm_mats{i,j} = bsxfun(@rdivide, S, sum(S,2)+eps);
        S_nm_mats{j,i} = bsxfun(@rdivide, S, sum(S,1)+eps);
        S_nm_mats{j,i} = S_nm_mats{j,i}';
    end
end

% network dimensions

k = 100;
ns = cellfun(@length, G_mats);
ec_hlyrs = {[200, k], [200, k], [200, k], [200, k], [200, k]};
dc_hlyrs = {[200, ns(1)], [200, ns(2)], [200, ns(3)], [200, ns(4)], [200, ns(5)]};

%% run dmne algorithm

if strcmp(dmnetype, 'ed') == 1
    [ecs, dcs, Us, Hs, objs, ft_t] = dmne_ed(G_mats, O_mats, S_nm_mats, alpha, beta, lambda, ...
        batchsize, stepsize, momentum, maxepoch, decay, acttype, gpuidx, ec_hlyrs, dc_hlyrs);
elseif strcmp(dmnetype, 'pd') == 1
    [ecs, dcs, Us, Hs, objs, ft_t] = dmne_pd(G_mats, O_mats, S_nm_mats, alpha, beta, lambda, ...
        batchsize, stepsize, momentum, maxepoch, decay, acttype, gpuidx, ec_hlyrs, dc_hlyrs);
else
    error('dmne type can be ed or pd.');
end

embs = Us;
for i = 1:length(embs)
    embs{i} = embs{i} ./ ( repmat( sqrt(sum(embs{i}.^(2), 2)), 1, size(embs{i}, 2)) + eps);
end

%% write embeddings
fprintf('write embeddings ...\n');
destdir = 'emb/';
write_emb(embs, destdir, dmnetype);

%% evaluation

fprintf('evaluation of classification performance ...\n');

filepath = ['cls/cls_results_' dmnetype '.txt'];
fid = fopen(filepath, 'w');

[allmacfs, allmicfs] = eval_cls(embs, labels, tr_ratio);

fprintf(fid, 'training ratio: %.4f, macro-f1: %.4f +- %.4f, micro-f1: %.4f +- %.4f\n', ...
    tr_ratio, mean(allmacfs), var(allmacfs), mean(allmicfs), var(allmicfs));
fprintf('training ratio: %.4f, macro-f1: %.4f +- %.4f, micro-f1: %.4f +- %.4f\n', ...
    tr_ratio, mean(allmacfs), var(allmacfs), mean(allmicfs), var(allmicfs));

fclose(fid);

%% visualization

if vis == 1
    fprintf('visualization using tsne ...\n');
    eva_tsne(embs{1}, labels{1}, dmnetype);
end

end
