%% pretrain one network

function [ec, dc, rbmH, numbatch, allerrs, pt_t] = pretrain_one_net(A, ec_hlyr, dc_hlyr, ...
    stepsize, batchsize, maxepoch, acttype, outputtype, usegpu)

%% initialization

n = size(A,1);
k = ec_hlyr(end);
% A = A/sqrt(trace(A'*A));

% encoder

ec_layers = [n, ec_hlyr];
ec_types = {};
for i = 1:length(ec_hlyr)
    ec_types = [ec_types, {acttype}];
end
% ec_types{end} = 'linear';
ec = deepnetinit(ec_layers, ec_types);

% decoder

dc_layers = [k, dc_hlyr];
dc_types = {};
for i = 1:length(dc_hlyr)
    dc_types = [dc_types, {acttype}];
end
dc_types{end} = outputtype;
dc = deepnetinit(dc_layers, dc_types);

clear ec_types dc_types;

% %% gpu
% 
% gpuidx = 1;
% 
% if gpuDeviceCount > 0
%     fprintf('GPU detected. Trying to use it ...\n');
%     try
%         gpudev = gpuDevice(gpuidx);
%         A = gpuArray(single(full(A)));
%         usegpu = 1;
%         fprintf('Using GPU ...\n');
%     catch
%     end
% end

%% make minibatch

rpidx = randperm(n);
numbatch = ceil(n / batchsize);
batchA = cell(numbatch, 1);

for i = 1:numbatch
    sp = (i - 1) * batchsize + 1;
    ep = min(i * batchsize, n);
    idx = [rpidx(sp:ep), rpidx(1:max(0, i * batchsize - n))];
    batchA{i} = A(idx, :);
end

%% pretraining

prt_ti = tic;

numlyrs = length(ec);
allerrs = cell(numlyrs, 1);

for i = 1:numlyrs
    numhid = ec{i}.units;
    [batchposhidprobs, vishid, visbiases, hidbiases, errs] = rbm(batchA, numhid, maxepoch, 1, usegpu, stepsize(i));
    ec{i}.W = [vishid; hidbiases];
    dc{numlyrs + 1 - i}.W = [vishid'; visbiases];
    allerrs{i} = errs;
    batchA = batchposhidprobs;
end

clear batchA batchposhidprobs vishid hidbiases visbiases;

rbmH = gethidden(A, ec);
if usegpu
    rbmH = gather(rbmH);
    rbmH = double(rbmH);
    for i = 1:numlyrs
        ec{i}.W = gather(ec{i}.W);
        ec{i}.W = double(ec{i}.W);
        dc{i}.W = gather(dc{i}.W);
        dc{i}.W = double(dc{i}.W);
        allerrs{i} = gather(allerrs{i});
        allerrs{i} = double(allerrs{i});
    end
end

pt_t = toc(prt_ti);

end
