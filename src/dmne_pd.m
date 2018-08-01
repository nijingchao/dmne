%% dmne pd

function [ecs, dcs, Us, Hs, objvals, ft_t] = dmne_pd(G_mats, O_mats, S_nm_mats, alpha, beta, lambda, ...
    batchsize, stepsize, momentum, maxepoch, decay, acttype, gpuidx, ec_hlyrs, dc_hlyrs)

%% initialization

g = length(G_mats);
ns = cellfun(@length, G_mats);
batchsizes = batchsize * ones(g,1);
maxepoch1 = 1;
maxepoch2 = 5;
iter = 1;
delta = 0;
objvals = [];
usegpu = 0;

%% extract feature

for i = 1:g
    Gtmp = G_mats{i};
    Gtmp = randwalk(Gtmp, 3, 0.98);
    Gtmp = mat_ppmi(Gtmp);
    G_mats{i} = Gtmp;
end
clear Gtmp;

%% gpu

if gpuDeviceCount > 0
    fprintf('gpu detected, trying to use it ...\n');
    try
        gpudev = gpuDevice(gpuidx);
        for i = 1:g
            G_mats{i} = gpuArray(single(full(G_mats{i})));
        end
        usegpu = 1;
        fprintf('using gpu ...\n');
    catch
    end
end

%% pretraining

ecs = cell(g,1);
dcs = cell(g,1);

if exist('pretrainemb.mat', 'file') == 2
    fprintf('a pretrained model is detected.\n');
    load('pretrainemb.mat');
    Us = Hs;
else
    fprintf('do pretraining ...\n');
    [ecs, dcs, Hs] = do_pretrain(G_mats, ec_hlyrs, dc_hlyrs, acttype, usegpu);
    Us = Hs;
end

numbatches = ceil(ns ./ batchsizes);

%% concatenate weights

weight_vec = [];
ec_nets = cell(g, 1);
ec_cuts = zeros(g+1, 1);
dc_nets = cell(g, 1);
dc_cuts = zeros(g+1, 1);

% encoder

for i = 1:g
    ec_i = ecs{i};
    numlyrs = length(ec_i);
    ec_net_i = cell(1, numlyrs);
    for j = 1:numlyrs
      weight_vec = [weight_vec; ec_i{j}.W(:)];
      ec_net_i{j} = rmfield(ec_i{j}, 'W');
    end
    ec_nets{i} = ec_net_i;
    ec_cuts(i+1) = length(weight_vec);
end

% decoder

dc_cuts(1) = ec_cuts(g+1);

for i = 1:g
    dc_i = dcs{i};
    numlyrs = length(dc_i);
    dc_net_i = cell(1, numlyrs);
    for j = 1:numlyrs
      weight_vec = [weight_vec; dc_i{j}.W(:)];
      dc_net_i{j} = rmfield(dc_i{j},'W');
    end
    dc_nets{i} = dc_net_i;
    dc_cuts(i+1) = length(weight_vec);
end

% make weight gpuarray

if usegpu
    if ~isa(weight_vec, 'gpuArray')
        weight_vec = gpuArray(single(weight_vec));
    end
end

%% learning parameters

fprintf('dmne starts ...\n');

% objective value

objval = dmneobj_pd(G_mats, O_mats, S_nm_mats, Us, weight_vec, ecs, dcs, ns, alpha, beta, lambda);
objvals = [objvals; objval];

maxnumbatch = max(numbatches);

ft_ti = tic;

while iter <= maxepoch
    
    iter1 = 1;
    iter2 = 1;
    
    if usegpu && beta>0
        for i = 1:g
            U_i = Us{i};
            U_i = gpuArray(single(U_i));
            Us{i} = U_i;
        end
    end
    clear U_i;
    
    % update weight
    
    while iter1 <= maxepoch1
        
        % reduce learning rate
        eta = stepsize * decay^(iter-1);
        
        rpidxes = cell(g,1);
        for i = 1:g
            rpidxes{i} = randperm(ns(i));
        end
        
        grad = zeros(length(weight_vec), 1);
        if usegpu
            grad = gpuArray(single(grad));
        end
        
        for i = 1:maxnumbatch
            
            for j = 1:g
                
                if i <= numbatches(j)
                    
                    sp = (i - 1) * batchsizes(j) + 1;
                    ep = min(i * batchsizes(j), ns(j));
                    rpidxi = rpidxes{j};
                    idxj = [rpidxi(sp:ep), rpidxi(1:max(0, i * batchsizes(j) - ns(j)))];
                    
                    % reconstruction gradient
                    
                    batchAj = G_mats{j}(idxj, :);
                    
                    grada = recon_grad(batchAj, [ecs{j}, dcs{j}]);
                    grada = grada / (batchsizes(j) + eps);
                    
                    ec1_cut = ec_cuts(j);
                    ec2_cut = ec_cuts(j+1);
                    grad(ec1_cut+1:ec2_cut) = grad(ec1_cut+1:ec2_cut) + grada(1:ec2_cut-ec1_cut);
                    
                    dc1_cut = dc_cuts(j);
                    dc2_cut = dc_cuts(j+1);
                    grad(dc1_cut+1:dc2_cut) = grad(dc1_cut+1:dc2_cut) + grada(ec2_cut-ec1_cut+1:end);
                    
                    % regularization gradient
                    
                    if beta > 0
                        
                        batchUj = Us{j}(idxj,:);
                        gradr = dmne_reg_grad(batchAj, batchUj, ecs{j}, beta);
                        gradr = gradr / (batchsizes(j) + eps);
                        grad(ec1_cut+1:ec2_cut) = grad(ec1_cut+1:ec2_cut) + gradr;
                        
                    end
                    
                end
                
            end
            
        end
        
        clear batchAj batchUj grada gradr;
        
        % gradient descent
        
        grad = grad + 2 * lambda * weight_vec;
        delta = momentum * delta - eta * grad;
        weight_vec = weight_vec + delta;
        
        % reshape weight
        
        for i = 1:g
            weight_vec_i = weight_vec([ec_cuts(i)+1:ec_cuts(i+1), dc_cuts(i)+1:dc_cuts(i+1)]);
            [ec_i, dc_i] = reshapeweight(weight_vec_i, ec_nets{i}, dc_nets{i}, ns(i));
            ecs{i} = ec_i;
            dcs{i} = dc_i;
        end
        clear ec_i dc_i weight_vec_i;
        
        iter1 = iter1 + 1;
        
    end
    
    % update U
    
    Hs = cell(g,1);
    
    for i = 1:g
        
        H = gethidden(G_mats{i}, ecs{i});
        
        if usegpu && beta>0
            U = gather(Us{i});
            U = double(U);
            H = gather(H);
            H = double(H);
            Us{i} = U;
        end
        
        Hs{i} = H;
        
    end
    clear U H;
    
    while iter2 <= maxepoch2
        
        for i = 1:g
            U_i = update_u_pd(O_mats(i,:), O_mats(:,i), S_nm_mats(i,:), S_nm_mats(:,i), Us, Hs{i}, ns, i, alpha, beta);
            Us{i} = U_i;
        end
        
        iter2 = iter2 + 1;
        
    end
    
    % objective value
    
    objval = dmneobj_pd(G_mats, O_mats, S_nm_mats, Us, weight_vec, ecs, dcs, ns, alpha, beta, lambda);
    objvals = [objvals; objval];
    
    iter = iter + 1;
    
end

fprintf('running time: %.2f s.\n', toc(ft_ti));
ft_t = toc(ft_ti);

% figure;plot(objs(1:end));

if usegpu
    objvals = gather(objvals);
    objvals = double(objvals);
    for i = 1:g
        ec_i = ecs{i};
        dc_i = dcs{i};
        for j = 1:length(ec_i)
            ec_i{j}.W = gather(ec_i{j}.W);
            ec_i{j}.W = double(ec_i{j}.W);
            dc_i{j}.W = gather(dc_i{j}.W);
            dc_i{j}.W = double(dc_i{j}.W);
        end
        ecs{i} = ec_i;
        dcs{i} = dc_i;
    end
end

end
