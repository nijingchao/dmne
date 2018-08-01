%% pretrain model

function [ecs, dcs, Hs] = do_pretrain(As, ec_hlyrs, dc_hlyrs, acttype, usegpu)

%% parameters

batchsize = 200;
maxepoch = 300;
outputtype = 'sigmoid';
stepsizes = {[1e-3, 1e-2]; [1e-3, 1e-2]; [1e-3, 1e-2]; [1e-3, 1e-2]; [1e-3, 1e-2]};

%% pretrain

g = length(As);
ecs = cell(g,1);
dcs = cell(g,1);
Hs = cell(g,1);

t = 0;

for i = 1:g
    A = As{i};
    ec_hlyr = ec_hlyrs{i};
    dc_hlyr = dc_hlyrs{i};
    stepsize = stepsizes{i};
    [ec, dc, H, numbatch, allerrs, pt_t] = pretrain_one_net(A, ec_hlyr, dc_hlyr, ...
        stepsize, batchsize, maxepoch, acttype, outputtype, usegpu);
    ecs{i} = ec;
    dcs{i} = dc;
    Hs{i} = H;
    t = t + pt_t;
end

fprintf('running time: %.2f s.\n', t);

filepath = 'pretrainemb.mat';
save(filepath, 'ecs', 'dcs', 'Hs');

end
