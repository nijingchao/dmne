%% reshape the weights

function [ec_net, dc_net] = reshapeweight(weight_vec, ec_net, dc_net, d)

idx = 0;

% encoder

for i = 1:length(ec_net)
    units = ec_net{i}.units;
    w_seg = weight_vec(idx+1:idx+(d+1)*units);
    ec_net{i}.W = reshape(w_seg, d+1, units);
    idx = idx + (d+1)*units;
    d = units;
end

% decoder

d = ec_net{end}.units;
for i = 1:length(dc_net)
    units = dc_net{i}.units;
    w_seg = weight_vec(idx+1:idx+(d+1)*units);
    dc_net{i}.W = reshape(w_seg, d+1, units);
    idx = idx + (d+1)*units;
    d = units;
end

end
