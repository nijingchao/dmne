%% initialize the weights of the network

function net = deepnetinit(lyrs, lyrtypes, decays)

% input:
%
%   lyrs: vector containing number of units at each layer.
%
%   lyrtypes: type of activations at each layer. Possible types include
%             'linear', 'sigmoid', 'tanh', 'relu', 'cubic', 'logistic', 'softmax'.
%
%   decays: vector of weight decay parameters (l2 regularization) of 
%           weights at each layer.
%
% output:
%
%   net: cell array that contains all layers of the network. Each layer
%        has a field 'type' indicating the type of hidden activation, a field
%        'units' indicating the output dimension of the layer, a filed 'l'
%        indicating the weight decay parameter, and a field 'W' containing the
%        weight matrix.
%

numlyrs = length(lyrs) - 1;

if ~exist('decays', 'var') || isempty(decays)
    decays = zeros(1, numlyrs);
end

if length(lyrtypes) ~= numlyrs
    error('the lyrtypes has a different length from lyrs.');
end

if length(decays) ~= numlyrs
    error('the weight decay parameters has a different length from lyrs.');
end

net = cell(1, numlyrs);

for j = 1:numlyrs
    
    layer.type = lyrtypes{j};
    fan_in = lyrs(j);
    fan_out = lyrs(j+1);
    layer.units = fan_out;
    layer.W = zeros(fan_in+1, fan_out);
    
    switch layer.type
        case 'tanh'
            % suggested by Yoshua Bengio, normalized initialization
            layer.W(1:end-1, :) = 2 * (rand(fan_in, fan_out) - 0.5) * sqrt(6) / sqrt(fan_in + fan_out);
        
        case 'cubic'
            % suggested by Yoshua Bengio, normalized initialization
            layer.W(1:end-1, :) = 2 * (rand(fan_in, fan_out) - 0.5) * sqrt(6) / sqrt(fan_in + fan_out);
        
        case 'relu'
            layer.W(1:end-1, :) = 2 * (rand(fan_in, fan_out) - 0.5) * 0.01;
            % sqrt(6) / sqrt(fan_in + fan_out);
            % give some small postive bias so that initial activation is nonzero
            layer.W(end, :) = rand(1, fan_out) * 0.1;
        
        case 'sigmoid'
            % suggested by Yoshua Bengio, 4 times bigger than tanh
            layer.W(1:end-1, :) = 8 * (rand(fan_in, fan_out) - 0.5) * sqrt(6) / sqrt(fan_in + fan_out);
        
        otherwise
            % the 1/sqrt(fan_in) rule, small random values
            layer.W(1:end-1, :) = 2 * (rand(fan_in, fan_out) - 0.5) / sqrt(fan_in);
    end
    
    net{j} = layer;
    
end
