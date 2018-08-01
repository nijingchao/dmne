%% run back propagation

function grad = backwardpass(delta, hidlyrs, net)

numlyrs = length(net);

if numlyrs == 0
    grad = [];
else
    j = numlyrs;
    n = size(delta, 1);
%     delta = delta / (1 - dropprob(j+1));
%     hiddata = hidlyrs{j+1} * (1 - dropprob(j+1));
%     delta(hiddata == 0) = 0;
    hiddata = hidlyrs{j+1};
    
    switch lower(net{j}.type)
        
        case 'linear',
            % do nothing
        
        case 'relu',
            delta(hiddata <= 0) = 0;
        
        case 'cubic',
            delta = delta ./ (1 + hiddata.^2);
        
        case 'sigmoid',
            delta = delta .* hiddata .* (1 - hiddata);
        
        case 'tanh',
            delta = delta .* (1 - hiddata.^2);
        
        case 'logistic',
            delta = delta .* hiddata .* (1 - hiddata);
        
        case 'softmax',
            delta = delta .* hiddata - repmat(sum(delta .* hiddata, 2), 1, size(hiddata, 2)) .* hiddata;
        
        otherwise,
            error('invalid layer type: %s.\n', net{j}.type);
    end
    
    de = [hidlyrs{j}, ones(n,1)]' * delta;
%     de(1:end-1, :) = de(1:end-1, :) + 2 * net{j}.l * net{j}.W(1:end-1, :);
    grad = de(:);
    
    % delta for next layer
    delta = delta * (net{j}.W(1:end-1, :)');
    
    % other layers
    for j = numlyrs-1:-1:1
        
        hiddata = hidlyrs{j+1};
%         delta = delta / (1 - dropprob(j+1));
%         hiddata = hiddata * (1 - dropprob(j+1));
%         delta(hiddata == 0) = 0;
        
        switch lower(net{j}.type)
            
            case 'linear',
                % do nothing
            
            case 'relu',
                delta(hiddata <= 0) = 0;
            
            case 'cubic',
                delta = delta ./ (1 + hiddata.^2);
            
            case 'sigmoid',
                delta = delta .* hiddata .* (1 - hiddata);
            
            case 'tanh',
                delta = delta .* (1 - hiddata.^2);
            
            case 'logistic',
                delta = delta .* hiddata .* (1 - hiddata);
            
            otherwise,
                error('invalid layer type: %s.\n', net{j}.type);
        end
        
        de = [hidlyrs{j}, ones(n, 1)]' * delta;
%         de(1:end-1, :) = de(1:end-1, :) + 2 * net{j}.l * net{j}.W(1:end-1, :);
        grad = [de(:); grad];
        
        % delta for next layer
        delta = delta * (net{j}.W(1:end-1, :)');
        
    end
    
end
