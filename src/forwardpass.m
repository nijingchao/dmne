%% forward propagation

function hidlyrs = forwardpass(inputdata, net)

%% initialization

n = size(inputdata, 1);
numlyrs = length(net);
hiddata = inputdata;
hidlyrs = cell(1, numlyrs+1);

% % drop out the inputs
% j = 1;
% tmp = rand(size(hiddata));
% hiddata(tmp < dropprob(j))=0;
% hiddata = hiddata / (1 - dropprob(j));
% hidlyrs{j} = hiddata;

hidlyrs{1} = hiddata;

%% feed forward

for j = 1:numlyrs
    
    hiddata = [hiddata, ones(n, 1)] * net{j}.W;
    
    switch lower(net{j}.type)
        
        case 'linear',
            % do nothing
        
        case 'relu',
            hiddata(hiddata < 0) = 0;
        
        case 'cubic',
            hiddata = nthroot(1.5 * hiddata + sqrt(2.25 * hiddata.^2 + 1),3) + nthroot(1.5 * hiddata - sqrt(2.25 * hiddata.^2 + 1), 3);
            hiddata = real(hiddata);
        
        case 'sigmoid',
            hiddata = 1 ./ (1 + exp(-hiddata));
        
        case 'tanh',
            expa = exp(hiddata);
            expb = exp(-hiddata);
            hiddata = (expa - expb) ./ (expa + expb);
        
        case 'logistic',
            if size(net{j}.W,2) ~= 1
                error('logistic is only used for binary classification.\n');
            else
                hiddata = 1 ./ (1 + exp(-hiddata));
            end
            
        case 'softmax',
            hiddata=exp(hiddata); s=sum(hiddata,2); hiddata=diag(sparse(1./s))*hiddata;
        
        otherwise,
            error('invalid layer type: %s.\n', net{j}.type);
    end
    
    hidlyrs{j+1} = hiddata;
    
end

end
