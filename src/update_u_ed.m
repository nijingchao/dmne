%% update U in one iteration

function Uinew = update_u_ed(Oir, Oic, Bir, Bic, Us, Hi, ns, i, alpha, beta)

g = length(Us);
num = 0;
denom = 0;
n_i = ns(i);

for j = 1:g
    if i ~= j
        nj = ns(j);
        
        num1 = Oir{j}*(Bir{j}*Us{j});
        num2 = (Bic{j}')*(Oic{j}*Us{j});
        num = num + alpha*(1/(g-1+eps))*((1/(n_i+eps))*num1 + (1/(nj+eps))*num2);
        
        denom1 = Oir{j}*Us{i};
        denom2 = (Bic{j}')*(Bic{j}*Us{i});
        denom = denom + alpha*(1/(g-1+eps))*((1/(n_i+eps))*denom1 + (1/(nj+eps))*denom2);
    end
end

num = num + beta*(1/(n_i+eps))*Hi;
denom = denom + beta*(1/(n_i+eps))*Us{i};

Uinew = Us{i}.*((num./(denom+eps)).^(0.5));

end
