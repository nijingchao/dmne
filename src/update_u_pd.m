%% update U in one iteration

function Uinew = update_u_pd(Oir, Oic, Bir, Bic, Us, Hi, ns, i, alpha, beta)

g = length(Us);
num = 0;
denom = 0;
n_i = ns(i);

for j = 1:g
    if i ~= j
        n_j = ns(j);
        
        OBU = Oir{j}*(Bir{j}*Us{j});
        num1 = (OBU')*Us{i};
        num1 = OBU*num1;

        BOU = (Bic{j}')*(Oic{j}*Us{j});
        num2 = (BOU')*Us{i};
        num2 = BOU*num2;

        num = num + 2*alpha*(1/(g-1+eps))*((1/(n_i+eps))*num1 + (1/(n_j+eps))*num2);

        OOU = Oir{j}*Us{i};
        denom1 = (Us{i}')*OOU;
        denom1 = OOU*denom1;

        BBU = (Bic{j}')*(Bic{j}*Us{i});
        denom2 = (Us{i}')*BBU;
        denom2 = BBU*denom2;

        denom = denom + 2*alpha*(1/(g-1+eps))*((1/(n_i+eps))*denom1 + (1/(n_j+eps))*denom2);
    end
end

num = num + beta*(1/(n_i+eps))*Hi;
denom = denom + beta*(1/(n_i+eps))*Us{i};

Uinew = Us{i}.*((num./(denom+eps)).^(0.25));

end
