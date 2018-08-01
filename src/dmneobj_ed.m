%% objective value of dmne ed

function obj = dmneobj_ed(A_mats, O_mats, S_nm_mats, Us, weight_vec, ecs, dcs, ns, alpha, beta, lambda)

obj = 0;
g = length(A_mats);

% A

for i = 1:g
    
    H = gethidden(A_mats{i}, ecs{i});
    Ahat = gethidden(H, dcs{i});
    
    H = Us{i} - H;
    obj = obj + beta*(1/(ns(i)+eps))*norm(H, 'fro')^2;
    
    Ahat = A_mats{i} - Ahat;
    obj = obj + (1/(ns(i)+eps))*norm(Ahat, 'fro')^2;
    
end

clear H Ahat;

% Regularization

for i = 1:g
    for j = (i+1):g
        
        OU = O_mats{i,j}*Us{i};
        BU = S_nm_mats{i,j}*Us{j};
        obj_ij = norm((OU-BU), 'fro')^2;
        
        OU = O_mats{j,i}*Us{j};
        BU = S_nm_mats{j,i}*Us{i};
        obj_ji = norm((OU-BU), 'fro')^2;
        
        obj = obj + alpha*(1/(ns(i)+eps))*(1/(g-1+eps))*obj_ij + alpha*(1/(ns(j)+eps))*(1/(g-1+eps))*obj_ji;
        
    end
end

obj = obj + lambda*sum(weight_vec.^2);

end
