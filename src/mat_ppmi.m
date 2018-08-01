%% pointwise positive mutual information of a matrix

function A_ppmi = mat_ppmi(A)

A = mat_row_norm(A);

[p, q] = size(A);

if p ~= q
    error('p and q should be equal.');
else
    col_sum = sum(A, 1);
    row_sum = sum(A, 2);
    all_sum = sum(col_sum);
    A_ppmi = log((all_sum * A ./ ((row_sum * col_sum) + eps)) + eps);
    A_ppmi(A_ppmi < 0) = 0;
    idx = isnan(A_ppmi);
    A_ppmi(idx) = 0;
end

end
