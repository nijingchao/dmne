%% row normalization of a matrix

function A = mat_row_norm(A)

A = A - diag(diag(A));
A = bsxfun(@rdivide, A, sum(A, 2) + eps);

end
