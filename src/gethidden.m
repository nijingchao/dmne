function A = gethidden(A, net)

N=size(A,1);
Nlayers=length(net);

for j=1:Nlayers
    A=[A ones(N,1)]*net{j}.W;
    switch lower(net{j}.type)
      case 'linear',
        % Do nothing.
      case 'relu',
        A(A<0)=0;
      case 'cubic',
        A=nthroot(1.5*A+sqrt(2.25*A.^2+1),3)+nthroot(1.5*A-sqrt(2.25*A.^2+1),3);
        A=real(A);
      case 'sigmoid',
        A=1./(1+exp(-A));
      case 'tanh',
        expa=exp(A); expb=exp(-A);
        A=(expa - expb) ./ (expa + expb);
      case 'logistic',
        if size(F{j}.W,2)~=1
          error('logistic is only used for binary classification\n');
        else
          A=1./(1+exp(-A));
        end
      case 'softmax',
        A=exp(A); s=sum(A,2); A=diag(sparse(1./s))*A;
      otherwise,
        error('Invalid layer type: %s\n',F{j}.type);
    end
end

end