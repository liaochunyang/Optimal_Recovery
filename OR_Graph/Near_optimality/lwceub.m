function lwce = lwceub(epsilon,eta,L,n,Lambda,lab,Delta,Q)

% This function is used to compute an upper bound of the global worst-case
% error produced by z=Delta*lab.

z = Delta*lab;
% compute local worst-case error by using semidefinite relaxation
cvx_begin quiet
variable c1 nonnegative
variable c2 nonnegative
variable t nonnegative
minimize t
subject to
[c1*full(L) + c2*Lambda'*Lambda - Q'*Q, Q'*z-c2*Lambda'*lab;
-c2*lab'*Lambda+z'*Q, -c1*epsilon^2+c2*(norm(lab)^2-eta^2)+t-norm(z)^2] == semidefinite(n+1)
cvx_end

% local worst-case error
lwce = sqrt(t);
end