%% Section 4.4: Global optimal recovery in finite-dimensional Hilbert Spaces
% Recover linear quantity of interest Q
close all; clear;
N = 50;                        % ambient dimension
n = 15;                        % dimension of linear space V              
V = rand(N,n);                 % linear subspace V          
P1 = eye(N) - V*inv(V'*V)*V';  % orthogonal projector onto V^\perp  
epsilon = 0.4;                 % approximability parameter      
m = 25;                        % number of observations  
L = randn(m,N);                % observation map
%L = (sqrtm(L*L'))\L;          % comment out this line to make orthogonal observations
eta = 0.3;                     % data error

% Remark: When L is orthogonal, we prove the following result. However,
% numerical experiments also reveals that the result holds for arbitrary L. 
% The proof is left for future work.

% linear quantity of interest
s = 5;
Q = rand(s,N);

% compute lower bound and optimal regularization parameter
cvx_begin quiet
variable c nonnegative
variable d nonnegative
minimize c*epsilon^2 + d*eta^2
subject to
c*P1 + d*L'*L - Q'*Q == semidefinite(N);
cvx_end
lower_bound = sqrt(cvx_optval);
tau_opt = d/(c+d);

% compute upper bound for each regularization parameter
T = 100;
range_tau = linspace(0.01,0.99,T);
upper_bound = zeros(1,T);
for idx=1:T
    tau = range_tau(idx);
    Delta_tau = tau*inv((1-tau)*P1+tau*L'*L)*L';
    cvx_begin quiet
    variable c nonnegative
    variable d nonnegative
    minimize c*epsilon^2 + d*eta^2
    subject to
    [c*P1, zeros(N,m); zeros(m,N), d*eye(m)] - ...
    [eye(N)-L'*Delta_tau';Delta_tau']*Q'*Q*[eye(N)-Delta_tau*L,Delta_tau] == semidefinite(N+m)
    cvx_end
    upper_bound(idx) = sqrt(cvx_optval);
end

% Plot
p = plot(range_tau,upper_bound,'ro-',range_tau,lower_bound*ones(1,T),'b-');
set(p,{'LineWidth'},{0.5;2})
legend('upper bounds','lower bound')
title('Bounds on the global worst-case errors of regularization maps')
ylabel('global worst-case error')
ylim([lower_bound-0.01,max(upper_bound)+0.01])
xlabel('regularization parameter')
xticks(tau_opt)
xticklabels('optimal \tau')

%% Full recovery with orthonormal observations
clear all; clc;
% We keep the same notations as previous section.
N = 50;         
n = 15;                    
V = rand(N,n);            
epsilon = 0.4;           
m = 25;                     
L = randn(m,N);            
L = sqrtm(L*L')\L;          
eta = 0.3;                 
P1 = eye(N) - V*inv(V'*V)*V'; 
P2 = L'*L;
C = L*V;
C_linv = (C'*C)\C';
Delta_0 = V*C_linv;
Delta_1 = L' - L'*C*C_linv + V*C_linv; 

% compute lower bound
cvx_begin quiet
variable c nonnegative
variable d nonnegative
minimize c*epsilon^2 + d*eta^2
subject to
c*P1 + d*L'*L - eye(N)== semidefinite(N);
cvx_end
lower_bound = sqrt(cvx_optval);

% compute upper bound
T = 100;
range_tau = linspace(0,1,T);
upper_bound = zeros(1,T);
for idx=1:T
    tau = range_tau(idx);
    Delta_tau = (1-tau)*Delta_0 + tau*Delta_1;
    cvx_begin quiet
    variable c nonnegative
    variable d nonnegative
    minimize c*epsilon^2 + d*eta^2
    subject to
    [ c*P1 - (eye(N)-L'*Delta_tau')*(eye(N)-Delta_tau*L),-(eye(N)-L'*Delta_tau')*Delta_tau;... 
      -Delta_tau'*(eye(N)-Delta_tau*L), d*eye(m)-Delta_tau'*Delta_tau] == semidefinite(N+m);
    cvx_end
    upper_bound(idx) = sqrt(cvx_optval);
end

% Finally, we calculate the theoretical value of the global worst-case error 
% and verify that the previous bounds are all equal to this theoretical value.
F = @(t) min(eig((1-t)*P1 + t*P2)) - ((1-t)^2*epsilon^2-t^2*eta^2)/((1-t)*epsilon^2 - t*eta^2);
tau = fzero(F,1/2);
lambda_sharp = min(eig((1-tau)*P1 + tau*P2));
gwce = sqrt((1-tau)*epsilon^2/lambda_sharp + tau*eta^2/lambda_sharp);

fprintf(['The lower bound, upper bounds, and theoretical values are all the same, precisely:\n '...
    'the lower bound equals to %.5f, \n '...
    'the maximal upper bound is %.5f, \n '...
    'and the theoretical value is %.5f.'], lower_bound, max(upper_bound), gwce)