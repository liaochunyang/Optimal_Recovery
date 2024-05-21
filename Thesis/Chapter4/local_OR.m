%% Section 4.2: Local optimal recovery in complex finite-dimensional Hilbert Spaces
close all; clear;
% dimension of the ambiant Hilbert space
N = 50;         
% dimension of the approximation subspace
n = 15;                     
% subspace spanned by the columns of V
V = complex(randn(N,n),randn(N,n));
% projector onto the orthogonal complement of V
P = eye(N) - V*inv(V'*V)*V';         
% the approximability parameter
epsilon = 0.1;             
% number of observations
m = 25;                       
% the observation map (we use L instead of \Lambda)
L = complex(randn(m,N),randn(m,N));
% the uncertainty parameter
eta = 0.5;                    
% the original f and of its observation vector
aux = randn(N,1);
f = V*rand(n,1)+(epsilon/2)*aux/norm(aux);
% the inaccurate observations 
aux = randn(m,1);
y = L*f + (2*eta/3)*aux/norm(aux);

% linear quantity of interest
s = N;
Q = complex(rand(s,N),rand(s,N));
Q = (sqrtm(Q*Q'))\Q;

% For full recovery problem, use the following Q and comment above three lines out
% Q = eye(N)        

% computing localworst-case error
cvx_begin quiet
variable c nonnegative
variable d nonnegative
variable t nonnegative
minimize c*epsilon^2 + d*(eta^2 - norm(y)^2) + t
subject to
c*P + d*(L'*L) - Q'*Q == hermitian_semidefinite(N);
[eye(N) + c*Q'*Q*P+d*Q'*Q*(L'*L)-Q'*Q, d*Q'*Q*L'*y; d*y'*L*Q'*Q, t] == hermitian_semidefinite(N+1);
cvx_end
wce = sqrt(cvx_optval);
fprintf('The local worst-case error is %.4f',wce)
% Chebyshev center
cheb = d*Q*inv(eye(N)+c*Q'*Q*P+d*Q'*Q*(L'*L)-Q'*Q)*Q'*Q*L'*y;

%% Section 4.3: Local optimal recovery in real finite-dimensional Hilbert Spaces
% non-trivial V
clc; clear all;
N = 50;         
n = 15;
V = rand(N,n);             
epsilon = 0.5;             
m = 25;                     
L = randn(m,N);            
L = (sqrtm(L*L'))\L;       % so that L satisfies L*L'=I
eta = 0.2;                 
aux = randn(N,1);
f = V*rand(n,1)+(epsilon/2)*aux/norm(aux);
aux = randn(m,1);
y = L*f + (2*eta/3)*aux/norm(aux);

% some useful objects
P1 = eye(N) - V*inv(V'*V)*V';  % orthogonal projector onto the orthogonal complement of V
P2 = L'*L;                     % orthogonal projector onto the range of L'
C = L*V;                       % cross-gramian (see the appendix)
b = (C'*C)\(C'*y);
a = y-C*b;
f_0 = V*b;                     % the regularized solution corresponding to tau=0
f_1 = L'*a + V*b;              % the regularized solution corresponding to tau=1
delta = norm(f_0-f_1);         % the parameter in the implicit equation defining tau_#
delta_0 = norm(L*f_0-y);       % a second way to calculate delta
delta_1 = norm(P1*f_1);        % a third way to calculate delta
fprintf(['As they should, the three ways to calculate delta return the same values:\n' ...
    '%.3f , %.3f , and %.3f.\n'], delta, delta_0, delta_1)

% Using Newton's metod to find the optimal regularization parameter
% The smallest eigenvalue as a function of t
eigen = @(t) min(eig((1-t)*P1 + t*P2));
% The numerator of the fraction in the RHS and its derivative
nume = @(t) (1-t)^2*epsilon^2 - t^2*eta^2;
diff_nume = @(t) -2*(1-t)*epsilon^2 - 2*t*eta^2;
% The denominator of the fraction in the RHS its derivative
denom = @(t) (1-t)*epsilon^2 - t*eta^2 + (1-t)*t*(1-2*t)*delta^2;
diff_denom = @(t) -epsilon^2 - eta^2 + (1-6*t+6*t^2)*delta^2;

% Define the function F and its derivative 
F = @(t) eigen(t) - nume(t)/denom(t);
diff_F = @(t) (1-2*t)/t/(1-t)*eigen(t)*(1-eigen(t))/(1-2*eigen(t))...
    - (diff_nume(t)*denom(t) - diff_denom(t)*nume(t))/denom(t)^2;
% Find the optimal parameter tau (the zero of F)
tau = epsilon/(epsilon+eta);
iter = 25;
for i=1:iter
   tau = tau - F(tau)/diff_F(tau);
end

% For comparison, use MATLAB built-in function to find the zero of F
tau_builtin = fzero(F,1/2);

% Compare also with the output of the method from Beck and Eldar
% (not guaranteed to yield the optimal parameter in the real setting)
%[tau_BE,~,wce_BE] = BE(epsilon,eta,y,P1,L);
cvx_begin quiet
variable c nonnegative
variable d nonnegative
variable t nonnegative
minimize epsilon^2*c - (norm(y)^2-eta^2)*d + t
subject to
c*P1 + d*(L'*L) - eye(N) == hermitian_semidefinite(N);
[c*P1+d*(L'*L), -d*L'*y; -d*y'*L, t] == hermitian_semidefinite(N+1);
cvx_end
tau_BE = d/(c+d);
%cheb_center = (c*P+d*L'*L)\(d*L'*y);
wce_BE = sqrt(cvx_optval);

fprintf(['With H being a real finite-dimensional Hilbert space and using orthonormal observations,\n ' ...
    'our costum Newton method provides the optimal tau equal to %.8f,\n '...
    'the MATLAB built-in function yields the parameter equal to %.8f,\n '...
    'and the method of Beck--Eldar returns a parameter equal to %.8f.\n'], tau, tau_builtin, tau_BE)

%% Section 4.3:
% distinct case V={0}
clear all; clc; 
N = 10;
P1 = eye(N);
epsilon = 0.1;       
m = 6;                    
L = randn(m,N);             
L = inv(sqrtm(L*L'))*L;       
eta = 0.2;                    
aux = randn(N,1);
f = epsilon*aux/norm(aux);
aux = randn(m,1);
y = L*f + eta*aux/norm(aux);
if norm(y) <= eta
    opt_tau = 0;
    fprintf(['The norm of y is %.4f, which is smaller than or equal to eta,' ...
        ' so the optimal parameter if is 0.\n'], norm(y))    
else
    opt_tau = 1-eta/norm(y);
    fprintf(['The norm of y is %.3f, which is greater than eta,' ...
        ' so the optimal parameter is %.6f.\n'], norm(y), opt_tau)
end

% Here too, the method of Beck--Eldar seems to give the correct parameter
cvx_begin quiet
variable c nonnegative
variable d nonnegative
variable t nonnegative
minimize epsilon^2*c - (norm(y)^2-eta^2)*d + t
subject to
c*P1 + d*(L'*L) - eye(N) == hermitian_semidefinite(N);
[c*P1+d*(L'*L), -d*L'*y; -d*y'*L, t] == hermitian_semidefinite(N+1);
cvx_end
tau_BE = d/(c+d);
wce = sqrt(cvx_optval);
fprintf(['The parameter obtained after solving the semidefinite program of Beck--Eldar is %.6f,\n ' ...
    'which agrees with the value of the optimal parameter given by the above formula.'], tau_BE)