%% Two-space problem 
% This file contains numerical illustration for the two-space problem.
% We verify the following items numerically.
% 1. There is a lower bound for any recovery map
% 2. Compute the optimal regularization parameter \tau_opt
% 3. Compute upper bound for any regularization map

%% Problem setting
clear all;clc;
N = 20;               % dimension of ambient space

n1 = 6;               % dimension of V
V = rand(N,n1);
P1 = eye(N) - V*inv(V'*V)*V';   % orthogonal projector onto V^\perp

n2 = 10;              % dimension of W
W = rand(N,n2);
P2 = eye(N) - W*inv(W'*W)*W';   % orthogonal projector onto W^\perp

epsilon = 0.5;
nu = 0.4; 

m = 15;                     
L = randn(m,N);

% some auxilliary matrices
[Q,~] = qr(L');
H = Q(:,m+1:N);        % the columns of H form an ONB for ker(L)
P_kerL = H*H';         % the orthoprojector onto ker(L)

%% Lower bound for any recovery map
cvx_begin quiet
variable c nonnegative
variable d nonnegative
minimize c*epsilon^2 + d*nu^2
subject to
c*H'*P1*H + d*H'*P2*H - eye(N-m) == semidefinite(N-m)
cvx_end
gwce_lb = sqrt(c*epsilon^2 + d*nu^2);
tau_opt = d/(c+d);
fprintf('A lower bound of worst-case error is %.4f\n', gwce_lb)

%% Upper bound for any regularization recovery map
T = 100;
range_tau = linspace(0.01,0.99,T);
gwce = zeros(1,T);
for i=1:T
    tau = range_tau(i);
    % M_tau in notes
    M1 = (1-tau) * P_kerL * P1 + tau * P_kerL * P2;
    % M_tilde_tau in notes
    M2 = ((1-tau) * P_kerL * P1 + tau * P_kerL * P2) * H;
    % recovery map
    Delta_tau = (eye(N) - H * inv(M2'*M2)*M2'*M1)*L'*inv(L*L');
    cvx_begin quiet
    variable c nonnegative
    variable d nonnegative
    minimize c*epsilon^2 + d*nu^2
    subject to
    [eye(N), eye(N)-Delta_tau*L;
     (eye(N)-Delta_tau*L)', c*P1+d*P2] == semidefinite(2*N)
    cvx_end
    gwce(i) = sqrt(c*epsilon^2+d*nu^2);
end

%% Plot
plot(range_tau,gwce,'k-',range_tau,gwce_lb*ones(1,T),'r-','LineWidth',1)
legend('Upper bound','Lower bound','fontsize',14,'Location','best')
xlabel('regularization parameter','fontsize',14)
ylabel('worst-case error','fontsize',14)
xticks(tau_opt)
title('Numerical Experiments on bounds for worst-case error for full recovery problem')

%% Quantity of interest 
% We generalize our result to a linear quantity of
% interest. Previous result can be viewed as taking $Q$ to be identity map.
% Numerical experiment shows a way to find the optimal regularization
% parameter but it is not proved.

clear all;clc;
N = 20;               % dimension of ambient space

n1 = 6;               % dimension of V
V = rand(N,n1);
P1 = eye(N) - V*inv(V'*V)*V';   % orthogonal projector onto V^\perp

n2 = 10;              % dimension of W
W = rand(N,n2);
P2 = eye(N) - W*inv(W'*W)*W';   % orthogonal projector onto W^\perp

epsilon = 0.5;
nu = 0.4; 

m = 15;                     
L = randn(m,N);

% some auxilliary matrices
[Q,~] = qr(L');
H = Q(:,m+1:N);        % the columns of H form an ONB for ker(L)
P_kerL = H*H';         % the orthoprojector onto ker(L)

s = 6;
q = rand(s,N);         % linear quantity of interest

% Lower bound for any recovery map
cvx_begin quiet
variable c nonnegative
variable d nonnegative
minimize c*epsilon^2 + d*nu^2
subject to
c*H'*P1*H + d*H'*P2*H - H'*q'*q*H == semidefinite(N-m)
cvx_end
gwce_lb = sqrt(c*epsilon^2 + d*nu^2);
tau_opt = d/(c+d);
fprintf('A lower bound of worst-case error is %.4f\n', gwce_lb)

% Upper bound for any regularization recovery map
T = 100;
range_tau = linspace(0.01,0.99,T);
gwce = zeros(1,T);
for i=1:T
    tau = range_tau(i);
    % M_tau in notes
    M1 = (1-tau) * P_kerL * P1 + tau * P_kerL * P2;
    % M_tilde_tau in notes
    M2 = ((1-tau) * P_kerL * P1 + tau * P_kerL * P2) * H;
    % recovery map
    Delta_tau = (eye(N) - H * inv(M2'*M2)*M2'*M1)*L'*inv(L*L');
    cvx_begin quiet
    variable c nonnegative
    variable d nonnegative
    minimize c*epsilon^2 + d*nu^2
    subject to
    [eye(s), q*(eye(N)-Delta_tau*L);
     (eye(N)-Delta_tau*L)'*q', c*P1+d*P2] == semidefinite(N+s)
    cvx_end
    gwce(i) = sqrt(c*epsilon^2+d*nu^2);
end

% Plot
plot(range_tau,gwce,'k-',range_tau,gwce_lb*ones(1,T),'r-','LineWidth',1)
ylim([gwce_lb-0.02,max(gwce)+0.02])
legend('Upper bound','Lower bound','fontsize',14,'Location','best')
xlabel('regularization parameter','fontsize',14)
ylabel('worst-case error','fontsize',14)
xticks(tau_opt)
title('Numerical Experiments on bounds for worst-case error for recovering quantity of interest')