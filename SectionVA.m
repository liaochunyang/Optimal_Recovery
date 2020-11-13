%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reproducible MATLAB file accompanying the paper
%  LEARNING FROM NON-RANDOM DATA IN HILBERT SPACES: 
%        AN OPTIMAL RECOVERY PERSPECTIVE     
% by S. Foucart, C. Liao, S. Shahrampour, and Y. Wang
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CVX [2] is required to execute this reproducible
% Gurobi [3] and MOSEK [4] are required for parts of this reproducible
clear all; clc;

%% Section V.A: Comparison of Worst-Case Errors

cvx_quiet true
rng(3)
times = 5;
% range of approximability parameter
epsilon = 0.2:0.001:0.205;
wce_OR = zeros(length(epsilon),times);
wce_ERM2V = zeros(length(epsilon),times);
wce_ERM2K = zeros(length(epsilon),times);
wce_ERM1V = zeros(length(epsilon),times);
wce_ERM1K = zeros(length(epsilon),times);

% Set up the problem in the finite-dim Hilbert space L_2^N
N = 200;
% the approximation space is spanned by the columns of V
n = 20;
V = rand(N,n);
% the measurement process is defined by the linear map L
m = 50;
L = rand(m,N);

% some auxilliary matrices
G = L*L';              % Gramian of the representers of the L_i's 
Ginv = inv(G);         % inverse of this Gramian
C = L*V;               % cross-Gramian with a basis for V
[QV,~] = qr(V,0);      % the columns of QV form an ONB for V
P_V = QV*QV';          % the orthoprojector onto V
[Q,~] = qr(L');
H = Q(:,m+1:N);        % the columns of H form an ONB for ker(L)
P_kerL = H*H';         % the orthoprojector onto ker(L)

% create an element f0 in the model set and its observations y
aux = (eye(N)-P_V)*rand(N,1);
f0 = V*rand(n,1)+epsilon(1)*aux/norm(aux);
y = L*f0;

for j=1:times
for i=1:length(epsilon)
    
% Construct recovery algorithms
% We produce the outputs of the optimal recovery map,
% and of empirical risk minimization (p=1,2) with  
% two different constraints f\in K and f\in V

% the element f_OR learned with the optimal recovery map
b = (C'*Ginv*C)\(C'*Ginv*y);
a = Ginv*(y - C*b);
f_OR = L'*a + V*b;

% the element f_ERM2V learned with empirical risk minimization (p=2)
% with constraint f\in V
c_ERM2V = (C'*C)\C'*y;
f_ERM2V = V*c_ERM2V;

% the element f_ERM2K learned with empirical risk minimization (p=2)
% with constraint f\in K
cvx_solver gurobi
cvx_begin
variable f_ERM2K(N)
minimize norm(y-L*f_ERM2K)
subject to
norm(f_ERM2K-P_V*f_ERM2K) <= epsilon(i)
cvx_end

% the element f_ERM1V learned with empirical risk minimization (p=1)
% with constraint f\in V
cvx_solver mosek
cvx_begin
variable c_ERM1V(n)
variable e(m)
minimize sum(e)
subject to
e - y + L*V*c_ERM1V >= 0;
e + y - L*V*c_ERM1V >= 0;
cvx_end
f_ERM1V = V*c_ERM1V;

% the element f_ERM1K learned with empirical risk minimization (p=1)
% with constraint f\in K
cvx_solver gurobi
cvx_begin
variable f_ERM1K(N)
variable e(m)
minimize sum(e)
subject to
e - y + L*f_ERM1K >= 0;
e + y - L*f_ERM1K >= 0;
norm(f_ERM1K-P_V*f_ERM1K) <= epsilon(i)
cvx_end

% Compute worst-case errors for each algorithm 

h = L'*Ginv*y;
w = h - P_V*h;
% the optimal recovery map
g = f_OR;
g_kerL = P_kerL*g;
cvx_solver mosek
cvx_begin
variable c
variable d
minimize c
subject to
d >= 0;
[H'*((d-1)*eye(N)-d*P_V)*H, H'*(d*w+g_kerL);...
    (g_kerL'+d*w')*H, c+d*(norm(w)^2-epsilon(i)^2)] ...
    == semidefinite(N-m+1);
cvx_end
wce_OR(i,j) = sqrt( norm(h-(g-g_kerL))^2 + norm(g_kerL)^2 + c );

% the empirical risk minimization f_ERM2V with p=2 and f\in V
g = f_ERM2V;
g_kerL = P_kerL*g;
cvx_solver mosek
cvx_begin
variable c
variable d
minimize c
subject to
d >= 0;
[H'*((d-1)*eye(N)-d*P_V)*H, H'*(d*w+g_kerL);...
    (g_kerL'+d*w')*H, c+d*(norm(w)^2-epsilon(i)^2)] ...
    == semidefinite(N-m+1);
cvx_end
wce_ERM2V(i,j) = sqrt( norm(h-(g-g_kerL))^2 + norm(g_kerL)^2 + c );

% the empirical risk minimization f_ERM2K with p=2 and f\in K
g = f_ERM2K;
g_kerL = P_kerL*g;
cvx_solver mosek
cvx_begin
variable c
variable d
minimize c
subject to
d >= 0;
[H'*((d-1)*eye(N)-d*P_V)*H, H'*(d*w+g_kerL);...
    (g_kerL'+d*w')*H, c+d*(norm(w)^2-epsilon(i)^2)] ...
    == semidefinite(N-m+1);
cvx_end
wce_ERM2K(i,j) = sqrt( norm(h-(g-g_kerL))^2 + norm(g_kerL)^2 + c );

% the empirical risk minimization f_ERM1V with p=1 and f\in V
g = f_ERM1V;
g_kerL = P_kerL*g;
cvx_solver mosek
cvx_begin
variable c
variable d
minimize c
subject to
d >= 0;
[H'*((d-1)*eye(N)-d*P_V)*H, H'*(d*w+g_kerL);...
    (g_kerL'+d*w')*H, c+d*(norm(w)^2-epsilon(i)^2)] ...
    == semidefinite(N-m+1);
cvx_end
wce_ERM1V(i,j) = sqrt( norm(h-(g-g_kerL))^2 + norm(g_kerL)^2 + c );

% the empirical risk minimization f_ERM1K with p=1 and f \in K
g = f_ERM1K;
g_kerL = P_kerL*g;
cvx_solver mosek
cvx_begin
variable c
variable d
minimize c
subject to
d >= 0;
[H'*((d-1)*eye(N)-d*P_V)*H, H'*(d*w+g_kerL);...
    (g_kerL'+d*w')*H, c+d*(norm(w)^2-epsilon(i)^2)] ...
    == semidefinite(N-m+1);
cvx_end
wce_ERM1K(i,j) = sqrt( norm(h-(g-g_kerL))^2 + norm(g_kerL)^2 + c );

end
end
y_min = min(min(wce_OR));
y_max = max(max(wce_ERM1V));
figure(1)
wce_OR_ave = mean(wce_OR,2);
wce_ERM1V_ave = mean(wce_ERM1V,2);
wce_ERM1K_ave = mean(wce_ERM1K,2);
wce_ERM2V_ave = mean(wce_ERM2V,2);
wce_ERM2K_ave = mean(wce_ERM2K,2);
plot(epsilon,wce_OR_ave,'g-o',epsilon,wce_ERM1V_ave,'r-*',epsilon,wce_ERM2V_ave,'b-.')
xlabel('approximability parameter epsilon','Fontsize',14)
ylabel('worst case error','Fontsize',14)
ylim([y_min-0.01,y_max+0.01])
legend({'optimal recovery map','ERM1','ERM2'},'FontSize',12)

figure(2)
plot(epsilon,wce_OR_ave,'g-o',epsilon,wce_ERM1K_ave,'m-*',epsilon,wce_ERM2K_ave,'k-.')
xlabel('approximability parameter epsilon','Fontsize',14)
ylabel('worst case error','Fontsize',14)
ylim([y_min-0.01,y_max+0.01])
legend({'optimal recovery map','ERM1','ERM2'},'FontSize',12)


%% References

% 1. S. Foucart, C. Liao, S. Shahrampour, and Y. Wang
% "Learning from Non-Random Data in Hilbert Spaces: 
% An Optimal Recovery Perspective",
% Preprint.

% 2. CVX Research, Inc., 
% "CVX: MATLAB software for disciplined convex programming"
% version 2.1, 2014, http://cvxr.com/cvx.

% 3. Gurobi Optimization, LLC, 
% Gurobi optimizer reference manual, 2020 [Online]. 
% Available: https://www.gurobi.com

% 4. MOSEK ApS,
% MOSEK documentation, 2019 [Online].
% Available: https://www.mosek.com


