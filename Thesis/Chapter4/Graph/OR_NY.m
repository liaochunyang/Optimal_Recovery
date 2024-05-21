%% Error versus regularization parameter

%% Load a small graph
clear all;clc;
load StateGraphs/NY-graph.mat
rng(13)
%assert(issymmetric(A))

%% Compute the Laplacian Matrix and Decomposition

d = sum(A,2);
D = diag(d);
L = D-A;
%L = L/sum(sum(A));       % scale all entries in L by a factor 1/|E|
[V,E] = eig(full(L));
n = size(A,1);

% This is here to confirm the graph is connected; 
% the theory can likely be extended even if not, but we
% should just be careful about this
assert(E(2,2) > 0) 

%% Get ground truth labels: choose one of six options

i = 2;
x = R(:,i);

x = (x - min(x)) / (max(x)-min(x));
%x = x/norm(x);
epsilon = sqrt(x'*L*x);    

fprintf('You choose the regression values for %s \n',Categories{i})

% Then I guess you can decide whether or not to add errors on top of this.

%% Extract a few labeled points with error

nl = 15;            % number of labeled points
eta = epsilon/2;           % bound on 2-norm of error vector
error_model = 1;    % see 'add_error' function for examples of different types of error

l = sort(randsample(n,nl));                     % sampled points
e = add_error(x(l),d(l),error_model,eta);       % error vector at samples

lab = x(l)+e;         % these are the observed labels, i.e., true plus error

% Given 'lab' and the graph A (with Laplacian matrix L),
% the goal is to infer the true labels x, especially on non-labeled nodes,
% but getting better predictions on labeled nodes is also useful

%% Define quantity of interest and observation map
Q = zeros(n-nl,n);
Q(:,setdiff(1:n,l)) = eye(n-nl);

% observation map (Lambda*Lambda' = Id)
Lambda = zeros(nl,n);           
Lambda(:,l) = eye(nl); 

%epsilon = x'*L*x/3;    % how to choose epsilon?

%% Range of tau
T = 100;
range_tau = linspace(0.01,0.99,T);
err = zeros(1,T);
gwce = zeros(1,T);
for i=1:T
    tau = range_tau(i);
    % Regularization map
    Delta_tau = inv((1-tau)*full(L) + tau*Lambda'*Lambda)*(tau*Lambda');
    err(i) = norm(x(setdiff(1:n,l))-Q*Delta_tau*lab);
    gwce(i) = gwceub(epsilon,eta,full(L),n,nl,Lambda,Q*Delta_tau,Q);
end

%% Two special cases tau=0 and tau=1
% tau = 0: using the average
diff_ave = x - mean(lab)*ones(size(x));
err_0 = norm(diff_ave(setdiff(1:n,l)));

% tau=1: interpolation, explicit formula
S2 = L(l,setdiff(1:n,l));
S3 = L(setdiff(1:n,l),setdiff(1:n,l));
f_1 = -S3\(S2'*lab);
err_1 =  norm(x(setdiff(1:n,l))-f_1);
gwce_1 = gwceub(epsilon,eta,L,n,nl,Lambda,full(-inv(S3)*S2'),Q);

%% Main Function (Recover non-labeled nodes)

% Find optimal parameter tau using semidefinite pragramming
cvx_begin quiet
variable c nonnegative
variable d nonnegative
minimize c*epsilon^2 + d*eta^2
subject to
c*full(L) + d*Lambda'*Lambda - Q'*Q == semidefinite(n)
cvx_end
lb = sqrt(c*epsilon^2+d*eta^2);
opt_tau = d/(c+d);           % Optimal parameter
Delta_opt = inv((1-opt_tau)*full(L) + opt_tau*Lambda'*Lambda)*(opt_tau*Lambda');
%lab_or = Q*Delta_opt*lab;
%norm(lab_or)

%% Plot
hold on
yyaxis left
plot([0,range_tau,1],[err_0,err,err_1],'b-','Linewidth',1)
%yline(err_0,'Linewidth',1)
ylim([min(err)-0.1,max(err)+0.1])
ylabel('Prediction error','Fontsize',16)

yyaxis right
plot([range_tau,1],[gwce,gwce(T)],'r--','Linewidth',1)
yline(lb,'r-.','Linewidth',1)
legend('Prediction Error','Upper Bound on Worst-case Error','Lower Bound on Worst-case Error','fontsize',14,'location','best')
ylim([lb-0.5,max(gwce)+0.5])
xticks(opt_tau)
ylabel('Worst-case error','Fontsize',16)
title('New York','Fontsize',16)
hold off