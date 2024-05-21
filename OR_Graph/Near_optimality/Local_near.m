%% Reproducible file accompanying the paper
% "On the Optimal Recovery of Graph Signals"
% by S. Foucart, C.Liao, and N. Veldt. 
% CVX is required to execute this reproducible.
% Here we produce Figure 2 in the main text.
%% Load a small graph
clear;clc;
load smallgraphs/KarateA.mat
rng(7)        % comment out this line to produce a different plot
%% Compute the Laplacian Matrix and Decomposition

d = sum(A,2);
D = diag(d);
L = D-A;
[V,E] = eig(full(L));
n = size(A,1);

% This is here to confirm the graph is connected; 
% the theory can likely be extended even if not, but we
% should just be careful about this
assert(E(2,2) > 0) 

%% Generate the synthetic "ground truth" labels
% Similar to (15) in survey: "Learning Graphs From Data" https://arxiv.org/pdf/1806.00848.pdf

c = mvnrnd(zeros(n,1),pinv(E))';      
x = V*c;

% normalize to [0,1]
x = (x - min(x)) / (max(x) - min(x));
% true parameters
epsilon = 2*sqrt(x'*L*x);
eta = 2;

%% Extract a few labeled points
nl = 10;                           % number of labeled points
l = sort(randsample(n,nl));        % sampled points

%% Define observation map and quantity of interest

% observation map (Lambda*Lambda' = Id)
Lambda = zeros(nl,n);           
Lambda(:,l) = eye(nl); 

Q = zeros(n-nl,n);
Q(:,setdiff(1:n,l)) = eye(n-nl);

S2 = L(l,setdiff(1:n,l));
S3 = L(setdiff(1:n,l),setdiff(1:n,l));

error_model = 1;
e = add_error(x(l),d(l),error_model,eta);       % error vector at samples
lab = x(l)+e;         % these are the observed labels, i.e., true plus error

cvx_begin quiet
cvx_solver mosek
variable x_hat(n)
variable c
minimize c
subject to 
x_hat'*L*x_hat <= c;
sum_square(Lambda*x_hat-lab) <= c*eta^2/epsilon^2
cvx_end
[x_hat'*L*x_hat, epsilon^2/eta^2*norm(Lambda*x_hat-lab)^2]

%% Near optimal regularization parameter \tau
g = @(t) ((1-t)*full(L) + t*(Lambda'*Lambda))\(t*Lambda'*lab);
g1 = @(t) g(t)'*L*g(t)-epsilon^2/eta^2*norm(Lambda*g(t)-lab)^2;

% Using built-in function to find the parameter
opt_tau = fzero(g1,0.5);

%% Compute an upper bound on local worst-case error for every parameter \tau
T = 100;
range_tau = [0,linspace(0.01,0.99,T),1];
lwce = zeros(1,T+2);
lwce(1) = lwceub(epsilon,eta,full(L),n,Lambda,lab,1/nl*Q*ones(n,nl),Q);
for i=2:T+1
    tau = range_tau(i);
    Delta_tau = ((1-tau)*full(L) + tau*Lambda'*Lambda)\(tau*Lambda');
    lwce(i) = lwceub(epsilon,eta,L,n,Lambda,lab,Q*Delta_tau,Q);
end
lwce(T+2) = lwceub(epsilon,eta,full(L),n,Lambda,lab,full(-inv(S3)*S2'),Q);

%% Plot
plot(range_tau,lwce,'b-','Linewidth',1)
xticks(opt_tau)
xline(opt_tau,'--','Linewidth',0.5)
legend('Upper bound on local worst-case error','Fontsize',18,'location','northwest')
ylim([min(lwce)-0.5,max(lwce)+0.5])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)
xlabel('Regularization parameter \tau','Fontsize',18)
ylabel('Local worst-case error','Fontsize',18)
title('Near optimality of a regularization parameter','Fontsize',18)
