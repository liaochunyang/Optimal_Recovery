%% tau vs ration epsilon/eta
% The optimal value of tau chosen by the optimal recovery framework changes
% as we vary the ratio.

%% Load a small graph
clear all;clc;
load StateGraphs/NY-graph.mat

%assert(issymmetric(A))

%% Compute the Laplacian Matrix and Decomposition

d = sum(A,2);
D = diag(d);
L = D-A;
[V,E] = eig(full(L));
n = size(A,1);

%L = L/sum(sum(A));

% This is here to confirm the graph is connected; 
% the theory can likely be extended even if not, but we
% should just be careful about this
assert(E(2,2) > 0) 

%% Get ground truth labels: choose one of six options

i = 2;
x = R(:,i);

x = (x - min(x)) / (max(x)-min(x));
%x = x/norm(x);
epsilon = sqrt(x'*L*x);    % how to choose epsilon?
fprintf('You choose the regression values for %s \n',Categories{i})

% Then I guess you can decide whether or not to add errors on top of this.

%% Extract a few labeled points with error

nl = 15;            % number of labeled points
error_model = 1;    % see 'add_error' function for examples of different types of error
l = sort(randsample(n,nl));                     % sampled points


%% Define quantity of interest and observation map
Q = zeros(n-nl,n);
Q(:,setdiff(1:n,l)) = eye(n-nl);

% observation map (Lambda*Lambda' = Id)
Lambda = zeros(nl,n);           
Lambda(:,l) = eye(nl); 

ratio = 0.1:0.5:30.1;
range_tau = zeros(1,length(ratio));

for i = 1:length(ratio)
    eta = epsilon/ratio(i);           % bound on 2-norm of error vector
    e = add_error(x(l),d(l),error_model,eta);       % error vector at samples
    lab = x(l)+e;         % these are the observed labels, i.e., true plus error


% Given 'lab' and the graph A (with Laplacian matrix L),
% the goal is to infer the true labels x, especially on non-labeled nodes,
% but getting better predictions on labeled nodes is also useful

    % Find optimal parameter tau using semidefinite pragramming
    cvx_begin quiet
    variable c1 nonnegative
    variable c2 nonnegative
    minimize c1*epsilon^2 + c2*eta^2
    subject to
    c1*full(L) + c2*Lambda'*Lambda - Q'*Q == semidefinite(n)
    cvx_end
    range_tau(i) = c2/(c1+c2);
end

%%
plot(log(ratio),range_tau,'k-o','Linewidth',1,'markersize',3)
xlabel('log( \epsilon / \eta )','Fontsize',16)
ylabel('Optimal regularization parameter \tau','Fontsize',16)
title('Log Ratio versus Optimal Regularization Parameter','fontsize',16)
