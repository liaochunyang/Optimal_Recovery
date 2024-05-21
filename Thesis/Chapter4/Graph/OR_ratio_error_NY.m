%% Error versus ratio epsilon/eta
% Monte Carlo simulation.

%% Load a small graph
clear all;clc;
load StateGraphs/NY-graph.mat

%assert(issymmetric(A))

%% Compute the Laplacian Matrix and Decomposition

d = sum(A,2);
D = diag(d);
L = D-A;
L = L/sum(sum(A));       % scale all entries in L by a factor 1/|E|
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
epsilon = sqrt(x'*L*x);    

fprintf('You choose the regression values for %s \n',Categories{i})

% Then I guess you can decide whether or not to add errors on top of this.

%% Extract a few labeled points with error

nl = 15;            % number of labeled points
error_model = 4;    % see 'add_error' function for examples of different types of error
l = sort(randsample(n,nl));                     % sampled points

%% Define quantity of interest and observation map
Q = zeros(n-nl,n);
Q(:,setdiff(1:n,l)) = eye(n-nl);

% observation map (Lambda*Lambda' = Id)
Lambda = zeros(nl,n);           
Lambda(:,l) = eye(nl); 

ratio = 0.1:0.2:2.5;

Monte = 50;

gwce = zeros(Monte,length(ratio));
err = zeros(Monte,length(ratio));


for j = 1:length(ratio)
    eta = epsilon/ratio(j);           % bound on 2-norm of error vector
    
    for i=1:Monte
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
        gwce(i,j) = sqrt(c1*epsilon^2+c2*eta^2);
        opt_tau = c2/(c1+c2);           % Optimal parameter
        Delta_opt = inv((1-opt_tau)*full(L) + opt_tau*Lambda'*Lambda)*(opt_tau*Lambda');
        lab_or = Q*Delta_opt*lab;
        err(i,j) = norm(x(setdiff(1:n,l))-lab_or)/(n-nl);
    end
end
%% Plot
figure(1)
plot(ratio,mean(err),'b-o','Linewidth',1.5,'markersize',3)
xlabel('Ratio \epsilon / \eta','Fontsize',16)
ylabel('Prediction error','Fontsize',16)
title('Prediction error versus Ratio','Fontsize',16)

figure(2)
plot(ratio,mean(gwce),'r-*','Linewidth',1.5,'markersize',3)
xlabel('Ratio \epsilon / \eta','Fontsize',16)
ylabel('Worst-case error','Fontsize',16)
title('Worst-case error versus ratio','Fontsize',16)