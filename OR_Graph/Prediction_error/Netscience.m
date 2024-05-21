%% Reproducible file accompanying the paper
% "On the Optimal Recovery of Graph Signals"
% by S. Foucart, C.Liao, and N. Veldt. 
% CVX is required to execute this reproducible.
% Here we produce Figure 3(a) in the supplementary material.

%% Load a small graph
clear;clc;
load smallgraphs/Netscience.mat
rng(1)         % comment this line to procude a different plot

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

% normalize x to [0,1]
x = (x - min(x)) / (max(x) - min(x));

% true parameters and overestimations of parameters
epsilon = 2*sqrt(x'*L*x);
eta_true = 2;
eta = 2*eta_true;

%% Main function
range_nl = 5:5:n/2;
Monte = 50;
err_glo = zeros(Monte,length(range_nl));
err_glo_over = zeros(Monte,length(range_nl));
err_best = zeros(Monte,length(range_nl));
err_trivial = zeros(Monte,length(range_nl));

error_model = 1;         % generate uniform random noise
l = [];

T = 100;
range_tau = [0,linspace(0.01,0.99,T),1];

for i=1:length(range_nl)
    
    l = sort([l,randsample(setdiff(1:n,l),5)]);                     % sampled points
    nl = length(l);
    
    % observation map (Lambda*Lambda' = Id)
    Lambda = zeros(nl,n);           
    Lambda(:,l) = eye(nl); 
    
    % Quantity of interest
    Q = zeros(n-nl,n);
    Q(:,setdiff(1:n,l)) = eye(n-nl);
    
    S2 = L(l,setdiff(1:n,l));
    S3 = L(setdiff(1:n,l),setdiff(1:n,l));
    
    % Global OR: using true eta to find \tau
    cvx_begin quiet
    variable c1 nonnegative
    variable c2 nonnegative
    minimize c1*epsilon^2 + c2*eta_true^2
    subject to
    c1*full(L) + c2*Lambda'*Lambda - Q'*Q == semidefinite(n)
    cvx_end
    glo_tau = c2/(c1+c2);           % Optimal parameter
    
    % Global OR: using an overestimation of eta to find \tau
    cvx_begin quiet
    variable c1 nonnegative
    variable c2 nonnegative
    minimize c1*epsilon^2 + c2*eta^2
    subject to
    c1*full(L) + c2*Lambda'*Lambda - Q'*Q == semidefinite(n)
    cvx_end
    glo_tau_over = c2/(c1+c2);           % Optimal parameter
    
    for j=1:Monte
        
        e = add_error(x,d,error_model,eta_true);       % error vector at samples
        lab = x(l)+e(l);         % these are the observed labels, i.e., true plus error
        
        % Global OR: using true eta
        x_hat = Q*(((1-glo_tau)*full(L) + glo_tau*Lambda'*Lambda)\(glo_tau*Lambda'*lab));
        err_glo(j,i) = norm(x(setdiff(1:n,l))-x_hat);
        
        % Global OR: using an overestimated eta
        x_hat = Q*(((1-glo_tau_over)*full(L) + glo_tau_over*Lambda'*Lambda)\(glo_tau_over*Lambda'*lab));
        err_glo_over(j,i) = norm(x(setdiff(1:n,l))-x_hat);
        
        % Naive baseline \tau=0
        err_trivial(j,i) = norm(x(setdiff(1:n,l))-mean(lab)*ones(n-nl,1));
        
        % Best regularization
        err_lb = zeros(1,T+2);
        
        err_lb(1) = err_trivial(j,i);
        for k=2:T+1
            tau = range_tau(k);
            % Regularization map
            x_hat = Q*(((1-tau)*full(L) + tau*Lambda'*Lambda)\(tau*Lambda'*lab));
            err_lb(k) = norm(x(setdiff(1:n,l))-x_hat);
        end
        f_1 = -S3\(S2'*lab);
        err_lb(T+2) =  norm(x(setdiff(1:n,l))-f_1);
        
        err_best(j,i) = min(err_lb); 
        
    end
    
end


%% Plot
figure(1)
hold on
plot(range_nl,mean(err_trivial),'c-+','Linewidth',1)
plot(range_nl,mean(err_glo),'m-o',range_nl,mean(err_glo_over),'r-s','Linewidth',1)
plot(range_nl,mean(err_best),'b-*','Linewidth',1)
xticks(5:20:n/2)
xlim([5,n/2])
ylim([1.4,4.6])
legend('Naive baseline (\tau=0)','Global OR (true \eta)','Global OR (overestimated \eta)','Best regularization','Fontsize',20)
xlabel('Number of labeled nodes','Fontsize',20)
ylabel('Prediction error','Fontsize',20)
pbaspect([1.5 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',20)
outerpos = get(gca,'OuterPosition');
ti = get(gca,'TightInset');
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
set(gca,'Position',[left bottom ax_width ax_height]);
title('Prediction error vs. number of labeled nodes','Fontsize',22)
hold off