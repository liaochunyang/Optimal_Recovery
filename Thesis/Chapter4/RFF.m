%% OR and function approximation:
clear all;clc;
f1 = @(x) 1./sqrt(1+sum(x.^2));
%f2 = @(x) sqrt(1+sum(x.^2));
%f3 = @(x) sum(exp(-abs(x)));

d = 10;                % dimension of input data

m_train =  1000;
m_test = 1000;

N_range = 800:100:1200;    % number of observations

Monte = 30;

% test data errors
test_err_WLS = zeros(Monte,length(N_range));
test_err_RLS = zeros(Monte,length(N_range));
test_err_OR = zeros(Monte,length(N_range));

for j=1:length(N_range)
    
    N = N_range(j);        % number of Fourier features
    Omega = randn(d,N);    % Fourier Feature matrix
    P = eye(N);            % V={0} so P_V^{\perp} = eye(N)
    
    % training data
    X_train = 2*rand(d,m_train)-1;     % training data matrix    
    A_train = sin(X_train'*Omega);     % A_i,j = \phi(x_i,w_j)
    
    % test data
    X_test = 2*rand(d,m_test)-1;       % test data matrix
    A_test = sin(X_test'*Omega);
    y_test = f1(X_test)';
    
    % observational error
    eta = 3;    % bound on observational error
    cov = exp(-dist(X_train).^2);             % covariance matrix
    %cov = randn(m_train);
    %cov = cov*cov';
    
    for i=1:Monte
        e = mvnrnd(zeros(1,m_train),cov);             % observation errors
        y_train = f1(X_train)' + eta*e'/norm(e);      % noisy observations
    
        % weighted least-squares
        c_WLS = (A_train'*inv(eta^2/norm(e)^2*cov)*A_train)\(A_train'*inv(eta^2/norm(e)^2*cov)*y_train);
        test_err_WLS(i,j) = norm(y_test-A_test*c_WLS)/m_test;
        
        % regularized least squares
        c_RLS = (A_train'*A_train+0.0001*eye(N))\(A_train'*y_train);
        test_err_RLS(i,j) = norm(y_test-A_test*c_RLS)/m_test;
        
        % Optimal Recovery
        tau = max(1-eta/norm(y_train),0);
        if tau == 0
           c_OR = zeros(N,1);
        else
           c_OR = ((1-tau)*eye(N)+tau*A_train'*A_train)\(tau*A_train'*y_train);
        end
        test_err_OR(i,j) = norm(y_test-A_test*c_OR)/m_test;
    end
end

%
figure(1)
plot(N_range,mean(test_err_OR),'k-*',N_range,mean(test_err_RLS),'b-o',N_range,mean(test_err_WLS),'r-x','Linewidth',1.5,'MarkerSize',6);
legend('Optimal Recovery','Regularized LS','Weighted LS','Fontsize',14,'Location','northwest')
xlabel('Number of observations N','Fontsize',14)
ylabel('Mean-squared error','Fontsize',14)
title(' Function 1 ')

%% OR and function approximation:
clear all;clc;
rng(6)
%f1 = @(x) 1./sqrt(1+sum(x.^2));
f2 = @(x) sqrt(1+sum(x.^2));
%f3 = @(x) sum(exp(-abs(x)));

d = 10;                % dimension of input data

m_train =  1000;
m_test = 1000;

N_range = 800:100:1200;    % number of observations

Monte = 30;

% test data errors
test_err_WLS = zeros(Monte,length(N_range));
test_err_RLS = zeros(Monte,length(N_range));
test_err_OR = zeros(Monte,length(N_range));

for j=1:length(N_range)
    
    N = N_range(j);        % number of Fourier features
    Omega = randn(d,N);    % Fourier Feature matrix
    P = eye(N);            % V={0} so P_V^{\perp} = eye(N)
    
    % training data
    X_train = 2*rand(d,m_train)-1;     % training data matrix    
    A_train = sin(X_train'*Omega);     % A_i,j = \phi(x_i,w_j)
    
    % test data
    X_test = 2*rand(d,m_test)-1;       % test data matrix
    A_test = sin(X_test'*Omega);
    y_test = f2(X_test)';
    
    % observational error
    eta = 3;    % bound on observational error
    cov = exp(-dist(X_train).^2);             % covariance matrix
    %cov = randn(m_train);
    %cov = cov*cov';
    
    for i=1:Monte
        e = mvnrnd(zeros(1,m_train),cov);             % observation errors
        y_train = f2(X_train)' + eta*e'/norm(e);      % noisy observations
    
        % weighted least-squares
        c_WLS = (A_train'*inv(eta^2/norm(e)^2*cov)*A_train)\(A_train'*inv(eta^2/norm(e)^2*cov)*y_train);
        test_err_WLS(i,j) = norm(y_test-A_test*c_WLS)/m_test;
        
        % regularized least squares
        c_RLS = (A_train'*A_train+0.0001*eye(N))\(A_train'*y_train);
        test_err_RLS(i,j) = norm(y_test-A_test*c_RLS)/m_test;
        
        % Optimal Recovery
        tau = max(1-eta/norm(y_train),0);
        if tau == 0
           c_OR = zeros(N,1);
        else
           c_OR = ((1-tau)*eye(N)+tau*A_train'*A_train)\(tau*A_train'*y_train);
        end
        test_err_OR(i,j) = norm(y_test-A_test*c_OR)/m_test;
    end
end

%
figure(2)
plot(N_range,mean(test_err_OR),'k-*',N_range,mean(test_err_RLS),'b-o',N_range,mean(test_err_WLS),'r-x','Linewidth',1.5,'MarkerSize',6);
legend('Optimal Recovery','Regularized LS','Weighted LS','Fontsize',14,'Location','northwest')
xlabel('Number of observations N','Fontsize',14)
ylabel('Mean-squared error','Fontsize',14)
title(' Function 2 ')

%% OR and function approximation:
clear all;clc;
%f1 = @(x) 1./sqrt(1+sum(x.^2));
%f2 = @(x) sqrt(1+sum(x.^2));
f3 = @(x) sum(exp(-abs(x)));

d = 10;                % dimension of input data

m_train =  1000;
m_test = 1000;

N_range = 800:100:1200;    % number of observations

Monte = 30;

% test data errors
test_err_WLS = zeros(Monte,length(N_range));
test_err_RLS = zeros(Monte,length(N_range));
test_err_OR = zeros(Monte,length(N_range));

for j=1:length(N_range)
    
    N = N_range(j);        % number of Fourier features
    Omega = randn(d,N);    % Fourier Feature matrix
    P = eye(N);            % V={0} so P_V^{\perp} = eye(N)
    
    % training data
    X_train = 2*rand(d,m_train)-1;     % training data matrix    
    A_train = sin(X_train'*Omega);     % A_i,j = \phi(x_i,w_j)
    
    % test data
    X_test = 2*rand(d,m_test)-1;       % test data matrix
    A_test = sin(X_test'*Omega);
    y_test = f3(X_test)';
    
    % observational error
    eta = 3;    % bound on observational error
    cov = exp(-dist(X_train).^2);             % covariance matrix
    %cov = randn(m_train);
    %cov = cov*cov';
    
    for i=1:Monte
        e = mvnrnd(zeros(1,m_train),cov);             % observation errors
        y_train = f3(X_train)' + eta*e'/norm(e);      % noisy observations
    
        % weighted least-squares
        c_WLS = (A_train'*inv(eta^2/norm(e)^2*cov)*A_train)\(A_train'*inv(eta^2/norm(e)^2*cov)*y_train);
        test_err_WLS(i,j) = norm(y_test-A_test*c_WLS)/m_test;
        
        % regularized least squares
        c_RLS = (A_train'*A_train+0.0001*eye(N))\(A_train'*y_train);
        test_err_RLS(i,j) = norm(y_test-A_test*c_RLS)/m_test;
        
        % Optimal Recovery
        tau = max(1-eta/norm(y_train),0);
        if tau == 0
           c_OR = zeros(N,1);
        else
           c_OR = ((1-tau)*eye(N)+tau*A_train'*A_train)\(tau*A_train'*y_train);
        end
        test_err_OR(i,j) = norm(y_test-A_test*c_OR)/m_test;
    end
end

% 
figure(3)
plot(N_range,mean(test_err_OR),'k-*',N_range,mean(test_err_RLS),'b-o',N_range,mean(test_err_WLS),'r-x','Linewidth',1.5,'MarkerSize',6);
legend('Optimal Recovery','Regularized LS','Weighted LS','Fontsize',14,'Location','northwest')
xlabel('Number of observations N ','Fontsize',14)
ylabel('Mean-squared error','Fontsize',14)
title(' Function 3 ')