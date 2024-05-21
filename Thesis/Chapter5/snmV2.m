%%%%%%%%%%%%%%%%%% Optimal Recovery and Time series %%%%%%%%%%

% Data set description is on the paper:
% Optimal Error and GMDH Predictors A Comparison with Some Statistical Techniques

%% data loading and processing
clear all;clc;
X = readtable('snm.csv');
y = table2array(X(2900:3279,4));
n = length(y);
y = y-mean(y);
% plot(y)

%% Optimal Recovery Method

m = 11;         % Maximum delay
T = 50;         % approximate period
eta = 0;        % ell_2 bounded observation error
epsilon = 1;    % approximation parameter

x = 1:m;          
A = [ones(1,m);sin(2*pi*x/T)];
b = [1; sin(2*pi*(m+1)/T)];
cvx_begin quiet
variable a(m)
minimize norm(a,1) + eta/epsilon*norm(a,1)
subject to 
A * a == b
cvx_end

y_OR = [y(1:m)',zeros(1,length(y)-m)];
for i=1:length(y)-m
    x = i+1:m+i;
    y_OR(m+i) = a'*y(x);
end

plot(1:length(y),y,'-b',1:length(y),y_OR,'r-.')
legend('true signal','Optimal Recovery')

% 1-position delay
%norm(y_OR-y,inf)/length(y)

