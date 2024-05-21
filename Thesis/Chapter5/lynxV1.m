%%%%% Optimal Recovery in Time Series %%%%%%%%

% Data set description is on the paper:
% Optimal Error and GMDH Predictors A Comparison with Some Statistical Techniques

%% data loading and processing
clc; clear all;
data = readtable('lynx.csv');
y = data(:,2);
y = table2array(y);
y_log = log10(y);
y_log = y_log-mean(y_log);
%y_log = y;
%% Optimal Recovery Method

m = 15;         % Maximum delay
T = 50;         % approximate period
eta = 0.9;        % ell_2 bounded observation error
epsilon = 1;    % approximation parameter

x = 1:m;          
A = [ones(1,m);cos(2*pi*x/T)];
b = [1; cos(2*pi*(m+1)/T)];
cvx_begin quiet
variable a(m)
minimize norm(a,1) + eta/epsilon*norm(a,2)
subject to 
A * a == b
cvx_end

y_OR = [y_log(1:m)',zeros(1,length(y)-m)];
for i=1:length(y)-m
    x = i+1:m+i;
    y_OR(m+i) = a'*y_log(x);
end
%figure(1)
%plot(1:length(y_OR),y_OR,'-b',1:length(y_log),y_log,'r-.')
%legend('Optimal Recovery','true')
% 1 position delay?

%% Tong 1977 Model 1
m = 11;
a_1 = [-0.316;0.218;0.178;-0.03;0.019;-0.117;0.192;-0.350;0.273;-0.518;1.128];
y_1 = [y_log(1:m)',zeros(1,length(y)-m)];
for i=1:length(y)-m
    x = i:m+i-1;
    y_1(m+i) = a_1'*(y_log(x)) + sqrt(0.0437)*randn; 
end
%figure(2)
%plot(1:length(y_1),y_1,'-b',1:length(y_log),y_log,'r-.')

%% Tong 1977 Model 2
m = 11;
a_2 = [-0.3622;0.3224;0;0;0;0;0;-0.1265;0;-0.3571;1.0938];
y_2 = [y_log(1:m)',zeros(1,length(y)-m)];
for i=1:length(y)-m
    x = i:m+i-1;
    y_2(m+i) = a_2'*y_log(x) + sqrt(0.04405)*randn; 
end
%figure(3)
%plot(1:length(y_2),y_2,'-b',1:length(y_log),y_log,'r-.')

%% GMDH
m = 2;
y_3 = [y_log(1:m)',zeros(1,length(y)-m)];
for i=1:length(y)-m
    y_3(m+i) = 0.076 + 1.365*y_log(m+i-1) - 0.772*y_log(m+i-2) + 0.08*y_log(m+i-1) -...
        0.427*y_log(m+i-2) + 0.145*y_log(m+i-1)*y_log(m+i-2);
end
%figure(4)
%plot(1:length(y_3),y_3,'-b',1:length(y_log),y_log,'r-.')

%% 
figure(1)
plot(1:length(y_log),y_log,'k-',1:length(y_OR),y_OR,'r-.',1:length(y_1),y_1,'b--')
legend('true','Optimal Recovery','Linear autoregression')
%%
[norm(y_OR-y_log','inf'),norm(y_1-y_log','inf'),norm(y_2-y_log','inf'),norm(y_3-y_log','inf')]
[norm(y_OR-y_log'),norm(y_1-y_log'),norm(y_2-y_log'),norm(y_3-y_log')]


