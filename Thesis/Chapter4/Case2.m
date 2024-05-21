%%%%%%%%% Optimal Recovery and System Identification %%%%%%
% This Matlab file is used to generate plots and tables in section 4.5 in
% my thesis.
% The example is motivated by the paper
% Optimal Algorithms Theory for Robust Estimation and Prediction

%% V={0} and bounded error
clear all;clc;
rng(5)        % comment out this line to generate different random numbers
% the true function u is known
u = @(t) cos(t.^2/10-5) + 1./(t.^2+1);
dt = 0.2;
% we try to approximate coefficient vector a
x = @(a,t) a(1) * u(t-2*dt) + a(2) * u(t-dt) + a(3)*u(t);

% Generate the true coefficient vector
aux = rand(3,1);
% f is the true coefficient vector
f = 2/3*aux/norm(aux);
% approximation parameter
epsilon = 1;

% generate observations
T = 15;             % number of observations
% observation map
L = [u((-1:(T-2))'*dt), u((0:(T-1))'*dt), u((1:T)'*dt)];

eta = 3;         % change eta here, I use eta = 0.5, 1, 2, 3.

T_test = 10000;
time = linspace(0,10,T_test);
x_true = x(f,time);

monte = 50;
x_loc = zeros(monte,T_test);
x_ls = zeros(monte,T_test);
or_app_err = zeros(1,monte);
or_pre_err = zeros(1,monte);
ls_app_err = zeros(1,monte);
ls_pre_err = zeros(1,monte);

for i=1:monte
    
    % generate error
    err = eta/norm(L*f)*abs(L*f).*(2*randi(2,T,1)-3);
    y = L*f + err;

    % local optimal recovery
    opt_tau = max(1-eta/norm(y),0);
    if opt_tau == 0
        f_loc = zeros(3,1);
    else
        f_loc = (opt_tau*L'*L+(1-opt_tau)*eye(3))\(opt_tau*L'*y);
    end
    or_app_err(i) = norm(f-f_loc);
    
    or_pre_err(i) = norm(x_true-x(f_loc,time));
    x_loc(i,:) = x(f_loc,time);
    
    % Least square
    f_ls = (L'*L)\(L'*y);
    ls_app_err(i) = norm(f-f_ls);
    
    ls_pre_err(i) = norm(x_true-x(f_ls,time));
    x_ls(i,:) = x(f_ls,time);
end

% plot and result
[mean(or_app_err),mean(ls_app_err)]
[std(or_app_err),std(ls_app_err)]
[mean(or_pre_err),mean(ls_pre_err)]
[std(or_pre_err),std(ls_pre_err)]

%
figure(1)
hold on
plot(time,x_true,'-k',time,mean(x_loc),'b--',time,mean(x_ls),'r-.','LineWidth',1.5)
xlabel('Time','Fontsize',14)
ylabel('Function Value','Fontsize',14)

%plot(time,mean(x_loc)+2*std(x_loc),'r-.',time,mean(x_loc)-2*std(x_loc),'r-.')
time1 = [time, fliplr(time)];
inBetween = [mean(x_loc)+2*std(x_loc), fliplr(mean(x_loc)-2*std(x_loc))];
fill(time1, inBetween, 'b','FaceAlpha',0.4);

%plot(time,mean(x_ls)+2*std(x_ls),'r-.',time,mean(x_ls)-2*std(x_ls),'r-.')
inBetween = [mean(x_ls)+2*std(x_ls), fliplr(mean(x_ls)-2*std(x_ls))];
fill(time1, inBetween, 'r','FaceAlpha',0.1);
legend('true function','Local OR','Least-squares','Two standard deviations for LOR','Two standard deviations for LS','Fontsize',12,'Location','best')
hold off