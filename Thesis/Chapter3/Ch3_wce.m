% This file contains the computation of worst-case error using S-Lemma.
% We also verify the following items numerically:
% 1. We can compute the local worst-case error for any linear quantity of
% interest $Q$
% 2. We verify that both S-Lemma method (in my thesis) and computation (Theorem
% 10.4 and Eq 10.7-10.8) from reference [1] can compute the local worst-case error
% 3. Direct method and indirect method are same (proved in section 3.1.3)
% 4. According to definition, global worst-case error is greater than local worst-case error. 
% Here we check it numerically when approximating a linear functional.

%% Item 1: Compute local worst-case error for any linear quantity of interest
clear all;clc;
% generate the approximation space V and the observation map L
N = 40;                  % dimension of the ambient Hilbert space
n = 10;                  % dimension of the space V
epsilon = 0.1;           % the approximation parameter
V = randn(N,n);          % the columns of this matrix form a basis of the space V
V = V*inv(sqrtm(V'*V));  % the columns now form an orthornomal basis 
m = 20;                  % number of observations
L = randn(m,N);          % the observation map
% generate an element in the approximability set and its observation vector
aux = randn(N,1);
f = V*rand(n,1) + 2/3*epsilon*aux/norm(aux);
y = L*f;

% define the Gramian Gu and the cross-Gramian C (the Gramian Gv is the identity)
Gu = L*L';
C = L*V;
% produce f_star
Gu_inv = inv(Gu);
b = (C'*Gu_inv*C)\(C'*Gu_inv*y);
a = Gu_inv*(y-C*b);
f_star = L'*a + V*b;

% linear quantity of interest
s = 15;                % Q(f) is element in R^s
q = rand(N,s);         % Q(f) = q'*f


P1 = eye(N)-V*V';     % the orthogonal projector onto the orthogonal space to V
z = q'*f_star;
[Q,~] = qr(L');
K = Q(:,m+1:N);       % the columns form an orthonormal basis for ker(L) 
h = L'*((L*L')\y);    % the element orthogonal to ker(L) for which L(h)=y
cvx_begin quiet
variable c
variable d nonnegative
minimize c
subject to 
[K'*(d*P1-q*q')*K, K'*(d*P1*h-q*q'*h+q*z);...
    (d*h'*P1-h'*q*q'+z'*q')*K, d*(norm(P1*h)^2-epsilon^2)+c-norm(q'*h-z)^2] == semidefinite(N-m+1)
cvx_end
wce = sqrt(c);
fprintf('The local worst-case error of recovering Q is %.4f\n',wce)

%% Item 2: Two methods can be used to compute the local worst-case error
clear all;clc;
% Problem setting: the same as Item 1
N = 40;                  
n = 10;                  
epsilon = 0.1;           
V = randn(N,n);          
V = V*inv(sqrtm(V'*V));  
m = 20;                  
L = randn(m,N);          
aux = randn(N,1);
f = V*rand(n,1) + 2/3*epsilon*aux/norm(aux);
y = L*f;

% define the Gramian Gu and the cross-Gramian C (the Gramian Gv is the identity)
Gu = L*L';
C = L*V;
% produce f_star
Gu_inv = inv(Gu);
b = (C'*Gu_inv*C)\(C'*Gu_inv*y);
a = Gu_inv*(y-C*b);
f_star = L'*a + V*b;

% Method 1: Theorem 10.4 and Eq 10.7-10.8 from reference [1]
mu = 1/sqrt(min(eig(C'*Gu_inv*C)));
P1 = eye(N)-V*V';
wce1 = mu*sqrt(epsilon^2 - norm(P1*f_star)^2);
fprintf('The local worst-case error computed by the first method is %.4f\n',wce1)

% Method 2: Use S-Lemma to recast the original optimization problem as SDP
z = f_star;
[Q,~] = qr(L');
K = Q(:,m+1:N);       % the columns form an orthonormal basis for ker(L) 
h = L'*((L*L')\y);    % the element orthogonal to ker(L) for which L(h)=y
cvx_begin quiet
variable c
variable d nonnegative
minimize c
subject to 
[K'*(d*P1-eye(N))*K, K'*(d*P1*h-h+z);...
    (d*h'*P1-h'+z')*K, d*(norm(P1*h)^2-epsilon^2)+c-norm(h-z)^2] == semidefinite(N-m+1)
cvx_end
wce2 = sqrt(c);
fprintf('The local worst-case error computed by solving SDP is %.4f\n',wce2)

%% Item 3 and Item 4: Recovering a linear functional
clear all;clc;
% Problem setting:
N = 40;                  
n = 10;                   
epsilon = 0.1;           
V = randn(N,n);          
V = V*inv(sqrtm(V'*V)); 
m = 20;                  
L = randn(m,N);          

aux = randn(N,1);
f = V*rand(n,1) + 2/3*epsilon*aux/norm(aux);
y = L*f;

% define the Gramian Gu and the cross-Gramian C (the Gramian Gv is the identity)
Gu = L*L';
C = L*V;
% produce f_star
Gu_inv = inv(Gu);
b = (C'*Gu_inv*C)\(C'*Gu_inv*y);
a = Gu_inv*(y-C*b);
f_star = L'*a + V*b;

% linear quantity of interest
q = rand(N,1);        % recovery of linear functional Q(f) = q*f

P1 = eye(N)-V*V';     
z = q'*f_star;
[Q,R] = qr(L');
K = Q(:,m+1:N);       
P2 = K*K';            
h = L'*((L*L')\y);    
cvx_begin quiet
variable c
variable d nonnegative
minimize c
subject to 
[K'*(d*P1-q*q')*K, K'*(d*P1*h-q*q'*h+q*z);...
    (d*h'*P1-h'*q*q'+z'*q')*K, d*(norm(P1*h)^2-epsilon^2)+c-norm(q'*h-z)^2] == semidefinite(N-m+1)
cvx_end
wce = sqrt(c);

% Compute global worst-case error
% Optimization method:
cvx_begin quiet
variable a(m)
minimize norm(q-L'*a)
subject to
C'*a == V'*q
cvx_end
wce1 = epsilon*norm(q-L'*a);

% another way to compute:
Delta_V = L'*Gu_inv - (L'*Gu_inv*C-V)*inv(C'*Gu_inv*C)*C'*Gu_inv;
a_delta = q'*Delta_V;
wce2 = epsilon*norm(q-L'*a_delta');

fprintf('Our goal is to recovery a linear functional Q(f).\n')
fprintf('Numerically, we verify that the direct method and indirect method are same by looking at\n')
fprintf('the norm of difference produced coefficient vectors, which is %.4s.\n',norm(a-a_delta'))
fprintf('We also notice that the produced local worst-case error %.4f is smaller than global worst-case error %.4f',wce,wce1)

%% Reference
% [1] S. Foucart
% "Mathematical Pictures at a Data Science Exhibition",
% 2022, Cambridge University Press. 