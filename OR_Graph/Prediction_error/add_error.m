function er = add_error(x,d,m,eta)
% Add noise, with 2-norm error bounded by eta,
% to the vector x using model m:
%
%  m = 1: add random gaussian noise
%  m = 2: add uniform noise
%  m = 3: add noise proportional to degree
%  m = 4: add noise proportional to inverse of degree
%  m = 5: add all noise to the max degree node

k = numel(x);

ernorm = rand*eta;  % error between 0 and eta

switch m
    case 5
        [~,t] = max(d);
        er = zeros(k,1);
        er(t) = 1;
    case 2
        er = ones(k,1);
    case 3
        er = d;
        er = er - mean(er);
    case 4
        er = 1./d;
        er = er - mean(er);
    otherwise
        er = rand(k,1);
        er = er - mean(er);
end
        
er = ernorm*er/norm(er);