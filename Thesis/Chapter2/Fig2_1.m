% This file is used to produce Fig 2.1 in my thesis

f = @(x) sin(pi*x.^3+x.^2+1);
m = 8;
eqpts = linspace(-1,3,m);

% plot of function passing through points
y = f(eqpts);

% interpolation polynomial
c_least = polyfit(eqpts,y,m-1);

% plot of original function, piecewise linear function and polynomial
grid = linspace(-3.1,3.1,201);
plot(eqpts,y,'ok',grid,f(grid),'b-.',eqpts,y,'k--',grid,polyval(c_least,grid),'r-','LineWidth',1)
xlim([-1,3.1])
ylim([-1.5,3])
legend('Datapoints', 'trig function', 'piecewise-linear','polynomial','Location','best');
title('Fit a univariate function through finite points')
