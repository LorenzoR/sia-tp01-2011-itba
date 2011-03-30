% Derivada expresada en funcion de si misma
function y = tanHipDeriv(h, beta)

%y = beta * (1 - tanHip(h, beta).^2);
y = beta * (1 - tanh(beta*h) .^2);

