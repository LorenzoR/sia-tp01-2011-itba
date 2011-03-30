% Derivada expresada en funcion de si misma
function y = expDeriv(h, beta)

y = 2 * beta * exponential(h, beta) .* (1 - exponential(h, beta));