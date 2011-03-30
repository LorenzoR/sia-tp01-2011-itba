% Utilizada para unidades entre 0 y 1
function y = exponential(h, beta)

y = 1 ./ (1 + exp(-2 * beta * h));