% El tamaño de la entrada a la red esta dada por num_inputs CON EL BIAS
% V es el vector que me da los valores binarios, V(1)=0 / V(1)=-1 , V(2)=1
% V(1) es FALSE, V(2) es TRUE
% Detecta paridad impar, poniendo un 1 a la salida
function [train_inputs, expected_outputs, num_patterns] = generatePatterns(min, max, cant)

x1 = linspace(min, -4, cant*200);

x2 = linspace(-4,4,cant*5);

x3 = linspace(4,max,cant*200);

x = [x1 x2 x3];

y = 5.*sin(x).^2+cos(x) ;

num_patterns = size(x);

num_patterns = num_patterns(2);

train_inputs = [x' -ones(num_patterns, 1)];
expected_outputs = y';