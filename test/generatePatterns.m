% El tamaño de la entrada a la red esta dada por num_inputs CON EL BIAS
% V es el vector que me da los valores binarios, V(1)=0 / V(1)=-1 , V(2)=1
% V(1) es FALSE, V(2) es TRUE
% Detecta paridad impar, poniendo un 1 a la salida
function [train_inputs, expected_outputs, num_patterns] = generatePatterns(min, max, cant)

%min = min /10;
%max = max / 10;

step = (max-min)/cant;

x = min:step:max;

%y = 5.*sin(10*x).^2+cos(10*x) ;
y = 5.*sin(x).^2+cos(x) ;
%y = sin(x);

%expected_outputs = ( y' - 2 ) ./ 3;

num_patterns = size(x);

num_patterns = num_patterns(2);

train_inputs = [x' -ones(num_patterns, 1)];
expected_outputs = y';

% Normalizamos las entradas
% train_inputs = x;
% mu_inp = mean(train_inputs);
% sigma_inp = 2*std(train_inputs);
% train_inputs = (train_inputs(:,:) - mu_inp(:,1)) / sigma_inp(:,1);
% train_inputs = [train_inputs' -ones(num_patterns, 1)];


% Normalizamos las salidas
% train_out = y';
% mu_out = mean(train_out);
% sigma_out = 2*std(train_out);
% train_out = (train_out(:,:) - mu_out(:,1)) / sigma_out(:,1);
% expected_outputs = train_out;
