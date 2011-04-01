function network = newNetwork(n, neuronsPerLayerArray)
% n: cantidad de capas ( hidden layers + 1).
% neuronsPerLayerArray: cada elemento de este arreglo indica la cantidad de 
% 			neuronas en cada capa(entrad, ocultas, salida). 
% 			Se debe cumplir => length(neuronsPerLayerArray) = n + 1
% ejemplo: nn = newNetwork(2, [2 3 1]); 
%   2 capas (entrada + capas ocultas)
% 	2 (primer elemento del arrgelo) es para la cantidad de entradas 
%	3 neuronas en la capa ocult
%	1 es la cantidad de salidas 

network.layers = n;
% Agrego una neurona a cada capa para el umbral (bias)
network.neuronsPerLayer = neuronsPerLayerArray + 1;

% Le quito la neurona extra a la capa de salida, ya que esta capa no requiere umbral
network.neuronsPerLayer(length(network.neuronsPerLayer)) = network.neuronsPerLayer(length(network.neuronsPerLayer)) - 1;

% Cell es una forma "matricial" donde puedo guardar elementos de distintos tamaños
% Inicializo las matrices para guardar los pesos de la red, los cambios de peso,
% y los cambios en los indices de aprendizaje de la red
network.weights = cell(n,1);
network.weightsChange = cell(n,1);
network.previousWeights = cell(n,1);

for i = 1:n
     % Inicializa la matriz de pesos con valores aleatorios entre -0.5 y 0.5
     % En neuronsPerLayerArray(i)+1, el +1 es por el bias de la capa i
     network.weights{i} = rand(neuronsPerLayerArray(i)+1, neuronsPerLayerArray(i+1)) - 0.5;
     %network.weights{i} = rand(neuronsPerLayerArray(i)+1, neuronsPerLayerArray(i+1))*0.1 - 0.05;
     
     % Inicializa la matriz de variacion de pesos
     network.weightsChange{i} = zeros(neuronsPerLayerArray(i)+1, neuronsPerLayerArray(i+1));

     % Para learning rate adaptativo, esta variable siempre guarda los
     % pesos de la epoca anterior
     network.previousWeights{i} = zeros(neuronsPerLayerArray(i)+1, neuronsPerLayerArray(i+1));
end
