function loadParams
globalParams;

% Parametros para determinar cuando cortar los ciclos
error_tolerance = 0.001;                % Cota de error para corte del algoritmo
err_rel_porcentual_min  = 0.000001;     % Para identificar cuando llegamos a un min local
epochs = 1000;                        % Cota para corte por epocas( minimo 50 epocas )

generalizationTolerance = 0.3;

printEpochsMessages = false;             % Determina si se imprime o no el numero de epocas
messagePerEpochs = floor( epochs / 50); % Cada cuantas epocas imprime mensaje

% Parametros de la red
num_inputs = 1+1;   % Entradas, incluye el BIAS
num_outputs = 1;    % Salidas     
num_hidden = 1000+1;   % numero de neuronas en la capa oculta, incluye el BIAS    
num_samples = 500;%2000   % Cantidad de muestras que se toman en el intervalo de la funcion

% Para redes Multicapas. La cant de neuronas es sin tener en cuenta el BIAS
layers = 2;          % Para definir la cantidad de capas de la red(entrada + capas ocultas)
neuronsPerLayer = [1 100 1]; % Arreglo que define la cant de unidades en cada capa

LR = 0.0007;           % Indice de aprendizaje
adaptativeLR = false;
a = LR/2;
b = LR/2;

momentum = false;
alpha = 0.0001;

%Funciones de activacion pa1a las capas ocultas
%activationFunction =  'exponential';
activationFunction =  'tanHip';
%activationFunctionDeriv = 'expDeriv'; 
activationFunctionDeriv = 'tanHipDeriv';

% true=la capa anterior a la salida tiene activacion lineal
% false=tiene activacion determinada por "activationFunction"
linealAdicional = false;


% Funcion de activacion para las unidades de salida UNICAMENTE
outputActivationFunction = 'lineal';
outputActivationFunctionDeriv = 'linealDeriv';

beta = 2;           % Constante de escalamiento para la funcion de activacion
weights_scale = 0.1;  %0.1; %1/sqrt(num_inputs)  % Escala los pesos iniciales de la red, para evitar saturacion

% Decido como genero los patrones en el intervalo de trabajo de la funcion
samplingType = 'uniform'; %'moreOnEdges';
edgesRatio = 10;

