function loadParams
globalParams;

% Parametros para determinar cuando cortar los ciclos
error_tolerance = 0.001;                % Cota de error para corte del algoritmo
err_rel_porcentual_min  = 0.000001;     % Para identificar cuando llegamos a un min local
epochs = 20000;                        % Cota para corte por epocas( minimo 50 epocas )

printEpochsMessages = false;             % Determina si se imprime o no el numero de epocas
messagePerEpochs = floor( epochs / 50); % Cada cuantas epocas imprime mensaje

% Parametros de la red
num_inputs = 1+1;   % Entradas, incluye el BIAS
num_outputs = 1;    % Salidas     
num_hidden = 1000+1;   % numero de neuronas en la capa oculta, incluye el BIAS    
num_samples = 500;   % Cantidad de muestras que se toman en el intervalo de la funcion

LR = 0.001;           % Indice de aprendizaje

%Funciones de activacion para las capas ocultas
%activationFunction =  'exponential';
activationFunction =  'tanHip';
%activationFunctionDeriv = 'expDeriv'; 
activationFunctionDeriv = 'tanHipDeriv';

% Funcion de activacion para las unidades de salida UNICAMENTE
outputActivationFunction = 'lineal';
outputActivationFunctionDeriv = 'linealDeriv';

beta = 10;           % Constante de escalamiento para la funcion de activacion
weights_scale = 1;  %0.1; %1/sqrt(num_inputs)  % Escala los pesos iniciales de la red, para evitar saturacion

% Para determinar si se aplica o no la modificacion de los patrones de 
% entrenamiento, con el fin de dirigir mejor el aprendizaje de la red
added_patterns = 0; %Hay que asignarla en el caso que modifyPatterns sea False
modifyPatterns = false;

% Define cuantos patrones se agregaran como un porcentaje de los patrones 
% de entrenamiento 
percentage_patterns = 4;