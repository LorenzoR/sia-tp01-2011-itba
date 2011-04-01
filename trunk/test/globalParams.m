% Definimos la variables globales de la red neuronal

clear;

% Parametros para determinar cuando cortar los ciclos
global error_tolerance;
global epochs;
global printEpochsMessages; % Booleano para determinar si imprimo los mensajes de epocas
global messagePerEpochs     % Cada cuantas epocas imprime mensaje de informacion
global err_rel_porcentual_min % Cota para determinar si llegamos a un min local

% Parametros de la red
global num_inputs;      % Entradas (este numero incluye el "bias")
global num_outputs;     % Salidas     
global num_hidden;      % Neuronas en la capa oculta ( incluye el "bias") 
global num_samples;     % Cantidad de muestras que se generan en el intervalo de la funcion    
global layers;          % Para definir la cantidad de capas de la red(entrada + capas ocultas)
global neuronsPerLayer; % Arreglo que define la cant de unidades en cada capa


global num_patterns;    % Cant de patrones para entrenar la red. Se calcula dinamicamente
global added_patterns;  % Cant de patrones agregados. Se calcula dinamicamente

global LR;              % Indice de aprendizaje

%LEARNING RATE ADAPTATIVO
global adaptativeLR;
global a;
global b;

% MOMENTUM
% Indica si la red usara momentum. Puede ser true o false.
global momentum;
% Parametro para momentum
global alpha;


% Funcion de activacion para las capas ocultas
global activationFunction;
global activationFunctionDeriv;

global linealAdicional; %true=la capa anterior a la salida tiene activacion lineal
                        % false=tiene activacion determinada por "activationFunction"

%Funcion de activacion para la capa de salida UNICAMENTE
global outputActivationFunction;
global outputActivationFunctionDeriv;

global beta;           % Constante de escalamiento para la funcion de activacion
global weights_scale;  % Para escalar los pesos iniciales de la red, y evitar saturacion rapida


% Para indicar si modifico los patrones de entrenamiento, para dirigir
% mejor el aprendizaje de la red
global modifyPatterns; 

% Define cuantos patrones se agregaran como un porcentaje de los patrones 
% de entrenamiento 
global percentage_patterns;