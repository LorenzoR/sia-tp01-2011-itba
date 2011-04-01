%==============================================================
% TPE1: Entrega Preliminar Redes Neuronales
%==============================================================
% Materia: 
%		Sistemas de Inteligencia Artificial
% Grupo: 
% Integrantes:
%		Argume, Hugo
%		Argume, Robert
%      	Rodrigo, Lorenzo
% ITBA, 2011
%==============================================================
% Cargo los parametros para la red
globalParams;
loadParams();

% Genero las entradas y las salidas esperadas
[train_inputs, train_outputs, num_patterns] = generatePatterns(-10, 10, num_samples);            
%[train_inputs, train_outputs, num_patterns] = generatePatternsParity([-1 1], num_inputs);            

%--------- add some control buttons ---------
%add button for early stopping
hstop = uicontrol('Style','PushButton','String','Stop', 'Position', [5 5 70 20],'callback','earlystop = 1;'); 
earlystop = 0;

%add button for resetting weights
hreset = uicontrol('Style','PushButton','String','Reset Wts', 'Position', get(hstop,'position')+[75 0 0 0],'callback','reset = 1;'); 
reset = 0;

%add slider to adjust the learning rate
hlr = uicontrol('Style','slider','value',LR,'Min',LR*0.1,'Max',LR*100,'SliderStep',[LR LR*10],'Position', get(hreset,'position')+[75 0 100 0], 'callback', 'changeLR=1;');
changeLR = 0;


% Para salvar info de la mejor solucion
min_error = 1000;
selected_num_hidden = -1;
selected_epoch = -1;
selected_out_val = -1;
selected_error = -1;
selected_err = -1;
solutionFound = false;
    
    % Creamos la red segun la arquitectura definida 
    nn = newNetwork( layers, neuronsPerLayer);
    
    % Inicializa variable aux
    aux = cell(layers,1);
    
    %--- Se entrena la red con backtracking ---------
    % Se inicializa con un error grande para que no termine inmediatamente
    prev_error = 1000;
    err_iter = 0;    
    %err = zeros(1, epochs);    
    
    alr = get(hlr,'value');
    LR = alr;
    fprintf('LR: %f\n', LR); 
    
    % Salvamos el parametro del momentum, para recuperarlo cuando el LR 
    % adaptativo hace regresar al paso anterior
    auxAlpha = alpha;
    
    for iter = 1 : epochs
        alr = get(hlr,'value');
        LR = alr;
          
        if( printEpochsMessages && mod(iter,messagePerEpochs) == 0 )
            fprintf('Epoque %d\n', iter);
        end
        
        % Salvo los pesos de la red antes de empezar otra epoca
        nn.previousWeights = nn.weights;
        
        % Forzamos que se cicle por todos los patrones en forma aleatoria        
        pat_num_array = randperm(num_patterns);
        for j = 1 : num_patterns
            % Estructura para guardar el calculo de deltas de la red	
            delta = cell(nn.layers, 1);
            
            % Selecciono un patron al azar
            pat_num = pat_num_array(j);
            
            % El patron seleccionado es el patron actual en el ciclo
            selected_pat = train_inputs(pat_num,:);
            expected_out = train_outputs(pat_num,:);
    
            % BACKPROPAGATION: Calculo hacia adelante
            aux{1} = selected_pat;
            for k = 1:nn.layers
                h{k} = aux{k} * nn.weights{k};
                % Agrego el umbral, salvo en la capa de salida
                if k == nn.layers
                    aux{k+1} = h{k}; % la capa de salida tiene activacion lineal
                else if k == nn.layers-1 && linealAdicional
                        aux{k+1} = [h{k} -1];
                    else
                        aux{k+1} = [feval(activationFunction, h{k}, beta) -1];
                    end
                end
            end
            out_val = aux{k+1};

            % Se calcula el error para el patron seleccionado
            error = expected_out - out_val;

            % BACKPROPAGATION: Calculo hacia atras             
            delta{nn.layers} = error;            

            % Calculo el delta de las capas ocultas
            for k = nn.layers-1:-1:1			                                
                temp = delta{k+1} * nn.weights{k+1}(1:nn.neuronsPerLayer(k+1)-1,:)';
                delta{k} = temp .* feval(activationFunctionDeriv, aux{k+1}(:,1:nn.neuronsPerLayer(k+1)-1), beta);
            end

            % Actualizo la matriz de pesos de cada capa
            for k = 1:nn.layers                                    
                nn.weightsChange{k} = LR .* ( aux{k}' * delta{k} ) + momentum * ( alpha .* nn.weightsChange{k} );
                nn.weights{k} = nn.weights{k} + nn.weightsChange{k};
            end
            % FIN ALGORITMO BACKPROPAGATION
        end
        
        % -- Termino otra epoca        
        % Calculo la salida para todos los patrones
        aux{1} = train_inputs;
        for j = 1:nn.layers
            h{j} = aux{j} * nn.weights{j};
            aux{j+1} = feval(activationFunction, h{j}, beta);            
            % Agrego el umbral, salvo en la capa de salida
            if j == nn.layers
                aux{j+1} = h{j};
            else if j == nn.layers-1 && linealAdicional
                    aux{j+1} = [h{j} -ones(num_patterns,1)];
                else
                    aux{j+1} = [feval(activationFunction, h{j}, beta) -ones(num_patterns,1) ];	
                end
            end
        end            
        out_val = aux{j+1};

        % Calculo error  
        error_patterns = train_outputs - out_val;
        error_all = (sum(error_patterns.^2)')/2;	
            
        % Finalmente se los promedia para poder compararlo con error_tolerance
        err(iter) = sum(error_all)/num_patterns;
            
        % aplico LR variable
        if iter ~= 1 && adaptativeLR
            delta_error = err(iter) - err(iter-1);        
            if( delta_error > 0 )
                delta_LR = -b*LR;
                % Deshacer los cambios en los pesos de la red
                nn.weights = nn.previousWeights;
                alpha = 0;
            elseif ( delta_error < 0)
                delta_LR = a;
                alpha  = auxAlpha;
            else
                delta_LR = 0;
                alpha  = auxAlpha;
            end
            LR = LR + delta_LR;
        end
        
    
        
    figure(1);
    plot(err)
    
    
    %reset weights if requested
    if reset
        weight_input_hidden = (rand(num_inputs,num_hidden-1) - 0.5).* weights_scale;
        weight_hidden_output = (rand(num_hidden,num_outputs) - 0.5).* weights_scale;
        fprintf('weights reaset after %d epochs\n',iter);
        iter = 1;
        cla;
        plot(err);
        reset = 0;
    end
    
    %stop if requested
    if earlystop
        fprintf('stopped at epoch: %d\n',iter); 
        break 
    end
    
    if changeLR
        fprintf('LR: %f\n', LR); 
        changeLR = 0;
    end
    
    %stop if error is small
    if err(iter) < error_tolerance
        fprintf('converged at epoch: %d  , error: %f\n',iter, err(iter));
        break 
    end
        

end

