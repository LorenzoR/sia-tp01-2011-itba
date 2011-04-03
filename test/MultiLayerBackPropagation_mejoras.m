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
patterns =generatePatterns(-10, 10, num_samples);
%[train_inputs, train_outputs, num_patterns] = generatePatterns(-10, 10, num_samples);            
%[train_inputs, train_outputs, num_patterns] = generatePatterns(-3, 3, num_samples);            
%[train_inputs, train_outputs, num_patterns] = generatePatternsParity([-1 1], num_inputs);            

%------------------------  Agregamos controles  --------------------------
%add button for early stopping
hstop = uicontrol('Style','PushButton','String','Stop', 'Position', [5 5 70 20],'callback','earlystop = 1;'); 
earlystop = 0;

%add button for resetting weights
hreset = uicontrol('Style','PushButton','String','Reset Wts', 'Position', get(hstop,'position')+[75 0 0 0],'callback','reset = 1;'); 
reset = 0;

%add slider to adjust the learning rate
hlr = uicontrol('Style','slider','value',LR,'Min',LR*0.1,'Max',LR*100,'SliderStep',[LR LR*10],'Position', get(hreset,'position')+[75 0 100 0], 'callback', 'changeLR=1;');
changeLR = 0;
%-------------------------------------------------------------------------

% Para salvar info de la mejor solucion
min_error = 1000;
selected_num_hidden = -1;
selected_epoch = -1;
selected_out_val = -1;
selected_error = -1;
selected_err = -1;

solutionFound = false;
intentos = 0;
%TEST
%weights_scale_array = [0.1 0.3 0.5 0.8 1];
%weights_scale = weights_scale_array(1);
while( solutionFound == false )
%for w=1:20
    fprintf('Intento %d\n', intentos);
    intentos = intentos +1;

    
    % Creamos la red segun la arquitectura definida 
    nn = newNetwork( layers, neuronsPerLayer);
    
    % Inicializamos variables auxiliares para el alg de backpropagation
    aux = cell(layers,1);
    h = cell(nn.layers);
        
    % Se inicializa con un error grande para que no termine inmediatamente
    prev_error = 1000;
    err_iter = 0;    
    err = zeros(1, epochs);    
    
    % Muestro el valor inicial de LR
    alr = get(hlr,'value');
    LR = alr;
    %fprintf(' ===> LR: %f\n', LR); 
    
    % Salvamos el parametro del momentum, para recuperarlo cuando el LR 
    % adaptativo hace regresar al paso anterior y hace alpha=0
    auxAlpha = alpha;
    
    for iter = 1 : epochs
        alr = get(hlr,'value');
        LR = alr;
          
        if( printEpochsMessages && mod(iter,messagePerEpochs) == 0 )
            fprintf('Epoque %d\n', iter);
        end
        
        % Salvo los pesos de la red antes de empezar otra epoca
        if adaptativeLR
            nn.previousWeights = nn.weights;
        end        
        
        % Forzamos que se cicle por todos los patrones en forma aleatoria        
        pat_num_array = randperm(patterns.num_patterns);
        for j = 1 : patterns.num_patterns
            % Estructura para guardar el calculo de deltas de la red	
            delta = cell(nn.layers, 1);
            
            % Selecciono un patron al azar
            pat_num = pat_num_array(j);
            
            % El patron seleccionado es el patron actual en el ciclo
            selected_pat = patterns.train_inputs(pat_num,:);
            expected_out = patterns.train_outputs(pat_num,:);
    
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
            % BACKPROPAGATION: Calculo hacia atras             
            % error = expected_out - out_val;
            % delta{nn.layers} = error;            
            delta{nn.layers} = expected_out - out_val;
            
            % Calculo el delta de las capas ocultas
            for k = nn.layers-1:-1:1			                                
                %temp = delta{k+1} * nn.weights{k+1}(1:nn.neuronsPerLayer(k+1)-1,:)';
                %delta{k} = temp .* feval(activationFunctionDeriv, aux{k+1}(:,1:nn.neuronsPerLayer(k+1)-1), beta);
                delta{k} = delta{k+1} * nn.weights{k+1}(1:nn.neuronsPerLayer(k+1)-1,:)'.* feval(activationFunctionDeriv, aux{k+1}(:,1:nn.neuronsPerLayer(k+1)-1), beta);
            end

            % Actualizo la matriz de pesos de cada capa
            for k = 1:nn.layers                                    
                nn.weightsChange{k} = LR .* ( aux{k}' * delta{k} ) + momentum * ( alpha .* nn.weightsChange{k} );
                nn.weights{k} = nn.weights{k} + nn.weightsChange{k};
            end
            % FIN ALGORITMO BACKPROPAGATION
        end
        
        % ---------------   Termino otra epoca   --------------------------       
        
        % Calculo la salida para todos los patrones
        aux{1} = patterns.train_inputs;
        for j = 1:nn.layers
            h{j} = aux{j} * nn.weights{j};
            aux{j+1} = feval(activationFunction, h{j}, beta);            
            % Agrego el umbral, salvo en la capa de salida
            if j == nn.layers
                aux{j+1} = h{j};
            elseif j == nn.layers-1 && linealAdicional
                aux{j+1} = [h{j} -ones(patterns.num_patterns,1)];
            else
                aux{j+1} = [feval(activationFunction, h{j}, beta) -ones(patterns.num_patterns,1) ];	
            end
        end            
        out_val = aux{j+1};
                
        % Calculo error  
        %error_patterns = train_outputs - out_val;
        %error_all = (sum(error_patterns.^2)')/2;	
        error_all = (sum((patterns.train_outputs - out_val).^2)')/2;	
        
        % Finalmente se los promedia para poder compararlo con error_tolerance
        err(iter) = sum(error_all)/patterns.num_patterns;
            
        % Aplico LR variable
        if adaptativeLR && iter ~= 1
            delta_error = err(iter) - err(iter-1);        
            if( delta_error > 0 )
                delta_LR = -b*LR;
                % Deshacer los cambios en los pesos de la red
                nn.weights = nn.previousWeights;
                alpha = 0;
            elseif ( delta_error < 0 )
                delta_LR = a;
                alpha  = auxAlpha;
            else
                delta_LR = 0;
                alpha  = auxAlpha;
            end
            LR = LR + delta_LR;
        end
        %------------------------------------------------------------------
        
        % Actualizo el grafico del error vs epocas
        figure(1);        
        plot(err(1:iter))

        % Reseteamos los pesos si se presiono el boton RESET
        if reset
            weight_input_hidden = (rand(num_inputs,num_hidden-1) - 0.5).* weights_scale;
            weight_hidden_output = (rand(num_hidden,num_outputs) - 0.5).* weights_scale;
            fprintf('weights reset after %d epochs\n',iter);
            %plot(err);
            %plot(err(1:iter));
            reset = 0;
        end

        % Detenemos el aprendizaje si se presiono el boton STOP
        if earlystop
            fprintf('stopped at epoch: %d, error: %f\n', iter, err(iter)); 
            break; 
        end

        % Si se cambio el LR, se imprime un aviso
        if changeLR
            fprintf('LR: %f\n', LR); 
            changeLR = 0;
        end

        
        % Si estoy en un min local, me voy y comienzo otra vez
        if iter > 1
            err_ant = err(iter-1);
        else
            err_ant = 1000;
        end    
        err_rel_porcentual = ( abs(err_ant - err(iter)) / err_ant )*100;
        if( err_rel_porcentual < err_rel_porcentual_min )
            fprintf('La red esta en un Minimo Local\n');
            fprintf('\t ==> Epoch: %d with error: %6.4f\n', iter, err(iter));            
            %close;  % para cerrar cualquier figure que este abierta
            %plot(err(1:iter));
            %return;
            break;
        end
        
        % Se cumple la condicion de cota min de error, termino el entrenamiento 
        if abs(err(iter)) < error_tolerance   
            solutionFound = true;
            % Se salva info de la mejor solucion
            if err(iter) < min_error
                min_error = err(iter);
                selected_epoch = iter;
                selected_out_val = out_val;
                %selected_error = error;
                selected_err = err;
            end
            break;        
        end
        
    end     % del FOR que itera por las epocas

    %fprintf('stopped at epoch: %d, error: %f\n', iter, err(iter));
    solutionFound = true;
    
    
    % Me fijo si ocurrio max cant de epocas, o si se encontro una solucion
    if selected_epoch == -1
        fprintf('La red no pudo aprender en %d epocas ..  err(iter)=%f\n', epochs, err(iter));        
    else
        fprintf('Covergio en la epoca: %d ,  error: %6.4f\n\n',selected_epoch,min_error);        
        
        % Grafico el error cuadratico medio vs las epocas
        plot(selected_err(1:selected_epoch))
        xlabel('Epocas')
        ylabel('Error')        
        % Salvo la imagen en formato eps
        %print('Error.eps','-deps')
    end
    
    % Calculo la generalizacion
    aux{1} = patterns.generalization_inputs;
    aux2{1} = patterns.train_inputs;
    for j = 1:nn.layers
        h{j} = aux{j} * nn.weights{j};
        h2{j} = aux2{j} * nn.weights{j};
        aux{j+1} = feval(activationFunction, h{j}, beta);            
        aux2{j+1} = feval(activationFunction, h2{j}, beta);            
        % Agrego el umbral, salvo en la capa de salida
        if j == nn.layers
            aux{j+1} = h{j};
            aux2{j+1} = h2{j};
        elseif j == nn.layers-1 && linealAdicional
            aux{j+1} = [h{j} -ones(patterns.num_patterns,1)];
            aux2{j+1} = [h2{j} -ones(patterns.num_patterns,1)];
        else
            aux{j+1} = [feval(activationFunction, h{j}, beta) -ones(patterns.num_patterns,1) ];	
            aux2{j+1} = [feval(activationFunction, h2{j}, beta) -ones(patterns.num_patterns,1) ];	
        end
    end            
    gen_out_val = aux{j+1};
    out_val = aux2{j+1};

    % Calculo de porcentaje de aprendizaje
    count = 0;
    learning_error = patterns.train_outputs - out_val;
    for i=1:patterns.num_patterns
        if abs(learning_error(i)) < generalizationTolerance
            count = count + 1;
        end
    end
        
    fprintf('Porcentaje de Aprendizaje\n');
    percentage_learning = count / patterns.num_patterns
    
    % Calculo de porcentaje de generalizacion
    count = 0;
    generalization_error = patterns.generalization_outputs - gen_out_val;
    for i=1:patterns.num_patterns
        if abs(generalization_error(i)) < generalizationTolerance
            count = count + 1;
        end
    end
        
    fprintf('Porcentaje de generalizacion\n');
    percentage_generalization = count / patterns.num_patterns
        
end     % del WHILE principal


    