
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
            
% Agrego un conj patrones adicionales adecuados para guiar el aprendizaje de la red
%if (modifyPatterns)
%    [train_inputs, train_outputs, num_patterns] = modifyPatternsAND(train_inputs, train_outputs, num_patterns);
%end

%--------- add some control buttons ---------
%add button for early stopping
hstop = uicontrol('Style','PushButton','String','Stop', 'Position', [5 5 70 20],'callback','earlystop = 1;'); 
earlystop = 0;

%add button for resetting weights
hreset = uicontrol('Style','PushButton','String','Reset Wts', 'Position', get(hstop,'position')+[75 0 0 0],'callback','reset = 1;'); 
reset = 0;

%add slider to adjust the learning rate
%hlr = uicontrol('Style','slider','value',.1,'Min',.01,'Max',1,'SliderStep',[0.01 0.1],'Position', get(hreset,'position')+[75 0 100 0], 'callback', 'changeLR=1;');
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

% intentos = 0;
% while( earlystop == 0 && solutionFound == false )
%     fprintf('Intento %d\n', intentos);
%     intentos = intentos +1;
    
    
    
    % Inicializo los pesos de la red    
    %weight_input_hidden = (rand(num_inputs,num_hidden-1) - 0.5).* weights_scale;
    %weight_hidden_output = (rand(num_hidden,num_outputs) - 0.5).* weights_scale;
    
    % Creamos la red a 
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
    for iter = 1 : epochs
        alr = get(hlr,'value');
        LR = alr;
          
        if( printEpochsMessages && mod(iter,messagePerEpochs) == 0 )
            fprintf('Epoque %d\n', iter);
        end
        
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
            
            %%%%%%% AQUI
            
            % ALGORITMO BACKPROPAGATION
            % BACKPROPAGATION: Calculo hacia adelante
            aux{1} = selected_pat;
            for k = 1:nn.layers
                h{k} = aux{k} * nn.weights{k};
                % Agrego el umbral, salvo en la capa de salida
                if k == nn.layers
                    aux{k+1} = h{k}; % la capa de salida tiene activacion lineal
                else if k == nn.layers-1 
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
            % Calculo el delta de la capa de salida
            delta{nn.layers} = error;            

            % Calculo el delta de las capas ocultas
            for k = nn.layers-1:-1:1			                                
                temp = delta{k+1} * nn.weights{k+1}(1:nn.neuronsPerLayer(k+1)-1,:)';
                delta{k} = temp .* feval(activationFunctionDeriv, aux{k+1}(:,1:nn.neuronsPerLayer(k+1)-1), beta);
            end

            % Actualizo la matriz de pesos de cada capa
            for k = 1:nn.layers                                    
                nn.weightsChange{k} = LR .* ( aux{k}' * delta{k} );
                nn.weights{k} = nn.weights{k} + nn.weightsChange{k};
            end
            % FIN ALGORITMO BACKPROPAGATION
            
            
            %%%%%%% FIN AQUI
            
            
            
%             % Calculo los valores de salida de la red            
%             % Calculo el Pot de membrana h, y aplicamos la Fn de activacion
%             h = selected_pat*weight_input_hidden;
%             hidden_val = [feval(activationFunction, h, beta) -1];
%             
%             out_h = hidden_val*weight_hidden_output;            
%             % out_val = feval(outputActivationFunction, out_h, beta); %AQUI
%             out_val = out_h; %porque la salida tiene fn de activ lineal
%             
%             % Calculo las variaciones o deltas de error para la capa de salida y la oculta
%             %out_delta = (expected_out - out_val) * feval(outputActivationFunctionDeriv, out_h, beta); %AQUI
%             out_delta = expected_out - out_val ;%porque la salida tiene fn de activ lineal
%             
%             %hidden_delta = out_delta*weight_hidden_output(1:num_hidden-1,:)' .* feval(activationFunctionDeriv, hidden_val(1,1:num_hidden-1), beta);
%             hidden_delta = out_delta*weight_hidden_output' .* feval(activationFunctionDeriv, hidden_val, beta);
%             
%             % Hago backpropagation para los pesos de la capa de salida (oculta-salida)
%             weightChangeHO = LR*hidden_val' * out_delta;
%             weight_hidden_output = weight_hidden_output + weightChangeHO;
% 
%             % Hago backpropagation para los pesos de la capa oculta (entrada-oculta)
%             weightChangeIH = LR*selected_pat'*hidden_delta(:,1:num_hidden-1);
%             weight_input_hidden = weight_input_hidden + weightChangeIH;

        end
        
        % -- Termino otra epoca
        
    
        aux{1} = train_inputs;
        for j = 1:nn.layers
            h{j} = aux{j} * nn.weights{j};
            aux{j+1} = feval(activationFunction, h{j}, beta);            
            % Agrego el umbral, salvo en la capa de salida
            if j == nn.layers
                aux{j+1} = h{j};
            else if j == nn.layers-1
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
            
            
        
%         % Obtengo la salida para todos los patrones, luego de entrenar la red
%         h = train_inputs*weight_input_hidden;
%         hidden_val = [feval(activationFunction, h, beta) -ones(num_patterns,1)];
% 
%         out_h = hidden_val*weight_hidden_output;            
%         out_val = out_h;            
%         %out_val = feval(outputActivationFunction, out_h, beta);
%         
%         % obtengo los errores para cada uno de los patrones
%         error = train_outputs - out_val;
%    
%             
%         %Si hay patrones adicionales repetidos, non los considero al
%         %calcular el error
%         %if( modifyPatterns )
%         %    error = error(1:(2^(num_inputs-1)));
%         %end
% 
%         % Calculo el error cuadratico medio
%         error = (sum(error.^2)')/2;	
% 
%         % finalmente los promedio para poder compararlo contra error_tolerance
%         err(iter) = sum(error)/num_patterns;
%         %err(iter) = error;
    
        
    figure(1);
    plot(err)
    
    
    %reset weights if requested
    if reset
%         weight_input_hidden = (randn(inputs,hidden_neurons) - 0.5)/10;
%         weight_hidden_output = (randn(1,hidden_neurons) - 0.5)/10;
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
        
        % Si el error relativo porcentual del paso n al n+1 es muy pequeño,
        % entonces el error no disminuye apreciablemente. Seguramente estoy en un minimo local
%         if iter > 1
%             err_ant = err(iter-1);
%         else
%             err_ant = 1000;
%         end    
%         err_rel_porcentual = ( abs(err_ant - err(iter)) / err_ant )*100;
%         if( err_rel_porcentual < err_rel_porcentual_min )
%             fprintf('La red esta en un Minimo Local\n');
%             fprintf('\t ==> Epoch: %d with error: %6.4f\n', iter, err(iter));            
%             plot(err(1:iter));
%             %return;
%             break;
%         end
        
        % Si la red ya no avanza apreciablemente a una mejor solucion,
        % pruebo otra configuracion       
        % if abs(prev_error - err(iter)) < error_tolerance            
            
        % Si el error es menor a la cota de error que se busca.    
%         if abs(err(iter)) < error_tolerance   
%             solutionFound = true;
%             % Se salva info de la mejor solucion
%             if err(iter) < min_error
%                 min_error = err(iter);
%                 selected_epoch = iter;
%                 selected_out_val = out_val;
%                 selected_error = error;
%                 selected_err = err;
%             end
%             break;
%         else
%             prev_error = err(iter);
%         end

    end



% if selected_epoch == -1
%     fprintf('La red no pudo aprender en %d epocas ..  err(iter)=%f\n', epochs, err(iter));
%     plot(err);    
% else
%     % Muestro los resultados. Error y valores calculados y esperados
%     fprintf('selected_epoch: %d with min_error: %6.4f\n\n',selected_epoch,min_error);        
%     %fprintf('|| esperado || calculado ||');        
%     %values = [ train_outputs selected_out_val]
%     %results = values(1:num_patterns-added_patterns,:)
%           
%     % Grafico el error cuadratico medio vs las epocas
%     plot(selected_err(1:selected_epoch))
%     xlabel('Epocas')
%     
%     ylabel('Error')
% 
%     % Salvo la imagen en formato eps
%     print('Error.eps','-deps')
% end


%end % end del while