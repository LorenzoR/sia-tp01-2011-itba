
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

[train_inputs, train_outputs, num_patterns] = generatePatterns(-10,10, num_samples);            
            
% Agrego un conj patrones adicionales adecuados para guiar el aprendizaje de la red
%if (modifyPatterns)
%    [train_inputs, train_outputs, num_patterns] = modifyPatternsAND(train_inputs, train_outputs, num_patterns);
%end

% Para salvar info de la mejor solucion
min_error = 1000;
selected_num_hidden = -1;
selected_epoch = -1;
selected_out_val = -1;
selected_error = -1;
selected_err = -1;
solutionFound = false;

intentos = 0;
while( solutionFound == false )
    fprintf('Intento %d\n', intentos);
    intentos = intentos +1;
    % Inicializo los pesos de la red    
    weight_input_hidden = (rand(num_inputs,num_hidden-1) - 0.5).* weights_scale;
    weight_hidden_output = (rand(num_hidden,num_outputs) - 0.5).* weights_scale;
    
    %--- Se entrena la red con backtracking ---------
    % Se inicializa con un error grande para que no termine inmediatamente
    prev_error = 1000;
    err_iter = 0;    
    err = zeros(1, epochs);    
    for iter = 1 : epochs
        if( printEpochsMessages && mod(iter,messagePerEpochs) == 0 )
            fprintf('Epoque %d\n', iter);
        end
        % Forzamos que se cicle por todos los patrones en forma aleatoria        
        pat_num_array = randperm(num_patterns);
        for j = 1 : num_patterns
            % Selecciono un patron al azar
            pat_num = pat_num_array(j);
            
            % El patron seleccionado es el patron actual en el ciclo
            selected_pat = train_inputs(pat_num,:);
            expected_out = train_outputs(pat_num,:);
            
            % Calculo los valores de salida de la red            
            % Calculo el Pot de membrana h, y aplicamos la Fn de activacion
            h = selected_pat*weight_input_hidden;
            hidden_val = [feval(activationFunction, h, beta) -1];
            
            out_h = hidden_val*weight_hidden_output;            
            % out_val = feval(outputActivationFunction, out_h, beta); %AQUI
            out_val = out_h; %porque la salida tiene fn de activ lineal
            
            % Calculo las variaciones o deltas de error para la capa de salida y la oculta
            %out_delta = (expected_out - out_val) * feval(outputActivationFunctionDeriv, out_h, beta); %AQUI
            out_delta = expected_out - out_val ;%porque la salida tiene fn de activ lineal
            
            %hidden_delta = out_delta*weight_hidden_output(1:num_hidden-1,:)' .* feval(activationFunctionDeriv, hidden_val(1,1:num_hidden-1), beta);
            hidden_delta = out_delta*weight_hidden_output' .* feval(activationFunctionDeriv, hidden_val, beta);
            
            % Hago backpropagation para los pesos de la capa de salida (oculta-salida)
            weightChangeHO = LR*hidden_val' * out_delta;
            weight_hidden_output = weight_hidden_output + weightChangeHO;

            % Hago backpropagation para los pesos de la capa oculta (entrada-oculta)
            weightChangeIH = LR*selected_pat'*hidden_delta(:,1:num_hidden-1);
            weight_input_hidden = weight_input_hidden + weightChangeIH;

        end
        
        % -- Termino otra epoca
        
	% Obtengo la salida para todos los patrones, luego de entrenar la red
        h = train_inputs*weight_input_hidden;
        hidden_val = [feval(activationFunction, h, beta) -ones(num_patterns,1)];

        out_h = hidden_val*weight_hidden_output;            
        out_val = out_h;            
        %out_val = feval(outputActivationFunction, out_h, beta);
        
        % obtengo los errores para cada uno de los patrones
        error = train_outputs - out_val;
   
            
        %Si hay patrones adicionales repetidos, non los considero al
        %calcular el error
        %if( modifyPatterns )
        %    error = error(1:(2^(num_inputs-1)));
        %end

        % Calculo el error cuadratico medio
        error = (sum(error.^2)')/2;	

        % finalmente los promedio para poder compararlo contra error_tolerance
        err(iter) = sum(error)/num_patterns;
       
        % Si el error relativo porcentual del paso n al n+1 es muy pequeño,
        % entonces el error no disminuye apreciablemente. Seguramente estoy en un minimo local
        if iter > 1
            err_ant = err(iter-1);
        else
            err_ant = 1000;
        end    
        err_rel_porcentual = ( abs(err_ant - err(iter)) / err_ant )*100;
        if( err_rel_porcentual < err_rel_porcentual_min )
            fprintf('La red esta en un Minimo Local\n');
            fprintf('\t ==> Epoch: %d with error: %6.4f\n', iter, err(iter));            
            plot(err(1:iter));
            %return;
            break;
        end
        
        % Si la red ya no avanza apreciablemente a una mejor solucion,
        % pruebo otra configuracion       
        % if abs(prev_error - err(iter)) < error_tolerance            
            
        % Si el error es menor a la cota de error que se busca.    
        if abs(err(iter)) < error_tolerance   
            solutionFound = true;
            % Se salva info de la mejor solucion
            if err(iter) < min_error
                min_error = err(iter);
                selected_epoch = iter;
                selected_out_val = out_val;
                selected_error = error;
                selected_err = err;
            end
            break;
        else
            prev_error = err(iter);
        end

    end



if selected_epoch == -1
    fprintf('La red no pudo aprender en %d epocas ..  err(iter)=%f\n', epochs, err(iter));
    plot(err);    
else
    % Muestro los resultados. Error y valores calculados y esperados
    fprintf('selected_epoch: %d with min_error: %6.4f\n\n',selected_epoch,min_error);        
    %values = [ (abs(sum((train_inputs' - 1)./2))'-1) train_outputs selected_out_val];
    %fprintf('|| esperado || calculado ||');        
    %values = [ train_outputs selected_out_val]
    %results = values(1:num_patterns-added_patterns,:)
          
    % Grafico el error cuadratico medio vs las epocas
    plot(selected_err(1:selected_epoch))
    xlabel('Epocas')
    
    ylabel('Error')

    % Salvo la imagen en formato eps
    print('Error.eps','-deps')
end


end % end del while