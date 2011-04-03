% El tamaño de la entrada a la red esta dada por num_inputs CON EL BIAS
% V es el vector que me da los valores binarios, V(1)=0 / V(1)=-1 , V(2)=1
% V(1) es FALSE, V(2) es TRUE
% Detecta paridad impar, poniendo un 1 a la salida
function patterns = generatePatterns(min, max, cant)

global samplingType;
global edgesRatio;


if ( strcmp(samplingType, 'uniform')  )
    %step = (max-min)/cant;
    %x = min:step:max;
    x = linspace(min, max, cant*2 );
elseif ( strcmp(samplingType, 'moreOnEdges')  )
%     x1 = linspace(min, -4, cant*200);
%     x2 = linspace(-4,4,cant*50);
%     x3 = linspace(4,max,cant*200);    
    x1 = linspace(min, -4, floor( (edgesRatio/(1+2*edgesRatio)) * cant) );
    x2 = linspace(-4, 4, ceil( (1/(1+2*edgesRatio)) * cant) );
    x3 = linspace(4, max, floor( (edgesRatio/(1+2*edgesRatio)) * cant) );
    x = [x1 x2 x3];
end

y = 5.*sin(x).^2+cos(x) ;
%y = sin(x);

patterns.num_patterns = size(x);
patterns.num_patterns = patterns.num_patterns(2);
i=1;
j=1;
train_inputs = zeros(1,patterns.num_patterns/2);
train_outputs = zeros(1,patterns.num_patterns/2);
generalization_inputs = zeros(1,patterns.num_patterns/2);
generalization_outputs = zeros(1,patterns.num_patterns/2);
for k=1:patterns.num_patterns
    if( mod(k,2) == 0 )
        train_inputs(i) = x(k);
        train_outputs(i) = y(k); 
        i = i+1;
    else
        generalization_inputs(j) = x(k);
        generalization_outputs(j) = y(k); 
        j = j+1;
    end    
end

patterns.num_patterns = patterns.num_patterns/2;


patterns.train_inputs = [ train_inputs' -ones(patterns.num_patterns, 1)];
patterns.train_outputs = train_outputs';

patterns.generalization_inputs = [ generalization_inputs' -ones(patterns.num_patterns, 1)];
patterns.generalization_outputs = generalization_outputs';



