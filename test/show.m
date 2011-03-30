function show(train_inputs, train_outputs, out_val)

plot(train_inputs(:,1), out_val, 'r')
hold

plot(train_inputs(:,1), train_outputs)