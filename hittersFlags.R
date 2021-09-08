FLAGS  = flags(flag_numeric("nodes", 4),
               flag_numeric("batch_size", 100),
               flag_string("activation", "relu"),
               flag_numeric("learning_rate", 0.01),
               flag_numeric("epochs", 30)
) 

model = keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation, 
              input_shape = dim(hitters_trainScaled)[2]) %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_sgd(lr=FLAGS$learning_rate), 
  loss = 'mse')

model %>% fit(hitters_trainScaled, hitters_trainScaledlabel, 
              epochs = FLAGS$epochs, 
              batch_size= FLAGS$batch_size, 
              validation_data = list(hitters_valScaled, hitters_valScaledlabel))
#define a flag for each hyper parameter you want to fine tune and a default value for that parameter