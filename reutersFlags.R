FLAGS  = flags(flag_numeric("nodes", 4),
               flag_numeric("batch_size", 100),
               flag_string("activation", "relu"),
               flag_numeric("learning_rate", 0.01),
               flag_numeric("epochs", 30)
) 

model = keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation, input_shape = ncol(reuters_Newtrain$x)) %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation) %>%
  layer_dense(units = 46, activation = 'softmax')

model %>% compile(
  optimizer = optimizer_adam(lr=FLAGS$learning_rate), 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy'))

model %>% fit(reuters_Newtrain$x, reuters_Newtrain$y, 
              epochs = FLAGS$epochs, 
              batch_size= FLAGS$batch_size, 
              validation_data=list(reuters_val$x, reuters_val$y))
#define a flag for each hyper parameter you want to fine tune and a default value for that parameter