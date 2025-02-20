# Bibliotecas
#library(tensorflow)
#install_tensorflow()

rm(list = ls())

library(tensorflow)
library(keras)
#install_keras(method="virtualenv", envname="myenv", pip_options = "--no-cache-dir")

learning_rate <- 0.010
dirc <- "D:/Code/Data Science/Concrete Crack Images for Classification"
train_dirc <- file.path(dirc, "treino")
train_dir <- file.path(dirc, "treino")
validation_dir <- file.path(dirc, "validacao")
test_dir <- file.path(dirc, "teste")
#Treino
train_1_dir <- file.path(train_dir, "1positivo")
train_0_dir <- file.path(train_dir, "0negativo")
#Valida??o
validation_1_dir <- file.path(validation_dir, "1positivo")
validation_0_dir <- file.path(validation_dir, "0negativo")
#Teste
test_1_dir <- file.path(test_dir, "1positivo")
test_0_dir <- file.path(test_dir, "0negativo")



# Arquitetura
densenet <- application_densenet121(
  include_top = FALSE,
  weights = NULL,
  input_shape = c(50, 50, 3)
)


# Camadas de classificaçao
model1 <- keras_model_sequential() %>%
  densenet %>%
  layer_flatten() %>%
  layer_dense(
    units = 512,
    activation = "relu",
    kernel_initializer = "he_normal"
  ) %>%
  layer_dense(units = 1, activation = "sigmoid")
model <- model1


# Compilar o modelo
model  %>% compile(loss = "binary_crossentropy",
                   optimizer = optimizer_adagrad(learning_rate = learning_rate),
                   metrics = c("acc"))



train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)


train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(50, 50),
  batch_size = 32,
  class_mode = "binary"
)

test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(50, 50),
  batch_size = 40,
  class_mode = "binary"
)


# Validação

val_datagen <- image_data_generator(rescale = 1/255)

validation_generator <- flow_images_from_directory(
  validation_dir,
  val_datagen,
  target_size = c(50, 50),
  batch_size = 40,
  class_mode = "binary"
)

history <- model  %>% fit(
  train_generator,
  steps_per_epoch = 60,
  epochs = 11, 
  validation_data = validation_generator,
  validation_steps = 10,
)



summary(model)

model %>% compile(loss= "binary_crossentropy",
                  optimizer="adagrad",
                  metrics="accuracy")


# Evaluate 
model %>% evaluate(test_generator)

results <- model %>% predict(test_generator) %>% k_argmax()

