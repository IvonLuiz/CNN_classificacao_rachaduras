#Códigos do Doutorado - 2021
#Métodos para Recomendação de Hiperparâmetros de Aprendizado de Máquina na 
#                 Classificação de Imagens da Construção Civil
#Estudo de Caso 1: Reconhecimento de Vegetação em Edificações 
#Objetivo: Experimentos de teste para análise de taxa de aprendizado e otimizador
#Autor: André Ottoni (UFRB e PPGEE/UFBA)
#Orientadora: Profª. Marcela Novo (UFBA)
#05/12/2021


#Limpa o ambiente
rm(list = ls())

#Biblioteca do Keras (versão 2.3.0.0)
library(keras)

#------------1) Declaração da base de dados (treinamento, validação e teste)---------------

#Diretórios da imagens
base_dir <- "C:/Dados/Pesquisa/Doutorado/DeepLearning/modeloscivil"
train_dir <- file.path(base_dir, "treino")
validation_dir <- file.path(base_dir, "validacao")
test_dir <- file.path(base_dir, "testereview2")
#Treino
train_v_dir <- file.path(train_dir, "vtrain")
train_p_dir <- file.path(train_dir, "ptrain")
#Validação
validation_v_dir <- file.path(validation_dir, "vval")
validation_p_dir <- file.path(validation_dir, "pval")
#Teste
test_v_dir <- file.path(test_dir, "vteste2")
test_p_dir <- file.path(test_dir, "pteste2")

#Verificar o número de imagens em cada pasta
#Treinamento
cat("Treino -  total de imagens com vegetação:", length(list.files(train_v_dir)), "\n")
cat("Treino -  total de imagens com parede limpa:", length(list.files(train_p_dir)), "\n")
#Validação
cat("Validação - total de imagens com vegetação:", length(list.files(validation_v_dir)), "\n")
cat("Validação - total de imagens com parede limpa:", length(list.files(validation_p_dir)), "\n")
#Teste
cat("Teste - total de imagens com vegetação:", length(list.files(test_v_dir)), "\n")
cat("Teste - total de imagens com parede limpa:", length(list.files(test_p_dir)), "\n")

#--------2) Especifações dos Experimentos---------------

#Inicialização de variáveis
i<-3#Arquitetura
j<-1 #Otimizador
k<-4 #taxa de aprendizado
ep<-3
TA<-c(0.001,0.005,0.010,0.015,0.020,0.025)

#-------------3) Treinamento ----------------------

#---------3.1) Seleção da Arquitetura Deep Learning

if (i==1){
  conv_base<-application_densenet121(
    weights = NULL,
    include_top = FALSE,
    input_shape = c(50, 50, 3)
  )
}

if (i==2){
  conv_base <- application_vgg16(
    weights = NULL,
    include_top = FALSE,
    input_shape = c(50, 50, 3)
  )
}

if (i==3){
  conv_base <- application_mobilenet(
    weights = NULL,
    include_top = FALSE,
    input_shape = c(50, 50, 3)
  )
}

#--------3.2) Adiciona Camadas de classificação
model1 <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(
    units = 512,
    activation = "relu",
    kernel_initializer = "he_normal"
  ) %>%
  layer_dense(units = 1, activation = "sigmoid")
model <- model1

#------3.3) Seleciona o otimizador----------------

if (j==1){
  #Compilar o modelo
  model  %>% compile(loss = "binary_crossentropy",
                     optimizer = optimizer_adagrad(lr = TA[k]),
                     #Definição do otimizador
                     metrics = c("acc"))
}

if (j==2){
  #Compilar o modelo
  model  %>% compile(loss = "binary_crossentropy",
                     optimizer = optimizer_sgd(lr = TA[k]),
                     #Definição do otimizador
                     metrics = c("acc"))
}


if (j==3){
  #Compilar o modelo
  model  %>% compile(loss = "binary_crossentropy",
                     optimizer = optimizer_adam(lr = TA[k]),
                     #Definição do otimizador
                     metrics = c("acc"))
}

if (j==4){
  #Compilar o modelo
  model  %>% compile(loss = "binary_crossentropy",
                     optimizer = optimizer_adamax(lr = TA[k]),
                     #Definição do otimizador
                     metrics = c("acc"))
}

#---------3.4) Especificação de Data Augmentation 
train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

#----------3.5) Especificações para Treinamento e Validação
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(50, 50),
  batch_size = 32,
  class_mode = "binary"
)

#Leitura dos Dados de validação
val_datagen <- image_data_generator(rescale = 1/255)
validation_generator <- flow_images_from_directory(
  validation_dir,
  val_datagen,
  target_size = c(50, 50),
  batch_size = 50,
  class_mode = "binary"
)

#-----------3.6) Configurações para salvar os resultados
id <-paste("Teste-",
           "Arc",
           i,
           "Opt", 
           j,
           "lr", 
           TA[k],
           "epoch",
           ep,
           sep = '-')
nomecsv <-paste("C:/Dados/lr-otim/logs/",id,".csv")
callbacks_list <-list(callback_csv_logger(nomecsv,separator = ",",append = FALSE))

#---------3.7) Simulação do modelo de rede neural - Treino e Validação
history <- model  %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = 1,
  #callbacks = callbacks_list
)

#Validação  com os pesos finais
res<-model%>% evaluate_generator(validation_generator, steps = 1)

#------------4) Teste
test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(50, 50),
  batch_size = 90,
  class_mode = "binary"
)

#Resultado de acurária para os dados de teste
restest<-model%>% evaluate_generator(test_generator, steps = 1)

#Salvar o modelo
model %>% save_model_hdf5("EC1-teste.h5.h5")

