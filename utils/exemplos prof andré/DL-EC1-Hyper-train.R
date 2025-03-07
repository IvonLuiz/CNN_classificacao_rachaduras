#C�digos do Doutorado - 2021
#M�todos para Recomenda��o de Hiperpar�metros de Aprendizado de M�quina na 
#                 Classifica��o de Imagens da Constru��o Civil
#Estudo de Caso 1: Reconhecimento de Vegeta��o em Edifica��es 
#Objetivo: Experimentos de treinamento para an�lise de taxa de aprendizado e otimizador
#Autor: Andr� Ottoni (UFRB e PPGEE/UFBA)
#Orientadora: Prof�. Marcela Novo (UFBA)
#03/12/2021


#Limpa o ambiente
rm(list = ls())

#Biblioteca do Keras (vers�o 2.3.0.0)
library(keras)

#------------1) Declara��o da base de dados (treinamento, valida��o e teste)---------------

#Diret�rios da imagens
base_dir <- "C:/Dados/Pesquisa/Doutorado/DeepLearning/modeloscivil"
train_dir <- file.path(base_dir, "treino")
validation_dir <- file.path(base_dir, "validacao")
test_dir <- file.path(base_dir, "testereview2")
#Treino
train_v_dir <- file.path(train_dir, "vtrain")
train_p_dir <- file.path(train_dir, "ptrain")
#Valida��o
validation_v_dir <- file.path(validation_dir, "vval")
validation_p_dir <- file.path(validation_dir, "pval")
#Teste
test_v_dir <- file.path(test_dir, "vteste2")
test_p_dir <- file.path(test_dir, "pteste2")

#Verificar o n�mero de imagens em cada pasta
#Treinamento
cat("Treino -  total de imagens com vegeta��o:", length(list.files(train_v_dir)), "\n")
cat("Treino -  total de imagens com parede limpa:", length(list.files(train_p_dir)), "\n")
#Valida��o
cat("Valida��o - total de imagens com vegeta��o:", length(list.files(validation_v_dir)), "\n")
cat("Valida��o - total de imagens com parede limpa:", length(list.files(validation_p_dir)), "\n")
#Teste
cat("Teste - total de imagens com vegeta��o:", length(list.files(test_v_dir)), "\n")
cat("Teste - total de imagens com parede limpa:", length(list.files(test_p_dir)), "\n")

#--------2) Especifa��es dos Experimentos---------------

#Inicializa��o de vari�veis
comb<-1
ep<-1
cont<-1

#I<-c(1,1,1,2,2,2,3,3,3)
#J<-c(1,2,3,1,2,3,1,2,3)

i<-2#Arquitetura
j<-2 #Otimizador
k<-1 #taxa de aprendizado
ep<-5
TA<-c(0.001,0.005,0.010,0.015,0.020,0.025)

#for (comb in 3:9){
  #i<-I[comb]
  #j<-J[comb]

#-------------3) Treinamento ----------------------
for (j in 1:3){ #J -> Otimizador
  for (k in 4:6){ # k-> taxa de aprendizado
    for (ep in 1:5) { # n�mero de repeti��es para cada combina��o
      
      #---------3.1) Sele��o da Arquitetura Deep Learning
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
      
      #--------3.2) Adiciona Camadas de classifica��o
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
      
      
      #------3.4) Seleciona o otimizador----------------
      if (j==1){
        model  %>% compile(loss = "binary_crossentropy",
                           optimizer = optimizer_adagrad(lr = TA[k]),
                           #Defini��o do otimizador
                           metrics = c("acc"))
      }
      if (j==2){
        #Compilar o modelo
        model  %>% compile(loss = "binary_crossentropy",
                           optimizer = optimizer_sgd(lr = TA[k]),
                           #Defini��o do otimizador
                           metrics = c("acc"))
      }
      if (j==3){
        #Compilar o modelo
        model  %>% compile(loss = "binary_crossentropy",
                           optimizer = optimizer_adam(lr = TA[k]),
                           #Defini��o do otimizador
                           metrics = c("acc"))
      }
      if (j==4){
        #Compilar o modelo
        model  %>% compile(loss = "binary_crossentropy",
                           optimizer = optimizer_adamax(lr = TA[k]),
                           #Defini��o do otimizador
                           metrics = c("acc"))
      }
      
      
      #---------3.5) Especifica��o de Data Augmentation 
      train_datagen <- image_data_generator(
        rescale = 1/255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = TRUE
      )
      
      #----------3.6) Especifica��es para Treinamento e Valida��o
      train_generator <- flow_images_from_directory(
        train_dir,
        train_datagen,
        target_size = c(50, 50),
        batch_size = 32,
        class_mode = "binary"
      )
      
      val_datagen <- image_data_generator(rescale = 1/255)
      validation_generator <- flow_images_from_directory(
        validation_dir,
        val_datagen,
        target_size = c(50, 50),
        batch_size = 50,
        class_mode = "binary"
      )
      
      #-----------3.7) Configura��es para salvar os resultados
      id <-paste("Res-Arch",
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
      
      #---------3.8) Simula��o do modelo de rede neural - Treino e Valida��o
      history <- model  %>% fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 5,
        validation_data = validation_generator,
        validation_steps = 1,
        callbacks = callbacks_list
      )
    }#Fim do loop de repeti��es
  }#Fim do loop de taxa de aprendizado
}#Fim do loop de otimizador


