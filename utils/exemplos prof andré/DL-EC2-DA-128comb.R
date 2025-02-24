#C�digos do Doutorado - 2021
#M�todos para Recomenda��o de Hiperpar�metros de Aprendizado de M�quina na 
#                 Classifica��o de Imagens da Constru��o Civil
#Estudo de Caso 2: Classifica��o de patologias em calhas de edifica��es 
#Objetivo: Experimentos de Data Augmentation
#Autor: Andr� Ottoni (UFRB e PPGEE/UFBA)
#Orientadora: Prof�. Marcela Novo (UFBA)
#04/03/2022

#Limpa o ambiente
rm(list = ls())

#Biblioteca do Keras (vers�o 2.3.0.0)
library(keras)

#Leitura das configura��es poss�veis: 128 combina��es
dados <-read.delim("combinacoes.txt")
attach(dados)

#------------1) Declara��o da base de dados (treinamento, valida��o e teste)

#Diret�rios da imagens
base_dir <- "C:/Dados/Pesquisa/Doutorado/DeepLearning/telhado3"
train_dir <- file.path(base_dir, "treino")
validation_dir <- file.path(base_dir, "validacao")
test_dir <- file.path(base_dir, "teste")
#Treino
train_1_dir <- file.path(train_dir, "1suja")
train_0_dir <- file.path(train_dir, "0limpa")
#Valida��o
validation_1_dir <- file.path(validation_dir, "1suja")
validation_0_dir <- file.path(validation_dir, "0limpa")
#Teste
test_1_dir <- file.path(test_dir, "1suja")
test_0_dir <- file.path(test_dir, "0limpa")

#Verificar o n�mero de imagens em cada pasta
#Treinamento
cat("Treino -  total de imagens com calhas sujas:", length(list.files(train_1_dir)), "\n")
cat("Treino -  total de imagens com calhas limpas:", length(list.files(train_0_dir)), "\n")
#Valida��o
cat("Valida��o - total de imagens com calhas sujas:", length(list.files(validation_1_dir)), "\n")
cat("Valida��o - total de imagens com calhas limpa:", length(list.files(validation_0_dir)), "\n")
#Teste
cat("Treino - total de imagens com calhas sujas:", length(list.files(test_1_dir)), "\n")
cat("Treino - total de imagens com calhas limpas:", length(list.files(test_0_dir)), "\n")

#--------2) Especifa��es dos Experimentos---------------

#Valores poss�veis para as tranforma��es de Data Augmentation
R <- c(0, 40)
H <- c(FALSE, TRUE)
V <- c(FALSE, TRUE)
He <- c(0, 0.2)
S <- c(0, 0.2)
W <- c(0, 0.2)
Z <- c(0, 0.2)

i<-1 #combina��o
ep<-1 #repeti��o da combina��o

#-------------3) Treinamento ----------------------

for (i in 1:128){ #loop da combina��o
  for (ep in 1:5) { #repeti��o da combina��o
     #------3.1) Arquitetura neural----------------
     conv_base <- application_mobilenet(
       weights = NULL,
       include_top = FALSE,
       input_shape = c(50, 50, 3)
     )
     
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

     #------3.3) Seleciona o otimizador----------------
    
    #Compilar o modelo
    model  %>% compile(loss = "binary_crossentropy",
                       optimizer = optimizer_adagrad(),
                       #Defini��o do otimizador
                       metrics = c("acc"))
    
     #---------3.4) Especifica��o de Data Augmentation 
    train_datagen <- image_data_generator(
      rescale = 1 / 255,
      rotation_range = R[R1[i] + 1],
      horizontal_flip = H[H1[i] + 1],
      vertical_flip = V[V1[i] + 1],
      height_shift_range = He[He1[i] + 1],
      shear_range = S[S1[i] + 1],
      width_shift_range = W[W1[i] + 1],
      zoom_range = Z[Z1[i] + 1]
    )
    
    #----------3.5) Especifica��es para Treinamento e Valida��o
    train_generator <- flow_images_from_directory(
      train_dir,
      train_datagen,
      target_size = c(50, 50),
      batch_size = 32,
      class_mode = "binary"
    )
    
    val_datagen <-image_data_generator(rescale = 1 / 255)
    validation_generator <- flow_images_from_directory(
      validation_dir,
      val_datagen,
      target_size = c(50, 50),
      batch_size = 50,
      class_mode = "binary"
    )
    
    #-----------3.6) Configura��es para salvar os resultados
    
    id <-paste("Res2Comb",
            i,
            "R",
            R[R1[i] + 1],
            "H",
            H[H1[i] + 1],
            "V",
            V[V1[i] + 1],
            "He",
            He[He1[i] + 1],
            "S",
            S[S1[i] + 1],
            "W",
            W[W1[i] + 1],
            "Z",
            Z[Z1[i] + 1],
            "epoch",
            ep,
            sep = '-')
    nomecsv <-paste("C:/Dados/logs/",id,".csv")
    callbacks_list <-list(callback_csv_logger(nomecsv,separator = ",",append = FALSE))
    
    #---------3.7) Simula��o do modelo de rede neural - Treino e Valida��o
    history <- model  %>% fit_generator(
      train_generator,
      steps_per_epoch = 100,
      epochs = 10,
      validation_data = validation_generator,
      validation_steps = 1,
      callbacks = callbacks_list
    )
  }#Fim do loop de repeti��es
}#FIm do loop de combina��es de Data Augmentation
