#M�todos para Recomenda��o de Hiperpar�metros de Aprendizado de M�quina na 
#                 Classifica��o de Imagens da Constru��o Civil
#AutoHyperTunigSK: vers�o AutoML do HyperTuningSK para opt e lr
#Experimentos para teste - Estudo de Caso 4: Classifica��o de Rachaduras
#Autor: Andr� Ottoni (UFRB e PPGEE/UFBA)
#Orientadora: Prof. Marcela Novo (UFBA)
#04/06/2022

#Limpa o ambiente
rm(list = ls())

#Biblioteca do Keras (vers�o 2.3.0.0)
library(keras)

#Scott-Knott Clustering Algorithm (vers�o 1.2-7)
library(ScottKnott) 

#------------1) Declara��o da base de dados (treinamento, valida��o e teste)---------------

base_dir <- "C:/Dados/Pesquisa/Doutorado/DeepLearning/Rachaduras/database"
train_dir <- file.path(base_dir, "treino")
validation_dir <- file.path(base_dir, "validacao")
test_dir <- file.path(base_dir, "teste")
#Treino
train_1_dir <- file.path(train_dir, "1positivo")
train_0_dir <- file.path(train_dir, "0negativo")
#Valida��o
validation_1_dir <- file.path(validation_dir, "1positivo")
validation_0_dir <- file.path(validation_dir, "0negativo")
#Teste
test_1_dir <- file.path(test_dir, "1positivo")
test_0_dir <- file.path(test_dir, "0negativo")

#Verificar o n�mero de imagens em cada pasta
#Treinamento
cat("Treino -  total de imagens com rachaduras (positivo):", length(list.files(train_1_dir)), "\n")
cat("Treino -  total de imagens sem rachaduras (negativo):", length(list.files(train_0_dir)), "\n")
#Valida��o
cat("Valida��o - total de imagens com rachaduras (positivo):", length(list.files(validation_1_dir)), "\n")
cat("Valida��o - total de imagens sem rachaduras (negativo):", length(list.files(validation_0_dir)), "\n")
#Teste
cat("Teste - total de imagens com rachaduras (positivo):", length(list.files(test_1_dir)), "\n")
cat("Teste - total de imagens sem rachaduras (negativo):", length(list.files(test_0_dir)), "\n")

#---------------------------------------
DLrun<-function(TA,ep,train_dir,validation_dir){
    i<-3 #Arquitetura Mobilenet
    j<-1 #Otimizador adagrad
  
    #---------3.1) Sele��o da Arquitetura Deep Learning
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
    if (opt == 1){
      model  %>% compile(loss = "binary_crossentropy",
                         optimizer = optimizer_adagrad(lr = TA),
                         #Defini��o do otimizador
                         metrics = c("acc"))
    }else{
      model  %>% compile(loss = "binary_crossentropy",
                         optimizer = optimizer_adadelta(lr = TA),
                         #Defini��o do otimizador
                         metrics = c("acc"))
    }
      
    #---------3.4) Especifica��o de Data Augmentation 
    train_datagen <- image_data_generator(
      rescale = 1/255,
      rotation_range = 40,
      width_shift_range = 0.2,
      height_shift_range = 0.2,
      shear_range = 0.2,
      zoom_range = 0.2,
      horizontal_flip = TRUE
    )
    
    #----------3.5) Especifica��es para Treinamento e Valida��o
    train_generator <- flow_images_from_directory(
      train_dir,
      train_datagen,
      target_size = c(50, 50),
      batch_size = 32,
      class_mode = "binary"
    )
    
    #Valida��o
    val_datagen <- image_data_generator(rescale = 1/255)
    validation_generator <- flow_images_from_directory(
      validation_dir,
      val_datagen,
      target_size = c(50, 50),
      batch_size = 40,
      class_mode = "binary"
    )
    #-----------3.6) Configura��es para salvar os resultados      
    id <-paste("Res-",
               "Arch",
               i,
               "Opt", 
               j,
               "lr", 
               TA,
               "run",
               ep,
               sep = '-')
    nomecsv <-paste("C:/Dados/logs/",id,".csv")
    callbacks_list <-list(callback_csv_logger(nomecsv,separator = ",",append = FALSE))
    
    #---------3.7) Simula��o do modelo de rede neural - Treino e Valida��o
    history <- model  %>% fit_generator(
      train_generator,
      steps_per_epoch = 100,
      epochs = 20, 
      validation_data = validation_generator,
      validation_steps = 10,
     # callbacks = callbacks_list
    )
   
    #return(history$metrics$val_acc)
    
    test_datagen <- image_data_generator(rescale = 1/255)
    
    test_generator <- flow_images_from_directory(
      test_dir,
      test_datagen,
      target_size = c(50, 50),
      batch_size = 40,
      class_mode = "binary"
    )
    
    #Resultado de acur�ria para os dados de teste
    restest<-model%>% evaluate_generator(test_generator, steps = 100)
    
    return(restest$acc)
}

Hypertuningsk<-function(TA,dadosTA){
  
  #---------------------Modelo de Anova
  met<-factor(dadosTA$X1)    #Xe (par�metro) � o fator independente 
  ymed <- dadosTA$X2           #y (dist�ncia) � o fator dependente
  
  modelo<-aov(ymed~factor(met)) #Modelo fatorial
  print("Informa��es do modelo Anova")
  summary(modelo)#Informa��es do Modelo
  paov <- anova(modelo) #p-valor do modelo anova
  
  #Teste de Normalidade KS
  pks<-ks.test(resid(modelo),'pnorm',mean(resid(modelo)),sd(resid(modelo))) 
  
  #Teste de Homogeneidade BT
  pbt<-bartlett.test(ymed~factor(met))
  
  
  Ha <- 0 #Hip�tese alternativa 
  
  #Verifica a suposi��o de normalidade e a signific�ncia do modelo
  
  if ((pks$p.value<0.05) || (pbt$p.value<0.05)){ 
    print("Res�duos n�o-normais ou vari�ncias n�o-homog�neas")    
  }else{
    print("Res�duos Normais e vari�ncias homog�neas") 
    if (paov$`Pr(>F)`[1]<0.05){
      print("Hip�tese Alternativa Aceita (Ha): Hiperpar�metros apresentam diferen�a")
      Ha <- 1 
    }else{
      print("Hip�tese inicial Aceita (Ho): Hiperpar�metros n�o apresentam diferen�a")
    }
  }
  
  #---Recomenda��o de par�metros - Scott-Knott Clustering Algorithm
  
  if (Ha==1){ #Se Ha foi aceita, ent�o realiza agrupamento de Scott-Knott 
    print("----Hyperparameter Tuning of CNN---------")
    print("Medidas de Adequa��o: Res�duos Normais e vari�ncias homog�neas") 
    print("Normalidade (pks):")
    print(pks$p.value)
    print("Homogeneidade (pbt):")
    print(pbt$p.value)
    print("Hip�tese Alternativa Aceita (Ha): Hiperpar�metros apresentam diferen�a")
    print("P-valor:")
    print(paov$`Pr(>F)`[1])
    sk1 <- SK(modelo) #Resultado do SK
    print("Ranking de Recomenda��o - Scott-Knott Clustering:")
    sk1s <- summary(sk1) #Ranking SK
    grupo <- sk1$groups[1] #Grupo que alcan�ou melhores solu��es
    grupo_h <- which(sk1$groups==grupo) #Busca elementos do melhor grupo
    tam_grupo <- length(grupo_h) #Verifica o tamanho do grupo
    ind <- sk1$ord[1:tam_grupo] #�ndices de elementos
    hsk <- TA[ind] #Par�metros recomendados
    print("Hiperpar�metros recomendados:")
    print(hsk)
  }else{
    #Par�metros n�o apresentam diferen�a: ent�o gera par�metro aleat�rio
    print("Hiperpar�metros n�o apresentam diferen�a ou medidas de adequa��o n�o satisfeitas!")
    #Par�metros n�o apresentam diferen�a: ent�o gera par�metro aleat�rio
    hsk<-sample(TA) #ordena o vetor de elementos de forma aleat�ria
    hsk<-hsk[1]     #seleciona o primeiro elemento
  }
  return(hsk)
  
}

#-----------Teste dos Hiperpar�metros-----------------------

run <- 5 #n�mero de repeti��es  
TA<-c(0.00366930813761428,0.0054259278124664,0.02219922) #TA para teste
TA<-sort(TA) 
tam_TA <-length(TA)
dacc <- matrix(rep(0,tam_TA*run*2),nrow=tam_TA*run, ncol = 2)
nlinhas <-0

for (k in 1:tam_TA){ #Varia at� o n�mero m�ximo de par�metros e
  for (z in 1:run){ #Varia o n�mero de �pocas
    
    #Acur�cia (valida��o) na repeti��o
    acuracia<-DLrun(TA[k],z,train_dir,validation_dir)
   
    #Resultados
    print(TA[k]) 
    print(mean(acuracia)) #M�dia de acur�cia na valida��o

    #Armazena os resultados das �pocas
    nlinhas <- nlinhas+1
    dacc[nlinhas,1] = TA[k]
    dacc[nlinhas,2] = mean(acuracia)
  }
}

dadosTA<-data.frame(dacc) #Transforma os dados em data frame
hsk<-Hypertuningsk(TA,dadosTA)

#---------------------Resultados---------------------------- 

print("Hiperpar�metros Recomendados no teste:")
print(hsk)

