#Métodos para Recomendação de Hiperparâmetros de Aprendizado de Máquina na 
#                 Classificação de Imagens da Construção Civil
#AutoHyperTunigSK: versão AutoML do HyperTuningSK para opt e lr
#Experimentos para treinamento - Estudo de Caso 4: Classificação de Rachaduras
#Autor: André Ottoni (UFRB e PPGEE/UFBA)
#Orientadora: Prof. Marcela Novo (UFBA)
#04/06/2022

#Limpa o ambiente
rm(list = ls())

#Biblioteca do Keras (versão 2.3.0.0)
library(keras)

#Scott-Knott Clustering Algorithm (versão 1.2-7)
library(ScottKnott) 

#------------1) Declaração da base de dados (treinamento, validação e teste)---------------

base_dir <- "C:/Dados/Pesquisa/Doutorado/DeepLearning/Rachaduras/database"
train_dir <- file.path(base_dir, "treino")
validation_dir <- file.path(base_dir, "validacao")
test_dir <- file.path(base_dir, "teste")
#Treino
train_1_dir <- file.path(train_dir, "1positivo")
train_0_dir <- file.path(train_dir, "0negativo")
#Validação
validation_1_dir <- file.path(validation_dir, "1positivo")
validation_0_dir <- file.path(validation_dir, "0negativo")
#Teste
test_1_dir <- file.path(test_dir, "1positivo")
test_0_dir <- file.path(test_dir, "0negativo")

#Verificar o número de imagens em cada pasta
#Treinamento
cat("Treino -  total de imagens com rachaduras (positivo):", length(list.files(train_1_dir)), "\n")
cat("Treino -  total de imagens sem rachaduras (negativo):", length(list.files(train_0_dir)), "\n")
#Validação
cat("Validação - total de imagens com rachaduras (positivo):", length(list.files(validation_1_dir)), "\n")
cat("Validação - total de imagens sem rachaduras (negativo):", length(list.files(validation_0_dir)), "\n")
#Teste
cat("Teste - total de imagens com rachaduras (positivo):", length(list.files(test_1_dir)), "\n")
cat("Teste - total de imagens sem rachaduras (negativo):", length(list.files(test_0_dir)), "\n")

#---------------------------------------
DLrun<-function(TA,opt,ep,train_dir,validation_dir){
    i<-3 #Arquitetura Mobilenet
  
    #---------3.1) Seleção da Arquitetura Deep Learning
    conv_base <- application_mobilenet(
       weights = NULL,
       include_top = FALSE,
       input_shape = c(50, 50, 3)
    )
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
      #Compilar o modelo
    if (opt == 1){
          model  %>% compile(loss = "binary_crossentropy",
                         optimizer = optimizer_adagrad(lr = TA),
                         #Definição do otimizador
                         metrics = c("acc"))
    }else{
          model  %>% compile(loss = "binary_crossentropy",
                         optimizer = optimizer_adadelta(lr = TA),
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
    
    #Validação
    val_datagen <- image_data_generator(rescale = 1/255)
    validation_generator <- flow_images_from_directory(
      validation_dir,
      val_datagen,
      target_size = c(50, 50),
      batch_size = 40,
      class_mode = "binary"
    )
    #-----------3.6) Configurações para salvar os resultados      
    id <-paste("Res-",
               "Arch",
               i,
               "Opt", 
               opt,
               "lr", 
               TA,
               "run",
               ep,
               sep = '-')
    nomecsv <-paste("C:/Dados/logs/",id,".csv")
    callbacks_list <-list(callback_csv_logger(nomecsv,separator = ",",append = FALSE))
    
    #---------3.7) Simulação do modelo de rede neural - Treino e Validação
    history <- model  %>% fit_generator(
      train_generator,
      steps_per_epoch = 100,
      epochs = 10, 
      validation_data = validation_generator,
      validation_steps = 10,
     # callbacks = callbacks_list
    )
   
    return(history$metrics$val_acc)
}

Hypertuningsk<-function(TA,dadosTA){
  
  #---------------------Modelo de Anova
  met<-factor(dadosTA$X1)    #Xe (parâmetro) é o fator independente 
  ymed <- dadosTA$X2           #y (distância) é o fator dependente
  
  modelo<-aov(ymed~factor(met)) #Modelo fatorial
  print("Informações do modelo Anova")
  summary(modelo)#Informações do Modelo
  paov <- anova(modelo) #p-valor do modelo anova
  
  #Teste de Normalidade KS
  pks<-ks.test(resid(modelo),'pnorm',mean(resid(modelo)),sd(resid(modelo))) 
  
  #Teste de Homogeneidade BT
  pbt<-bartlett.test(ymed~factor(met))
  
  
  Ha <- 0 #Hipótese alternativa 
  
  #Verifica a suposição de normalidade e a significância do modelo
  
  if ((pks$p.value<0.05) || (pbt$p.value<0.05)){ 
    print("Resíduos não-normais ou variâncias não-homogêneas")    
  }else{
    print("Resíduos Normais e variâncias homogêneas") 
    if (paov$`Pr(>F)`[1]<0.05){
      print("Hipótese Alternativa Aceita (Ha): Hiperparâmetros apresentam diferença")
      Ha <- 1 
    }else{
      print("Hipótese inicial Aceita (Ho): Hiperparâmetros não apresentam diferença")
    }
  }
  
  #---Recomendação de parâmetros - Scott-Knott Clustering Algorithm
  
  if (Ha==1){ #Se Ha foi aceita, então realiza agrupamento de Scott-Knott 
    print("----Hyperparameter Tuning of CNN---------")
    print("Medidas de Adequação: Resíduos Normais e variâncias homogêneas") 
    print("Normalidade (pks):")
    print(pks$p.value)
    print("Homogeneidade (pbt):")
    print(pbt$p.value)
    print("Hipótese Alternativa Aceita (Ha): Hiperparâmetros apresentam diferença")
    print("P-valor:")
    print(paov$`Pr(>F)`[1])
    sk1 <- SK(modelo) #Resultado do SK
    print("Ranking de Recomendação - Scott-Knott Clustering:")
    sk1s <- summary(sk1) #Ranking SK
    grupo <- sk1$groups[1] #Grupo que alcançou melhores soluções
    grupo_h <- which(sk1$groups==grupo) #Busca elementos do melhor grupo
    tam_grupo <- length(grupo_h) #Verifica o tamanho do grupo
    ind <- sk1$ord[1:tam_grupo] #Índices de elementos
    hsk <- TA[ind] #Parâmetros recomendados
    print("Hiperparâmetros recomendados:")
    print(hsk)
  }else{
    #Parâmetros não apresentam diferença: então gera parâmetro aleatório
    print("Hiperparâmetros não apresentam diferença ou medidas de adequação não satisfeitas!")
    #Parâmetros não apresentam diferença: então gera parâmetro aleatório
    hsk<-sample(TA) #ordena o vetor de elementos de forma aleatória
    hsk<-hsk[1]     #seleciona o primeiro elemento
  }

  return(hsk)
  
}
#------Estágio 1: Ajuste de otimizadotes---------------
run <- 5 #número de repetições  
TA_fix<-0.010 #taxa de aprendizado inicial
opt<-c(1,2) #otimizador: 1 - adagrad; 2 - sgd
tam_opt <- length(opt)
dacc <- matrix(rep(0,tam_opt*run*2),nrow=tam_opt*run, ncol = 2)
nlinhas <-0

for (k in 1:tam_opt){ #Varia até o número máximo de otimizadores e
  for (z in 1:run){ #Varia o número de épocas
    
    #Acurácia (validação) na repetição
    acuracia<-DLrun(TA_fix,opt[k],z,train_dir,validation_dir)
    
    #Resultados
    print(opt[k])         #Otimizado
    print(mean(acuracia)) #Média de acurácia na validação
    
    #Armazena os resultados das épocas
    nlinhas <- nlinhas+1
    dacc[nlinhas,1] = opt[k]
    dacc[nlinhas,2] = mean(acuracia)
  }
}

dadosopt<-data.frame(dacc) #Transforma os dados em data frame
hskopt<-Hypertuningsk(opt,dadosopt) #executa HypertuningSK

#-----------------------------------------------------------------

#-----------Estágio 2: Ajuste de taxa de aprendizado (TA) ------------
TA<-c(0.001,0.005,0.010,0.015,0.020,0.025,0.030) #TA referência
tam_TA <-length(TA)
hrand1<-rep(0,5)

#Gera valores aleatórios para taxa de aprendizado em intervalos
for (k in 1:(tam_TA-1)){ 
  hrand1[k] <- runif(1,TA[k]-0.001,TA[k+1]+0.001)
}

hrand1<-sort(hrand1) #ordenar os valores gerados
tam_hrand1<-length(hrand1)

dacc1 <- matrix(rep(0,tam_hrand1*run*2),nrow=tam_hrand1*run, ncol = 2)
nlinhas1 <-0

for (k in 1:tam_hrand1){ #Varia até o número máximo de parâmetros e
  for (z in 1:run){ #Varia o número de épocas
    
    #Acurácia (validação) na repetição
    acuracia1<-DLrun(hrand1[k],hskopt[1],z,train_dir,validation_dir)
   
    #Resultados
    print(hrand1[k]) 
    print(mean(acuracia1)) #Média de acurácia na validação

    #Armazena os resultados das épocas
    nlinhas1 <- nlinhas1+1
    dacc1[nlinhas1,1] = hrand1[k]
    dacc1[nlinhas1,2] = mean(acuracia1)
  }
}

dadosTA1<-data.frame(dacc1) #Transforma os dados em data frame
hsk1<-Hypertuningsk(hrand1,dadosTA1)

#----------------------------------------------------------------

#-----------Estágio 3: Ajuste de taxa de aprendizado em busca local----------
#Geração de hiperparâmetros aleatórios na região dos hiperparâmetros recomendados
hrand2 <- runif(3,min(hsk1)-0.001,max(hsk1)+0.001)
hrand2<-sort(hrand2) #ordenar os valores gerados
tam_hrand2<-length(hrand2)

dacc2 <- matrix(rep(0,tam_hrand2*run*2),nrow=tam_hrand2*run, ncol = 2)
nlinhas2 <-0

for (k in 1:tam_hrand2){ #Varia até o número máximo de parâmetros e
  for (z in 1:run){ #Varia o número de épocas
    
    #Acurácia (validação) na repetição
    acuracia2<-DLrun(hrand2[k],hskopt[1],z,train_dir,validation_dir)
    
    #Resultados
    print(hrand2[k]) 
    print(mean(acuracia2)) #Média de acurácia na validação
    
    #Armazena os resultados das repetições
    nlinhas2 <- nlinhas2+1
    dacc2[nlinhas2,1] = hrand2[k]
    dacc2[nlinhas2,2] = mean(acuracia2)
  }
}

dadoshrand2<-data.frame(dacc2) #Transforma os dados em data frame
hsk2<-Hypertuningsk(hrand2,dadoshrand2) #Função para o HyperTuningSK

#----------------------------------------------------------------

#-----------Resultados-----------------

print("Hiperparâmetros Recomendados na Etapa 1 (1 - adagrad; 2 - adadelta:")
print(hskopt)

print("Hiperparâmetros Recomendados na Etapa 2 (taxa de aprendizado):")
print(hsk1)

print("Hiperparâmetros Recomendados na Etapa 3 (taxa de aprendizado):")
print(hsk2)
