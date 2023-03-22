#Limpa o ambiente
rm(list = ls())

# Bibliotecas
library(keras)
library(caret)  # Calcular sensibilidade, recall
library(hash)   # Fazer bibliotecas
library(pROC)   # Analizar curvas ROC AUC 
library(writexl)# Exportar para excel


# Arquitetura
architecture <- function(choice, input_shape = c(50, 50, 3)){
  
  if (choice == 1 | choice == 'vgg16'){
    arc <- application_vgg16(
      include_top = FALSE,
      weights = NULL,
      input_shape = input_shape
    )
  }
  
  else if (choice == 2 | choice == 'resnet50'){
    arc <- application_resnet50(
      include_top = FALSE,
      weights = NULL,
      input_shape = input_shape
    )
  }

  
  else if (choice == 3 | choice == 'densenet121'){
    arc <- application_densenet121(
      include_top = FALSE,
      weights = NULL,
      input_shape = input_shape
    )
  }
  
  else if (choice == 4 | choice == 'efficientnet'){
    arc <- application_efficientnet_b0(
      include_top = FALSE,
      weights = NULL,
      input_shape = input_shape
    )
  }
  
  else if (choice == 5 | choice == 'mobilenet'){
    arc <- application_mobilenet(
      include_top = FALSE,
      weights = NULL,
      input_shape = input_shape
    )
  }
  

  else{
    stop("Favor escolher arquitetura válida.")
  }
  
  return(arc)
  
}


# Função cria o modelo de CNN com arquitetura escolhida e output binário
create_model <- function(choice, input_shape = c(50, 50, 3),
                         learning_rate = 0.010){
  
  arc <- architecture(choice=choice, input_shape=input_shape)
  
  # Camadas de classificaçao
  model <- keras_model_sequential() %>%
    arc %>%
    layer_flatten() %>%
    layer_dense(
      units = 512,
      activation = "relu",
      kernel_initializer = "he_normal"
    ) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  
  # Compilar o modelo
  model  %>% compile(loss = "binary_crossentropy",
                     optimizer = optimizer_adagrad(learning_rate = learning_rate),
                     metrics = c("acc"))
  
  return(model)
}


# Função que lida com todos os generators dos dados de entrada
handling_data <- function(train_dir, test_dir){
  
  # Data augmentation
  train_datagen <- image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE
  )
  
  # Dados de treinamento
  train_generator <- flow_images_from_directory(
    train_dir,
    train_datagen,
    target_size = c(50, 50),
    batch_size = 32,
    class_mode = "binary"
  )
  
  # Dados de teste
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
  
  return_list <- list(train_generator, test_generator, validation_generator)
  
  return(return_list)
}


# Função roda modelo
model_history <- function(model, epochs, train_generator, validation_generator){
  history <- model  %>% fit(
    train_generator,
    steps_per_epoch = 100,
    epochs = epochs, 
    validation_data = validation_generator,
    validation_steps = 10,
  )
}


# Função para guardar resultados (history + evaluate + system.time)
train_model_results <- function(model, epochs=30, repetition=5,
                             train_generator, validation_generator,
                             test_generator){
  results_list <- list()
  
  for (exec in (1:repetition)) {
    cat(sprintf("Rodando a interação: %i\ \n", exec))
    
    # Rodar modelo e guardar o tempo de execução
    st <- system.time(
      history <- model_history(model, epochs=epochs,
                               train_generator, validation_generator)
    )
    
    # Guardar resultados history
    results_list[[length(results_list)+1]] <- history
    
    # Evaluate
    evaluate <- model %>% evaluate(test_generator)
    
    # Guardar resultados evaluate e time
    results_list[[length(results_list)+1]] <- evaluate
    results_list[[length(results_list)+1]] <- st
  }
  
  return(results_list)
}


# Função para retornar matrix de confusão
confusion_matrix <- function(pred, actual){
  
  my_data1 <- data.frame(data = pred, type = "prediction")
  my_data2 <- data.frame(data = actual, type = "real")
  my_data3 <- rbind(my_data1, my_data2)
  
  # Verificar se os níveis(levels) são idênticos
  identical(levels(my_data3[my_data3$type == "prediction",1]) , levels(my_data3[my_data3$type == "real",1]))
  
  cm <- confusionMatrix(my_data3[my_data3$type == "prediction",1], 
                        my_data3[my_data3$type == "real",1],
                        dnn = c("Prediction", "Reference"),  mode = "everything",
                        positive="1")
  
  return(cm)
}


# Função para prever valores teste e retornar resultados
test_model_results <- function(model, test_generator_data){
  
  # Pegar os valores das classes previstas em 
  test_matrix <- as.matrix(test_generator_data[0][[2]])
  
  for (index in 1:(length(test_generator_data)-1)){
    test_matrix <- append(test_matrix, test_generator_data[index][[2]])
  }
  
  test_matrix <- factor(test_matrix)
  
  # Predict
  st <- system.time(
    predictions_prob <- model %>% predict(test_generator_data)
  )
  predicitons <- factor(round(predictions_prob))
  
  
  # accuracy: (tp + tn) / (p + n)
  # precision tp / (tp + fp)
  precision <- posPredValue(predicitons, test_matrix, positive="1")
  # recall: tp / (tp + fn)
  recall <- sensitivity(predicitons, test_matrix, positive="1")
  # f1: 2 tp / (2 tp + fp + fn)
  F1 <- (2 * precision * recall) / (precision + recall)
  
  
  # kappa
  kappa <- kappa(c(predicitons, test_matrix))
  # ROC AUC
  roc <- roc(test_matrix, as.vector(predictions_prob))
  auc <- auc(roc)
  plot(roc ,main ="ROC curve", print.auc = TRUE)
  # confusion matrix
  cm <- confusion_matrix(predicitons, test_matrix)
  
  results_list <- hash()
  #results_list[['Confusion Matrix: ']] <- cm[2]
  
  results_list[['Precision: ']] <- precision
  results_list[['Recall: ']] <- recall
  results_list[['F1: ']] <- F1
  results_list[['Kappa: ']] <- kappa
  results_list[['ROC AUC: ']] <- auc
  
  return <- list(cm, results_list, roc, st)
  
  return(return)
}


# Função para exportar dados de matriz para excel
export_data <- function(data_train, data_test, path, name){
  
  cm <- data_test[[1]] # Matriz de confusão
  auc <- data_test[[3]]$auc # ROC AUC
  time_test <- data_test[[4]][3] # Tempo
  
  # Transformar entradas em data frames
  my_data1 <- data.frame(training=c(acc=data_train[[1]]$metrics$acc[data_train[[1]]$params$epochs],
                                    val_acc=data_train[[1]]$metrics$val_acc[data_train[[1]]$params$epochs],
                                    time=data_train[[3]][3]),
                         validation=c(acc=data_train[[2]][2], '', ''),
                         test=c(cm$overall[1], '', time_test))
  my_data1 <- cbind(" "=rownames(my_data1), my_data1)
  my_data2 <- data.frame(cm$table)
  my_data3 <- data.frame(cbind(t(cm$overall),t(cm$byClass), auc))

  
  tocsv <- list(my_data1, my_data2, my_data3) # Juntar

  # Guardar num arquivo excel
  dir <- paste(path, name, sep = '')
  write_xlsx(tocsv, path = dir)
  
  cat(sprintf("Dados de treino, Matriz confusão e  %s foram salvos.\n", name))
}


#------------------------------------------------------------------------------#

# Diretório de dados
# Os dados foram divididos na proporção 70:20:10
dirc <- "Concrete Crack Images for Classification"
train_dir <- file.path(dirc, "treino")
validation_dir <- file.path(dirc, "validacao")
test_dir <- file.path(dirc, "teste")


# Dados
generators <- handling_data(train_dir, test_dir)

train_generator <- generators[[1]]
test_generator <- generators[[2]]
validation_generator <- generators[[3]]


# Esolher arquitetura a ser avaliada
choice <- 'densenet121'
learning_rate <- 0.010
model <- create_model(choice = choice, learning_rate = learning_rate)
summary(model)

# Path to save results
path_results <- "D:/Code/Data Science/R/Resultados/"

# RODAR 5 VEZES A ANALISE DE DADOS TESTE PARA 10, 20, 30, 40 e 50 EPOCAS, TRIENAR 5 VEZES TBBBB
repetition <- 5


# Rodar modelos e guardar o tempo gasto por cada para diferentes épocas
# O resultado decorrido é dado em segundos
#------------------------------------------------------------------------------#
for (rep in (1:repetition)){
  
  results_10_epochs <- train_model_results(model = model, epochs = 10, repetition = 1,
                                           train_generator = train_generator,
                                           validation_generator = validation_generator,
                                           test_generator = test_generator)
  results_10_epochs
  
  # Avaliar resultados dos dados de teste
  resultados <- test_model_results(model, test_generator)
  resultados
  
  # Exportar resultados
  file_name <- sprintf("10epochs_%s_%i.xlsx", choice, rep)
  export_data(results_10_epochs, resultados, path_results, file_name)
}


#------------------------------------------------------------------------------#
for (rep in (1:repetition)){
  
  results_20_epochs <- train_model_results(model = model, epochs = 20, repetition = 1,
                                           train_generator = train_generator,
                                           validation_generator = validation_generator,
                                           test_generator = test_generator)
  results_20_epochs
  
  # Avaliar resultados dos dados de teste
  resultados <- test_model_results(model, test_generator)
  resultados
  
  # Exportar resultados
  file_name <- sprintf("20epochs_%s_%i.xlsx", choice, rep)
  export_data(results_20_epochs, resultados, path_results, file_name)
}


#------------------------------------------------------------------------------#
for (rep in (1:repetition)){
  
  results_30_epochs <- train_model_results(model = model, epochs = 30, repetition = 1,
                                           train_generator = train_generator,
                                           validation_generator = validation_generator,
                                           test_generator = test_generator)
  results_30_epochs
  
  # Avaliar resultados dos dados de teste
  resultados <- test_model_results(model, test_generator)
  resultados
  
  # Exportar resultados
  file_name <- sprintf("30epochs_%s_%i.xlsx", choice, rep)
  export_data(results_30_epochs, resultados, path_results, file_name)
}


#------------------------------------------------------------------------------#
for (rep in (1:repetition)){
  
  results_40_epochs <- train_model_results(model = model, epochs = 40, repetition = 1,
                                           train_generator = train_generator,
                                           validation_generator = validation_generator,
                                           test_generator = test_generator)
  results_40_epochs
  
  # Avaliar resultados dos dados de teste
  resultados <- test_model_results(model, test_generator)
  resultados
  
  # Exportar resultados
  file_name <- sprintf("40epochs_%s_%i.xlsx", choice, rep)
  export_data(results_40_epochs, resultados, path_results, file_name)
}


#------------------------------------------------------------------------------#
for (rep in (1:repetition)){
  
  results_50_epochs <- train_model_results(model = model, epochs = 50, repetition = 1,
                                           train_generator = train_generator,
                                           validation_generator = validation_generator,
                                           test_generator = test_generator)
  results_50_epochs
  
  # Avaliar resultados dos dados de teste
  resultados <- test_model_results(model, test_generator)
  resultados
  
  # Exportar resultados
  file_name <- sprintf("50epochs_%s_%i.xlsx", choice, rep)
  export_data(results_50_epochs, resultados, path_results, file_name)
}
