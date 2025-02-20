#Códigos do Doutorado - 2021
#Métodos para Recomendação de Hiperparâmetros de Aprendizado de Máquina na 
#                 Classificação de Imagens da Construção Civil
#Estudo de Caso 1: Reconhecimento de Vegetação em Edificações 
#Objetivo: Experimentos para plotar imagens de teste
#Classes - 0: sem vegetação; 1: com vegetação
#Autor: André Ottoni (UFRB e PPGEE/UFBA)
#Orientadora: Profª. Marcela Novo (UFBA)
#05/12/2021

#---1) Incialização de Variáveis
VP = 0
FP = 0
VN = 0
FN = 0

#----2) Imagens da classe 0 (sem vegetação)-------------

fnames <- list.files(test_p_dir, full.names = TRUE)
op <- par(mfrow = c(3, 5), pty = "s", mar = c(1, 0, 1, 0))
for (k in 1:45) {
  img_path <- fnames[[k]]
  img <- image_load(img_path, target_size = c(50, 50))
  img_tensor <- image_to_array(img)
  img_tensor <- array_reshape(img_tensor, c(1, 50, 50, 3))
  img_tensor <- img_tensor / 255
  
  resp <- model%>% predict_classes(img_tensor)
  
  img2 <- image_load(img_path, target_size = c(150, 150))
  img_tensor2 <- image_to_array(img2)
  img_tensor2 <- array_reshape(img_tensor2, c(1, 150, 150, 3))
  img_tensor2 <- img_tensor2 / 255
  dim(img_tensor2)
  plot(as.raster(img_tensor2[1,,,]))
  
  if (resp == 0){ 
    rect(1, 2, 10, 50, col = "darkgreen") #Acerto: Verde
    VN = VN + 1
  }else{
    rect(1, 2, 10, 50, col = "darkred") #Erro: Vermelho
    FN = FN + 1
  }
}
par(op)

#-------3)Imagens da classe 1 (com vegetação)-------------
fnames <- list.files(test_v_dir, full.names = TRUE)
op <- par(mfrow = c(3, 5), pty = "s", mar = c(1, 0, 1, 0))
for (k in 1:45) {
  img_path <- fnames[[k]]
  
  img <- image_load(img_path, target_size = c(50, 50))
  img_tensor <- image_to_array(img)
  img_tensor <- array_reshape(img_tensor, c(1, 50, 50, 3))
  img_tensor <- img_tensor / 255
  
  resp <- model%>% predict_classes(img_tensor)
  
  img2 <- image_load(img_path, target_size = c(150, 150))
  img_tensor2 <- image_to_array(img2)
  img_tensor2 <- array_reshape(img_tensor2, c(1, 150, 150, 3))
  img_tensor2 <- img_tensor2 / 255
  dim(img_tensor2)
  plot(as.raster(img_tensor2[1,,,]))
  
  if (resp == 1){ 
    rect(1, 2, 10, 50, col = "darkgreen") #Acerto: Verde
    VP = VP + 1
  }else{
    rect(1, 2, 10, 50, col = "darkred") #Erro: Vermelho
    FP = FP + 1
  }
}
par(op)

#-------------4) Resultados da Matriz de Cofunsão

VN
FN
VP
FP

#-----------------------------------------------

#------------5) Acurácia no teste
#Leitura das imagens de teste
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
restest

