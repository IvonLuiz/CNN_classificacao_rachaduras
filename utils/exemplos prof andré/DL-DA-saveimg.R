#Métodos para Recomendação de Hiperparâmetros de Aprendizado de Máquina na 
#                 Classificação de Imagens da Construção Civil
#Código para data augmentation
#Gerador de imagens artificiais 
#Autor: André Ottoni (UFRB e PPGEE/UFBA)
#Orientadora: Prof. Marcela Novo (UFBA)
#Data 21/01/2021

#Biblioteca Keras
library(keras)

#Definição do endereço da pastas com as imagens originais
base_dir <- "C:/BaseImagens/"

#Configurações do data augmentation
train_datagen <- image_data_generator(
  rescale = 1/255,         #Normalização
  rotation_range = 40,     #Rotação  
  zoom_range = 0.2,        #Zoom
  horizontal_flip = TRUE,  #Giro Horizontal
  vertical_flip = TRUE,    #Giro Vertical  
)

cont<-0
#Loop para geração das novas imagens
for (j in 1:5){ #Número de imagens artificiais para cada imagem original
  for (i in 1:5) { #Número de imagens originais
    cont <- cont +1
    fnames <- list.files(base_dir, full.names = TRUE)
    img_path <- fnames[[i]]
    img <- image_load(img_path, target_size = c(1200, 1900))
    img_array <- image_to_array(img)
    img_array <- array_reshape(img_array, c(1, 1200, 1900, 3))
    
    #Leituras imagens no diretório e geração da nova imagem
    augmentation_generator <- flow_images_from_data(
      img_array,
      generator = train_datagen,
      batch_size = 1
    )
    batch <- generator_next(augmentation_generator)
    
    #Salvar a nova imagem em jpeg
    nome <-paste(cont,".jpeg") 
    jpeg(filename = nome,width = 1200, height = 900, units = "px")
    plot(as.raster(batch[1,,,]))
    dev.off()
  }
}
