#M�todos para Recomenda��o de Hiperpar�metros de Aprendizado de M�quina na 
#                 Classifica��o de Imagens da Constru��o Civil
#C�digo para data augmentation
#Gerador de imagens artificiais 
#Autor: Andr� Ottoni (UFRB e PPGEE/UFBA)
#Orientadora: Prof. Marcela Novo (UFBA)
#Data 21/01/2021

#Biblioteca Keras
library(keras)

#Defini��o do endere�o da pastas com as imagens originais
base_dir <- "C:/BaseImagens/"

#Configura��es do data augmentation
train_datagen <- image_data_generator(
  rescale = 1/255,         #Normaliza��o
  rotation_range = 40,     #Rota��o  
  zoom_range = 0.2,        #Zoom
  horizontal_flip = TRUE,  #Giro Horizontal
  vertical_flip = TRUE,    #Giro Vertical  
)

cont<-0
#Loop para gera��o das novas imagens
for (j in 1:5){ #N�mero de imagens artificiais para cada imagem original
  for (i in 1:5) { #N�mero de imagens originais
    cont <- cont +1
    fnames <- list.files(base_dir, full.names = TRUE)
    img_path <- fnames[[i]]
    img <- image_load(img_path, target_size = c(1200, 1900))
    img_array <- image_to_array(img)
    img_array <- array_reshape(img_array, c(1, 1200, 1900, 3))
    
    #Leituras imagens no diret�rio e gera��o da nova imagem
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
