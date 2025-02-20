# Bibliotecas
library(keras)
#install_keras(method="virtualenv", envname="myenv", pip_options = "--no-cache-dir")
library(tensorflow)
library(caTools)
#library(readr)
#library(purrr)
#library(graphics)
#install.packages("BiocManager") 
#BiocManager::install("EBImage")
library(EBImage)

# Diretório das imagens
positive = list.files(path = "pos", pattern = ".jpg", full.names = T)
negative = list.files(path = "neg", pattern = ".jpg", full.names = T)

# Ler imagens e guarda-las num array
imgs = list()
size_data = 20 # quantidade de imagens 

for (i in 1:size_data)
{imgs[[i]] <- readImage(positive[i])}


for (i in 1:size_data)
{imgs[[i+size_data]] <- readImage(negative[i])}


# Redimensionar
for (i in 1:(size_data*2)) {imgs[[i]] <- resize(imgs[[i]], 28, 28)}

# Reshape
for (i in 1:(size_data*2)) {imgs[[i]] <- array_reshape(imgs[[i]], c(28, 28, 3))}
str(imgs)



# Separando dados em treino e teste
train_x <- NULL
test_x <- NULL
train_y <- list()
test_y <- list()

split = sample.split(imgs, SplitRatio = 0.8)
train_x = list(imgs, split = TRUE)
test_x = list(imgs, split = FALSE)




size_to_train = size_data-10

for (i in 1:size_to_train) {train_x <- rbind(train_x, imgs[[i]])}
for (i in (size_data+1):(size_data+size_to_train)) {train_x <- rbind(train_x, imgs[[i]])}


for (i in (size_to_train+1):size_data) {test_x <- rbind(test_x, imgs[[i]])}
for (i in (size_data+1+size_to_train):(size_data*2)) {test_x <- rbind(test_x, imgs[[i]])}


for (i in 1:size_to_train) {train_y <- append(train_y, 1)}     # 1 = tem rachadura
for (i in 1:size_to_train) {train_y <- append(train_y, 0)}   # 0 = sem rachadura


for (i in 1:size_to_train) {test_y <- append(test_y, 1)}    # 1 = tem rachadura
for (i in 1:size_to_train) {test_y <- append(test_y, 0)}   # 0 = sem rachadura

# One Hot Encoding
train_labels <- to_categorical(train_y)
test_labels <- to_categorical(test_y)

# Construindo modelo
# Rede neural convolucional com matriz de tamanho 3x3
model <- keras_model_sequential() %>%
  layer_conv_2d(filters=32, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,3)) %>%
  layer_conv_2d(filters=64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters=64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2))

summary(model)


# Transformamos a matriz num vetor e passamos para o resto da rede neural
model %>%   
  layer_flatten() %>%           
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu")

summary(model)  


# Compilar modelo
model %>%
  compile(optimizer = "adam",
          loss = "sparse_categorical_cossentropy",
          metrics = "accuracy")

history <- model %>%
  fit(x = train_x, y = train_y,
      epochs = 10,
      validation_split=0.2,
      use_multiprocessing=TRUE)




# Evaluation e prediction - dados de treino
model %>% evaluate(train_x, train_labels)


# model <- keras_model_sequential()
# model %>%
#   layer_dense(units = 256, activation = 'relu', input_shape = c(2352)) %>%
#   layer_dense(units = 128, activation = 'relu') %>%
#   layer_dense(units = 2, activation = 'softmax')

# # Compilar
# model %>%
#   compile(loss = 'binary_crossentropy',
#           optimizer = optimizer_rmsprop(),
#           metrics = c('accuracy'))
# 
# # Fit model
# history <- model %>%
#   fit(train_x,
#       train_labels,
#       epochs = 30,
#       batch_size = 32,
#       validation_split = 0.2)