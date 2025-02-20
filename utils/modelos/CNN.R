# Bibliotecas
library(tensorflow)
#install_tensorflow()
library(keras)
#install_keras(method="virtualenv", envname="myenv", pip_options = "--no-cache-dir")

#install.packages("BiocManager") 
#BiocManager::install("EBImage")
library(EBImage)

# Diretório das imagens
positive = list.files(path = "C:/Users/j_ls_/Downloads/Positive", pattern = ".jpg", full.names = T)
negative = list.files(path = "C:/Users/j_ls_/Downloads/Negative", pattern = ".jpg", full.names = T)

# Ler imagens e guarda-las
imgs = list()
size_data = 100 # quantidade de imagens 

for (i in 1:size_data) {imgs[[i]] <- readImage(positive[i])}


for (i in 1:size_data) {imgs[[i+size_data]] <- readImage(negative[i])}


# Redimensionar
for (i in 1:(size_data*2)) {imgs[[i]] <- resize(imgs[[i]], 28, 28)}


# Separando dados em treino e teste
train_x <- list()
test_x <- list()
train_y <- list()
test_y <- list()

# Porcentagem para treinar
size_to_train = size_data*0.8

# train_x
for (i in 1:size_to_train) {train_x <- append(train_x, imgs[i])}
for (i in (size_data+1):(size_data+size_to_train)) {train_x <- append(train_x, imgs[i])}

# test_x
for (i in (size_to_train+1):size_data) {test_x <- append(test_x, imgs[i])}
for (i in (size_data+1+size_to_train):(size_data*2)) {test_x <- append(test_x, imgs[i])}


# Combinar todas as imagens em um so bloco
train_x <- combine(train_x)
test_x <- combine(test_x)

# Visualizar imagens
display(tile(train_x, size_to_train))
display(tile(test_x, (size_data-size_to_train)))

# Reordenar o número de instâncias para a primeira posição
train_x <- aperm(train_x, c(4,1,2,3))
test_x <- aperm(test_x, c(4,1,2,3))


# train and test y
train_labels <- rep(1:0, each=size_to_train)
test_labels <- rep(1:0, each=(size_data - size_to_train))


# One Hot Encoding
train_y <- to_categorical(train_labels)
test_y <- to_categorical(test_labels)



# Construindo modelo
# Rede neural convolucional com matriz de tamanho 3x3
model <- keras_model_sequential() %>%
 layer_conv_2d(filters=32, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,3)) %>%
 layer_conv_2d(filters=64, kernel_size = c(3,3), activation = "relu") %>%
 layer_max_pooling_2d(pool_size = c(2,2)) %>%
 layer_conv_2d(filters=64, kernel_size = c(3,3), activation = "relu") %>%
 layer_max_pooling_2d(pool_size = c(2,2))

# Transformamos a matriz num vetor e passamos para o resto da rede neural
model %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 2, activation = "sigmoid")

summary(model)


# Compilar modelo
model %>%
  compile(optimizer = "adam",
          loss = "categorical_crossentropy",
          metrics = "accuracy")


history <- model %>%
  fit(x = train_x, y = train_y,
      epochs = 50,
      validation_split=0.2,
      use_multiprocessing=TRUE)

# Avaliar modelo
model %>% evaluate(train_x, train_y)
model %>% evaluate(test_x, test_y)

# Predictions
model %>% predict(test_x) %>% k_argmax()



# Criando o modelo 2

model <- keras_model_sequential()

model %>% 
  layer_conv_2d(filters = 256,
                kernel_size = c(3,3),
                activation = "relu",
                input_shape = c(28,28,3))%>%
  layer_max_pooling_2d(pool_size = c(3,3), strides = 2)%>%
  layer_conv_2d(filters = 128,
                kernel_size = c(5,5),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = 2)%>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = 2)%>%
  layer_flatten()%>%
  layer_dense(units=256)%>% layer_activation_leaky_relu(alpha = 0.1)%>%
  layer_dropout(rate=0.4)%>%
  layer_dense(units=2, activation = "sigmoid")

summary(model)

model %>% compile(loss= "categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics="accuracy")

history <- model %>%
  fit(train_x, train_y, epoch=50,batch_size=32, validation_split=0.2)

# Evaluation e prediction - dados de treino
model %>% evaluate(train_x, train_y)
model %>% evaluate(test_x, test_y)

# Predictions

model %>% predict(train_x) %>% k_argmax()
