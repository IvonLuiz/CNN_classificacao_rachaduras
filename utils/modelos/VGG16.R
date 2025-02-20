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
size_data = 80 # quantidade de imagens 

for (i in 1:size_data) {imgs[[i]] <- readImage(positive[i])}


for (i in 1:size_data) {imgs[[i+size_data]] <- readImage(negative[i])}


# Redimensionar
for (i in 1:(size_data*2)) {imgs[[i]] <- resize(imgs[[i]], 224, 224)}


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

model <- keras_model_sequential() %>%
  layer_conv_2d(filters=64, 
                kernel_size= c(3, 3),
                padding='same',
                activation='relu',
                input_shape= c(224,224,3),
                name='conv1_1') %>%
  layer_conv_2d(filters=64, 
                kernel_size= c(3, 3),
                padding='same',
                activation='relu',
                name='conv1_2') %>%
  layer_max_pooling_2d(pool_size = c(2,2),
                       strides= c(2, 2),
                       name='max_pooling2d_1')



model %>%
  layer_conv_2d(filters= 128, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu', 
                name='conv2_1') %>%
  layer_conv_2d(filters= 128, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                name='conv2_2') %>%
  layer_max_pooling_2d(pool_size=c(2,2), 
                       strides=c(2,2), 
                       name='max_pooling2d_2')


model %>%
  layer_conv_2d(filters= 256, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                input_shape= c(224,224,3),
                name='conv3_1') %>%
  layer_conv_2d(filters= 256, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                name='conv3_2') %>%
  layer_conv_2d(filters= 256, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                name='conv3_3') %>%
  layer_max_pooling_2d(pool_size=c(2,2), 
                       strides=c(2,2), 
                       name='max_pooling2d_3')


model %>%
  layer_conv_2d(filters= 512, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                input_shape= c(224,224,3),
                name='conv4_1') %>%
  layer_conv_2d(filters= 512, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                name='conv4_2') %>%
  layer_conv_2d(filters= 512, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                name='conv4_3') %>%
  layer_max_pooling_2d(pool_size=c(2,2), 
                       strides=c(2,2), 
                       name='max_pooling2d_4')


model %>%
  layer_conv_2d(filters= 512, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                input_shape= c(224,224,3),
                name='conv5_1') %>%
  layer_conv_2d(filters= 512, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                name='conv5_2') %>%
  layer_conv_2d(filters= 512, 
                kernel_size= c(3, 3), 
                padding='same', 
                activation='relu',
                name='conv5_3') %>%
  layer_max_pooling_2d(pool_size=c(2,2), 
                       strides=c(2,2), 
                       name='max_pooling2d_5')


model %>%
  layer_flatten(name='flatten')%>%
  layer_dense(4096, activation = 'relu', name = 'fc_1') %>%
  layer_dropout(0.5, name='dropout_1') %>%
  layer_dense(4096, activation = 'relu', name = 'fc_2') %>%
  layer_dropout(0.5, name='dropout_4') %>%
  layer_dense(2, activation='softmax', name='output')




summary(model)


# Compilar modelo
model %>%
  compile(optimizer = "adam",
          loss = "binary_crossentropy",
          metrics = "accuracy")


#history <- model %>%
  fit(x = train_x, y = train_y,
      epochs = 50,
      validation_split=0.2,
      use_multiprocessing=TRUE)

# Avaliar modelo
model %>% evaluate(train_x, train_y)
model %>% evaluate(test_x, test_y)

# Predictions
model %>% predict(test_x) %>% k_argmax()




summary(base_model)

base_model <- application_vgg16(include_top = TRUE,
                                weights = NULL,
                                input_tensor = NULL,
                                input_shape = NULL,
                                pooling = NULL,
                                classes = 2,
                                classifier_activation = "softmax")

# Compilar modelo
base_model %>%
  compile(  loss = "categorical_crossentropy",
            optimizer = optimizer_adam(learning_rate = 0.01, decay = 1e-6),
            metrics = "accuracy")


history <- base_model %>%
fit(x = train_x, y = train_y,
    epochs = 12,
    validation_split=0.2,
    use_multiprocessing=TRUE)

base_model %>% evaluate(train_x, train_y)
base_model %>% evaluate(test_x, test_y)

# Predictions
base_model %>% predict(test_x) %>% k_argmax()


features <- base_model %>% predict(train_x)
