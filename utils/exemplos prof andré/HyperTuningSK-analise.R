#Métodos para Recomendação de Hiperparâmetros de Aprendizado de Máquina na 
#                 Classificação de Imagens da Construção Civil
#HyperTuningSK 1.0 - Recomendação de Hiperparâmetros usando ANOVA e Scott-Knott 
#Autor: André Luiz C. Ottoni - CETEC/UFRB e PPGEE/UFBA
#Orientadora: Prof. Marcela Novo
#Modificado: 14/11/2022

#Limpa dados
rm(list=ls())

#Scott-Knott Clustering Algorithm
#Package ScottKnott version 1.2-7
library(ScottKnott)

#-----------------Leitura dos dados - Selecionar os dados para analisar e comentar os demais
dados <- read.delim("C:/Dados/Pesquisa/Doutorado/Divulgacao-site/dadosHyperTuningSK/densenet01.txt")
h<-c("adagrad001", "adagrad005", "adagrad010", "adagrad015", "adagrad020", "adagrad025", "adam001", "adamax001", "sgd001", "sgd005", "sgd010", "sgd015","sgd020", "sgd025")
titulo="Estudo de Caso 1 - Densenet"

#dados <- read.delim("C:/Dados/Pesquisa/Doutorado/Divulgacao-site/dadosHyperTuningSK/vgg1601.txt")
#h<-c("adagrad001", "adagrad005", "adagrad010", "adagrad015", "adagrad020", "adamax001", "sgd001", "sgd005", "sgd010", "sgd015","sgd020","sgd025")
#titulo="Estudo de Caso 1 - VGG16"

#dados <- read.delim("C:/Dados/Pesquisa/Doutorado/Divulgacao-site/dadosHyperTuningSK/densenet02.txt")
#h<-c("adagrad001", "adagrad005", "adagrad010", "adagrad015", "adagrad020", "adagrad025", "adam001", "adamax001", "sgd001", "sgd005", "sgd01", "sgd015","sgd020", "sgd025")
#titulo="Estudo de Caso 2 - Densenet"

#dados <- read.delim("C:/Dados/Pesquisa/Doutorado/Divulgacao-site/dadosHyperTuningSK/vgg1602.txt")
#h<-c("adagrad001", "adagrad005", "adagrad010", "adagrad015", "adagrad020", "adagrad025", "adamax001", "sgd005", "sgd010", "sgd015","sgd020", "sgd025")
#titulo="Estudo de Caso 2 - VGG16"

#dados <- read.delim("C:/Dados/Pesquisa/Doutorado/Divulgacao-site/dadosHyperTuningSK/densenet03.txt")
#h<-c("adagrad001", "adagrad005", "adagrad010", "adagrad015", "adagrad020", "adagrad025", "adam001", "adamax001", "sgd001", "sgd005", "sgd01", "sgd015","sgd020", "sgd025")
#titulo="Estudo de Caso 3 - Densenet"

#dados <- read.delim("C:/Dados/Pesquisa/Doutorado/Divulgacao-site/dadosHyperTuningSK/vgg1603.txt")
#h<-c("adagrad001", "adagrad005", "adagrad015", "sgd001", "sgd005", "sgd015", "sgd025")
#titulo="Estudo de Caso 3 - VGG16"

attach(dados)

tam_h<-length(h)

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
  hsk <- h[ind] #Parâmetros recomendados
  print("Hiperparâmetros recomendados (otimizador + lr)):")
  print(hsk)
}else{
  #Parâmetros não apresentam diferença: então gera parâmetro aleatório
  print("Hiperparâmetros não apresentam diferença ou medidas de adequação não satisfeitas!")
}


plot(sk1,
     col=rainbow(max(sk1$groups)),
     rl=FALSE,
     id.las=2,
     cex.main = 0.9,
     title=titulo,
     font = 1,
     family = "serif",
     xlab = "",
     ylab = "Acurácia (Média)")



