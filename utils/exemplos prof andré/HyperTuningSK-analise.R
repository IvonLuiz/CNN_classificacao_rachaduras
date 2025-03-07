#M�todos para Recomenda��o de Hiperpar�metros de Aprendizado de M�quina na 
#                 Classifica��o de Imagens da Constru��o Civil
#HyperTuningSK 1.0 - Recomenda��o de Hiperpar�metros usando ANOVA e Scott-Knott 
#Autor: Andr� Luiz C. Ottoni - CETEC/UFRB e PPGEE/UFBA
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
  hsk <- h[ind] #Par�metros recomendados
  print("Hiperpar�metros recomendados (otimizador + lr)):")
  print(hsk)
}else{
  #Par�metros n�o apresentam diferen�a: ent�o gera par�metro aleat�rio
  print("Hiperpar�metros n�o apresentam diferen�a ou medidas de adequa��o n�o satisfeitas!")
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
     ylab = "Acur�cia (M�dia)")



