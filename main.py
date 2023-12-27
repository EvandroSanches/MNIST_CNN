from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import cv2
import numpy as np

#Reconhecer digitos através de imagens
#Modelo foi treinado apenas com imagens de numero brancos em fundo preto
#Apesar de converter imagens para preto e branco, não trabalha bem com imagens coloridas

batch_size = 100
epochs = 5


def Pre_Processamento(previsores):
    img = np.empty((0,28,28))

    #removendo canal de cor
    previsores = previsores.reshape(previsores.shape[0],-1)

    #redimencionando a imagem
    previsores = cv2.resize(previsores,(28,28))

    #invertendo cores
    previsores = cv2.bitwise_not(previsores)

    #reformulando shape
    previsores = np.append(img, np.array([previsores]), axis=0)
    previsores = previsores.reshape(previsores.shape[0],28,28,1)


    #Transformando varial em formato float32 e normalizando valores
    previsores = previsores.astype('float32')
    previsores = previsores / 255


    return previsores

def CarregaDados():
    #Carregando base de dados do próprio Keras
    (x,y), (x_teste, y_teste) = mnist.load_data()

    #Unificando dados de treinamento e teste
    previsores = np.concatenate((x, x_teste))
    classes = np.concatenate((y,y_teste))

    previsores = previsores.reshape((previsores.shape[0],28,28,1))

    #Transformando varial em formato float32 e normalizando valores
    previsores = previsores.astype('float32')
    previsores = previsores / 255

    #Codificando classes
    classes = to_categorical(classes, 10)

    return previsores, classes

def CriaModelo():
    modelo = Sequential()

    #Primeira camada de convolução
    #Detecta caracteristicas, normaliza dados para processamento mais rapido e maximiza as caracteristicas
    modelo.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1) ))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D((2,2)))

    #Segunda camada de convolução
    modelo.add(Conv2D(32, (3,3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D(2,2))

    #Camada que transforma a entreda matriz em um vetor
    modelo.add(Flatten())

    #Camada densa com dropout para evitar overfitting
    modelo.add(Dense(units=140, activation='relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=140, activation='relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=10, activation='softmax'))

    #Compila o modelo com parametro de função de perda e optimizador
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo

def Treinamento():
    previsores, classes = CarregaDados()
    modelo = KerasClassifier(build_fn=CriaModelo, batch_size=batch_size, epochs=epochs)
    result = cross_val_score(estimator=modelo, X=previsores, y=classes, cv=10)

    print(result)
    print(result.mean())
    print(result.std())

def GeraModelo():
    modelo = CriaModelo()
    previsores, classes = CarregaDados()

    modelo.fit(previsores, classes, batch_size=batch_size, epochs=epochs)
    modelo.save('Modelo.0.1')


#Deverá ser passado o caminho da imagem
def Previsao(caminho):
    modelo = load_model('Modelo.0.1')
    previsores = cv2.imread(caminho)

    previsores = Pre_Processamento(previsores)

    result = modelo.predict(previsores)
    result = np.argmax(result)
    print(result)



