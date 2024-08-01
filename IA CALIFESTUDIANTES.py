import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

num_capas = int(input("Ingrese el número de capas de la red: "))
numneurona = [int(input(f"Ingrese el número de neuronas para la capa {i+1}: ")) for i in range(num_capas)]
epochs = int(input("Ingrese el número de épocas de entrenamiento: "))

df = pd.read_csv("basededatos1.txt", sep="\t")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_clases = len(encoder.classes_)
y_one_hot = to_categorical(y_encoded, num_classes=num_clases)

def ModeloRedNeuro(num_capas, numneurona, epochs):
    Tabla = []
    for test_size in np.arange(0.1, 1.0, 0.1):
        train_size = 1 - test_size
        X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=test_size, random_state=42)

        modelo = Sequential()

        for i in range(num_capas):
            if i == 0:
                modelo.add(Dense(numneurona[i], input_dim=X_train.shape[1], activation='relu'))
            else:
                modelo.add(Dense(numneurona[i], activation='relu'))
        modelo.add(Dense(num_clases, activation='softmax'))

        modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        print(f"Cargando {test_size}")
        modelo.fit(X_train, y_train, epochs=epochs, batch_size=5, verbose=0, callbacks=[early_stopping], validation_data=(X_test, y_test))
        perdida, exactitud = modelo.evaluate(X_test, y_test, verbose=0)
        Tabla.append([train_size, exactitud, perdida])


    Tabla.reverse()
    Tabla_final =pd.DataFrame(Tabla, columns=['|Entrenamiento|', '|Exactitud|','|Perdida|'])
    print("##### RESULTADOS FINALES:######")
    print(Tabla_final)

ModeloRedNeuro(num_capas, numneurona, epochs)
