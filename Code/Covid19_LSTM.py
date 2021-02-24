"""
 Covid-19 Machine Learning: Redes Neuronales 
                  (2021)
  
 .. moduleauthor:: 
 				Alfonso Castillo Orozco
 				Luis Carlos Solano Mora

 About
 =====
  
 Este módulo carga los datos de casos referentes a la pandemia Covid-19, desde el repositorio: 
 COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University.
 A partir de este se hace el tratamiento de los datos para predecir la curva de casos para Costa Rica, mediante
 un modelo LSTM utilizando redes neuronales.

 References
 ==========
 
  * https://github.com/CSSEGISandData/COVID-19
 
"""

#_____________________________________________________________________________________________
# Bibliotecas
#_____________________________________________________________________________________________
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#_____________________________________________________________________________________________
# Funciones
#_____________________________________________________________________________________________
def CargarArchivo(file, online=False):
    if online:
    # Se carga el archivo online
        datos = pd.read_csv(file)
    else:
    # Se carga el archivo local
	    datos = pd.read_csv('../Datos/' + file)
	
	#Se convierte en data frame con pandas 
    datos_pd = pd.DataFrame(datos)
    # Se eliminan las columnas con los nombres "Province/State", "Lat" y "Long"
    datos_pd_reducidos = datos_pd.drop(['Province/State', 'Lat', 'Long'], axis=1)
    # Retorna datos con las 3 columnas eliminadas
    return datos_pd_reducidos

def SeleccionarPaisesAcumulados(datos, paises=''):
    #Se establece la columna 'Country/Region' como índice
    datos = datos.set_index('Country/Region')
    #Se elige solo la info de los países de la lista
    if paises != '':
        datos_paises = datos.loc[paises]
        #Retorna datos solo de los países
        return datos_paises
    #Retorna datos solo de los países
    else:
        return datos

def Analice_archivoAcumulados(archivo, paises):
    datos_reducidos = CargarArchivo(archivo, online =True)
    datos_paises = SeleccionarPaisesAcumulados(datos_reducidos, paises)
    #e. Se crea una lista con elementos desde 0 hasta el número de fechas menos 1
    n_fechas = datos_paises.shape[1]
    x_eje = list(range(0, n_fechas))
    
    #f. Se grafican los datos
    #--Se define los posibles títulos para el gráfico
    titulo_grafico = ''
    if archivo=='time_series_covid19_confirmed_global.csv' or archivo[-40:]=='time_series_covid19_confirmed_global.csv':
        titulo_grafico = 'Casos Confirmados Acumulados de Covid-19'
    elif archivo=='time_series_covid19_deaths_global.csv' or archivo[-37:]=='time_series_covid19_deaths_global.csv':
        titulo_grafico = 'Casos de Muerte Acumulados por Covid-19'
    elif archivo=='time_series_covid19_recovered_global.csv' or archivo[-40:]=='time_series_covid19_recovered_global.csv':
        titulo_grafico = 'Casos Recuperados Acumulados de Covid-19'
    
    #--Impresiones en Consola
    #Columnas Eliminadas
    print("-----------------> ANÁLISIS DE DATOS: "+titulo_grafico)
    #Países de Interés
    print("\n--> Países seleccionados: "+str(paises))
    print(datos_paises)    
        
    #--For para graficar cada país
    for i in range(len(paises)):
        plt.plot(x_eje, datos_paises.loc[paises[i]],label=paises[i])
    #--Títulos del gráfico
    plt.title(titulo_grafico)
    plt.xlabel('Días desde el '+datos_paises.columns[0]+' hasta el '+datos_paises.columns[n_fechas-1]+'  [mes/día/año]')
    plt.ylabel('Casos reportados')
    plt.legend()
    plt.show()
    plt.close()
    print("<----------------- FINALIZA ANÁLISIS DE DATOS <-----------------")

def SeleccionarPaisesDiarios(datos, paises=''):
    #Se establece la columna 'Country/Region' como índice
    datos = datos.set_index('Country/Region')
    #Se elige solo la info de los países de la lista
    if paises != '':
        datos_paises = datos.loc[paises]
        #Se calculan los casos diarios apartir de una resta de los acumulados
        datos_paises = datos_paises.diff(axis=1).fillna(datos_paises.iloc[0,0]).astype(np.int64)
        #Retorna datos solo de los países
        return datos_paises
    #Retorna datos solo de los países
    else:
        #Se calculan los casos diarios apartir de una resta de los acumulados
        datos = datos.diff(axis=1).fillna(datos.iloc[0,0]).astype(np.int64)
        return datos

def Analice_archivoDiarios(archivo, paises):
    datos_reducidos = CargarArchivo(archivo, online =True)
    datos_paises = SeleccionarPaisesDiarios(datos_reducidos, paises)
    #e. Se crea una lista con elementos desde 0 hasta el número de fechas menos 1
    n_fechas = datos_paises.shape[1]
    x_eje = list(range(0, n_fechas))
    
    #f. Se grafican los datos
    #--Se define los posibles títulos para el gráfico
    titulo_grafico = ''
    if archivo=='time_series_covid19_confirmed_global.csv' or archivo[-40:]=='time_series_covid19_confirmed_global.csv':
        titulo_grafico = 'Casos Confirmados Diarios de Covid-19'
    elif archivo=='time_series_covid19_deaths_global.csv' or archivo[-37:]=='time_series_covid19_deaths_global.csv':
        titulo_grafico = 'Casos de Muerte Diarios por Covid-19'
    elif archivo=='time_series_covid19_recovered_global.csv' or archivo[-40:]=='time_series_covid19_recovered_global.csv':
        titulo_grafico = 'Casos Recuperados Diarios de Covid-19'
    
    #--Impresiones en Consola
    #Columnas Eliminadas
    print("-----------------> ANÁLISIS DE DATOS: "+titulo_grafico)
    #Países de Interés
    print("\n--> Países seleccionados: "+str(paises))
    print(datos_paises)    
        
    #--For para graficar cada país
    for i in range(len(paises)):
        plt.plot(x_eje, datos_paises.loc[paises[i]],label=paises[i])
    #--Títulos del gráfico
    plt.title(titulo_grafico)
    plt.xlabel('Días desde el '+datos_paises.columns[0]+' hasta el '+datos_paises.columns[n_fechas-1]+'  [mes/día/año]')
    plt.ylabel('Casos reportados')
    plt.legend()
    plt.show()
    plt.close()
    print("<----------------- FINALIZA ANÁLISIS DE DATOS <-----------------")

def walk_forward_format_2(train_data, in_size, out_size, step=1):
    # Aplanamiento de datos
    data = train_data.reshape(-1,1,1)
    # Cantidad de muestras
    samples = math.floor((data.shape[0]-(in_size+out_size))/step)
    if (samples<1):
        raise NameError("El tamaño de las entradas más las salidas es mayor al tamaño de los datos")
    # Tamaño de los datos de validación
    validation_size = out_size
    # Tamaño de los datos de entrada
    input_size = in_size
    # Arreglo de entrada con muestras walk-forward
    wf_data_in = []
    # Arreglo de validacion con datos walk-forward
    wf_data_val = []
    # Se divide en cada una de las muestras
    for i in range(0,samples):
        wf_data_in.append(data[i*step:i*step+input_size])
        wf_data_val.append(data[i*step+input_size:i*step+input_size+validation_size])
    # Se convierten a  np arrays con la forma deseada
    wf_data_in_np = np.array(wf_data_in).reshape(samples,-1)
    wf_data_val_np = np.array(wf_data_val).reshape(samples,-1)
    return wf_data_in_np, wf_data_val_np

def multi_walk_forward_format(train_data, in_size, out_size, step=1):
    # Se inicializan listas vacías donde se colocarán las muestras
    x_res = []
    y_res = []
    # Se hace un loop para recorrer cada país
    for i in train_data:
        # Se aplana el dato
        data_temp = i.reshape(1,-1)
        # Se llama a la función que divide los datos para un solo país
        x_temp,y_temp = walk_forward_format_2(data_temp,in_size, out_size, step)
        # Cada muestra obtenida para el país dse guarda en la lista con todas las muestras de todos los paises
        # Esto se hace parta los datos de entrada
        for j in x_temp:
            x_res.append(j)
            # Y para los de salida
        for j in y_temp:
            y_res.append(j)
        # Finalmente se convierten las listas a arreglos de numpay y se les da la forma deseada
        x_np = np.array(x_res).reshape(-1,in_size)
        y_np = np.array(y_res).reshape(-1,out_size)
    return x_np,y_np


#####################
# 		MAIN: Análisis de datos
#####################

# Main Parte 3: Se visualizan los datos

# Datos locales de respaldo
file1 = 'time_series_covid19_confirmed_global.csv' 
file2 = 'time_series_covid19_deaths_global.csv'
file3 = 'time_series_covid19_recovered_global.csv'

# URL con los datos
url1 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url2 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url3 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

paises = ['Costa Rica']

Analice_archivoAcumulados(url1, paises)
Analice_archivoDiarios(url1,paises)

Analice_archivoAcumulados(url2, paises)
Analice_archivoDiarios(url2,paises)

Analice_archivoAcumulados(url3, paises)
Analice_archivoDiarios(url3,paises)

#####################
# 		MAIN: Redes Neuronales
#####################

#_____________________________________________________________________________________________
# Datos
#_____________________________________________________________________________________________

# Se escoge cúales datos se utilizarán (confirmados, defunciones o recuperados)
url = url1
#url = url2
#url = url3

# Se cargan los datos a un dataframe
datos_totales = CargarArchivo(url,online=True)
# Se imprimen los datos para observarlos
print(datos_totales)
#Se comprueba que no hayan campos vacíos en el dataframe
print(datos_totales.isnull().values.any())

# Días a excluir de los datos reales
dias_pronostico = 30
# Días que se ingresarán como entrada a la red
dias_disponibles = 320
#paises = ['Costa Rica','Guatemala', 'El Salvador', "Panama", 'Honduras']
paises = ['Costa Rica']

#datos_pais = SeleccionarPaisesAcumulados(CargarArchivo(url,online=True), paises)
datos_pais = SeleccionarPaisesDiarios(CargarArchivo(url,online=True), paises)
datos_pais


#Se seleccionan los datos para el entrenamiento
train_data = datos_pais.iloc[:,:-dias_pronostico].to_numpy()
print(train_data.shape)

train_x,train_y = multi_walk_forward_format(train_data,dias_disponibles,dias_pronostico,10)

print(train_x.shape)

#_____________________________________________________________________________________________
# Escalar datos
#_____________________________________________________________________________________________

# Se establece que la escala se hará utilizando los datos de entreno para los máximos y mínimos,
#es decir, al valor mínimo en datos_entreno se asigna como 0, y el máximo valor de datos_entreno se asigna como 1

sc_in = MinMaxScaler(feature_range=(0,1))
sc_out = MinMaxScaler(feature_range=(0,1))

# Se hace el ajuste de los datos, tanto para los datos de entrada com para los de salida, cada uno por aparte
sc_in.fit(train_x)
sc_out.fit(train_y)

# Se escalan los datos de entrada y salida del entreno
train_x_s = sc_in.transform(train_x)
train_y_s = sc_out.transform(train_y)

train_x_r = train_x_s.reshape(train_x_s.shape[0],train_x_s.shape[1],1)
train_y_r = train_y_s.reshape(train_y_s.shape[0],train_y_s.shape[1],1)


#_____________________________________________________________________________________________
# Definir parámetros
#_____________________________________________________________________________________________


verbose, epochs, batch_size = 1, 1, 1
n_timesteps, n_features, n_outputs = train_x_r.shape[1], train_x_r.shape[2], train_y_r.shape[1]

#_____________________________________________________________________________________________
# Modelo LSTM
#_____________________________________________________________________________________________


#Modelo A
#Inicialización del modelo
model = Sequential()

#Se agrega una capa interna LSTM 
model.add(LSTM(30, return_sequences=False, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.2))

#Se agrega la capa externa de salida
model.add(Dense(n_outputs, activation='relu'))

#_____________________________________________________________________________________________
# Compile/Fit del Modelo LSTM
#_____________________________________________________________________________________________

# Se establecen el método para obtener el error y para minimizarlo
model.compile(loss='mse', optimizer='adam')

# Se entrena el modelo con los datos de entreno
model.fit(train_x_r, train_y_r, epochs=epochs, batch_size=batch_size, verbose=verbose)

#Se despliega el modelo implementado, Modelo A
model.summary()


pais = ['Costa Rica']
#datos_pais_test = SeleccionarPaisesAcumulados(CargarArchivo(url,online=True), pais)
datos_pais_test = SeleccionarPaisesDiarios(CargarArchivo(url,online=True), paises)
# Se seleccionan los últimos datos disponibles menos los que se usaran para comprobar si la predicción es correcta
test_x   = datos_pais_test.iloc[:,-(dias_pronostico+dias_disponibles):-dias_pronostico].to_numpy()
# Se escalan los datos de entrada
test_x_s = sc_in.transform(test_x)
# Se acomodan en la forma que el modelo de Keras los recibe
test_x_r = test_x.reshape(test_x_s.shape[0],test_x_s.shape[1],1)

#_____________________________________________________________________________________________
# Grafica de Datos Reales utilizados para entrenar el Modelo LSTM
#_____________________________________________________________________________________________
plt.figure("Datos de Entrenamiento: Costa Rica")
for i in test_x:
  plt.plot(i)
plt.title('Datos Diarios Confirmados, Costa Rica: Entrenamiento')
plt.xlabel('Días para entrenamiento')
plt.ylabel('Casos reportados')

#_____________________________________________________________________________________________
# Predicción de los datos y tratamiento de los datos obtenidos
#_____________________________________________________________________________________________ 

# Se realiza la predicción del modelo
prediction = model.predict(test_x_r)

print(prediction.shape)
print(prediction)


# El resultado está escalado, por lo que es necesario invertir el escalamiento para obtener los valores en la escala correcta
prediction_rescaled = sc_out.inverse_transform(prediction)
print(prediction_rescaled)

test_y = datos_pais.iloc[:,-dias_pronostico:].to_numpy()

print(prediction)
print(test_y)


# Se obtienen los datos reales que se querían predecir, para poder comparar
test_y = datos_pais.iloc[:,-dias_pronostico:].to_numpy()

print(prediction)
print(test_y)

#_____________________________________________________________________________________________
# Resultados Gráficos de Predicción
#_____________________________________________________________________________________________

#_____________________________________________________________________________________________
# Grafica de Datos de Predicción vs Datos Reales correspondientes a ese mismo periodo
#_____________________________________________________________________________________________
plt.figure("Resultados Costa Rica")
plt.plot(test_y[0])
plt.plot(prediction_rescaled[0])
plt.title('Datos Diarios Confirmados Predicción vs Reales: Costa Rica')
plt.legend(("Real", "Pred"))
plt.xlabel('Días para Predicción')
plt.ylabel('Casos reportados')
plt.show()
plt.close()

#_____________________________________________________________________________________________
# Grafica de Datos de Predicción vs Datos Reales TOTALES, todo el archivo
#_____________________________________________________________________________________________
plt.figure("Resultados Generales Costa Rica")
n_fechas = datos_pais_test.shape[1]
x_eje_r = list(range(0, n_fechas))
x_eje_p = list(range(n_fechas-dias_pronostico, n_fechas))
plt.plot(x_eje_r, datos_pais_test.loc[pais[0]])
plt.plot(x_eje_p, prediction_rescaled[0])
plt.title('Datos Totales Diarios Confirmados Predicción vs Reales: Costa Rica')
plt.legend(("Real", "Pred"))
plt.xlabel('Días para Entrenamiento + Predicción')
plt.ylabel('Casos reportados')
plt.show()
plt.close()