from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def CargarArchivo(file, online=False):
  # Se carga el archivo y se convierte en data frame con pandas
  # Se añade la carpeta "Datos" al path si el archivo es local
  if online:
    datos = pd.read_csv(file)
  else:
	  datos = pd.read_csv('../Datos/' + file)
   
  datos_pd = pd.DataFrame(datos)
  # Se eliminan las columnas con los nombres "Province/State", "Lat" y "Long"
  datos_pd_reducidos = datos_pd.drop(['Province/State', 'Lat', 'Long'], axis=1)
  # Se colocan los paises como índices de los datos
  #datos_pd_reducidos = datos_pd_reducidos.set_index('Country/Region')
  # Retorna datos con las 3 columnas eliminadas
  return datos_pd_reducidos

def SeleccionarPaisesAcumulados(datos, paises=''):
	#Se establece la columna 'Country/Region' como índice
	datos = datos.set_index('Country/Region')
	#Se elige solo la info de los países de la lista
	if paises != '':
		datos_paises = datos.loc[paises]
		return datos_paises
	#Retorna datos solo de los países
	return datos

def SeleccionarPaisesDiarios(datos, paises):
    #Se establece la columna 'Country/Region' como índice
    datos = datos.set_index('Country/Region')
    #Se elige solo la info de los países de la lista
    datos_paises = datos.loc[paises]
    #Se calculan los casos diarios apartir de una resta de los acumulados
    datos_paises_diarios = datos_paises.diff(axis=1).fillna(datos_paises.iloc[0,0]).astype(np.int64)
    #Retorna datos solo de los países
    return datos_paises_diarios

def walk_forward_format_2(train_data, in_size, out_size, step=1):
  # Aplanamiento de datos
  data = train_data.reshape(-1,1,1)
  # print(data)
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
  for i in range(0,samples):
    wf_data_in.append(data[i*step:i*step+input_size])
    # print(wf_data_in)
    wf_data_val.append(data[i*step+input_size:i*step+input_size+validation_size])
  # Se convierten a  np arrays con la forma deseada
  wf_data_in_np = np.array(wf_data_in).reshape(samples,-1)
  wf_data_val_np = np.array(wf_data_val).reshape(samples,-1)
  return wf_data_in_np, wf_data_val_np

def multi_walk_forward_format(train_data, in_size, out_size, step=1):
  x_res = []
  y_res = []
  print(type(train_data))
  print(train_data.shape)
  for i in train_data:
    # print(type(i))
    # print(" Holi")
    data_temp = i.reshape(1,-1)
    # print(data_temp.shape)
    x_temp,y_temp = walk_forward_format_2(data_temp,in_size, out_size, step)
    # print(" FFF ", x_temp.shape )
    # plt.figure
    for j in x_temp:
      # print(x_temp[j])
      # print(x_temp.shape)
      # print(j)
      # plt.plot(j)
      x_res.append(j)
      
    for j in y_temp:
      y_res.append(j)
    # plt.show()
    # print(x_res)
    x_np = np.array(x_res).reshape(-1,in_size)
    y_np = np.array(y_res).reshape(-1,out_size)
  return x_np,y_np


#####################
# 		MAIN
#####################

# URL con los datos
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
#url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

# Se cargan los datos a un dataframe
datos_totales = CargarArchivo(url,online=True)
# Se imprimen los datos para observarlos
print(datos_totales)
print(datos_totales.isnull().values.any())

#

# Días a excluir de los datos reales
dias_pronostico = 30#5
dias_disponibles = 320#30
# paises = ['Costa Rica','Guatemala', 'El Salvador', "Panama", 'Honduras', 'Mexico']
#paises = ['Costa Rica','Guatemala', 'El Salvador', "Panama", 'Honduras']
paises = ['Costa Rica']

#datos_pais = SeleccionarPaisesAcumulados(CargarArchivo(url,online=True), paises)
datos_pais = SeleccionarPaisesDiarios(CargarArchivo(url,online=True), paises)
# datos_pais = SeleccionarPaisesAcumulados(CargarArchivo(url,online=True))
datos_pais

#

train_data = datos_pais.iloc[:,:-dias_pronostico].to_numpy()
# train_data_3d=train_data.reshape(train_data.shape[0],train_data.shape[1],1)
print(train_data.shape)

# train_x,train_y = walk_forward_format_2(train_data_3d,dias_disponibles,dias_pronostico,10)
# train_x,train_y = walk_forward_format_2(train_data,dias_disponibles,dias_pronostico,10)
train_x,train_y = multi_walk_forward_format(train_data,dias_disponibles,dias_pronostico,10)

# for i in train_x:
#   plt.plot(i)
# plt.show()

print(train_x.shape)
# print(train_y.shape)

#

sc_in = MinMaxScaler(feature_range=(0,1))
sc_out = MinMaxScaler(feature_range=(0,1))

# Se establece que la escala se hará utilizando los datos de entreno para los máximos y mínimos,
# es decir, al valor mínimo en datos_entreno se asigna como 0, y el máximo valor de datos_entreno se asigna como 1

# plain_train_x = train_x.reshape(1,-1)
# plain_train_y = train_y.reshape(1,-1)

# sc_0 = MinMaxScaler(feature_range=(0,1))
# sc_0.fit(train_x)
# print(train_x.shape)
# sc_info = sc_0.scale_
# print(sc_info.shape)
# print("Scale: ", sc_info)

print(train_x.shape)
sc_in.fit(train_x)
sc_out.fit(train_y)
#sc.fit(datos_entreno)

sc_info = sc_in.data_max_
print(sc_info.shape)
print("Scale: ", sc_info)

# Se escalan los datos de entrada y salida del entreno
# for sample in train_x: 
train_x_s = sc_in.transform(train_x)
train_y_s = sc_out.transform(train_y)
# print(train_y_s.shape)

train_x_r = train_x_s.reshape(train_x_s.shape[0],train_x_s.shape[1],1)
train_y_r = train_y_s.reshape(train_y_s.shape[0],train_y_s.shape[1],1)

# define parameters
verbose, epochs, batch_size = 1, 1, 1
n_timesteps, n_features, n_outputs = train_x_r.shape[1], train_x_r.shape[2], train_y_r.shape[1]
# define model
model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(n_outputs, activation='relu'))

model.compile(loss='mse', optimizer='adam')
# fit network
model.fit(train_x_r, train_y_r, epochs=epochs, batch_size=batch_size, verbose=verbose)

pais = ['Costa Rica']
#datos_pais_test = SeleccionarPaisesAcumulados(CargarArchivo(url,online=True), pais)
datos_pais_test = SeleccionarPaisesDiarios(CargarArchivo(url,online=True), paises)
test_x = datos_pais_test.iloc[:,-(dias_pronostico+dias_disponibles):-dias_pronostico].to_numpy()
test_x_s = sc_in.transform(test_x)
test_x_r = test_x.reshape(test_x_s.shape[0],test_x_s.shape[1],1)
#for i in test_x_s:
for i in test_x:
  plt.plot(i)
prediction = model.predict(test_x_r)

print(prediction.shape)
print(prediction)
prediction_rescaled = sc_out.inverse_transform(prediction)
print(prediction_rescaled)

test_y = datos_pais.iloc[:,-dias_pronostico:].to_numpy()

print(prediction)
print(test_y)

model.summary()

plt.figure("Resultados")
plt.plot(test_y[0])
plt.plot(prediction_rescaled[0])
#plt.plot(prediction[0])
plt.legend(("Real", "Pred"))
plt.show()
#plt.close()

plt.figure("Resultados General Costa Rica")
n_fechas = datos_pais_test.shape[1]
x_eje_r = list(range(0, n_fechas))
x_eje_p = list(range(n_fechas-dias_pronostico, n_fechas))
plt.plot(x_eje_r, datos_pais_test.loc[pais[0]])
plt.plot(x_eje_p, prediction_rescaled[0])
plt.legend(("Real", "Pred"))
plt.show()
plt.close()