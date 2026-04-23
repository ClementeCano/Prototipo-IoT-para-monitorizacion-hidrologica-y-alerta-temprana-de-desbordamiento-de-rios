import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping


#Leo los datos
datos = pd.read_csv('Miranda_del_Ebro.csv')


#Limpio los datos: elimino las filas con datos faltantes, convierto a formato datetime la columna de fecha y ordeno los datos por fecha
datos = datos.dropna(axis=0)
datos['fecha'] = pd.to_datetime(datos['fecha'])
datos = datos.sort_values(by='fecha').reset_index(drop=True)

print(datos.head())
print("------------------------------")
print(datos.info())
print("------------------------------")
print(datos.isnull().sum())
print("------------------------------")


#Selecciono las variables
entradas = ["nivel_m", "caudal_m3s", "lluvia_mm"]
datos_entrada = datos[entradas].values


#Creo las ventanas de tiempo para el modelo de predicción
ventana = 7

def crear_ventanas_con_lluvia_futura(data, ventana=7):
    X_hist = []
    X_lluvia_futura = []
    y = []

    for i in range(len(data) - ventana):
        historial = data[i:i+ventana]          # Últimos 7 días
        lluvia_futura = data[i+ventana][2]     # Lluvia del día objetivo
        salida = data[i+ventana][:2]           # Nivel y caudal del día objetivo

        X_hist.append(historial)
        X_lluvia_futura.append([lluvia_futura])   # Lo guardo como vector de 1 valor
        y.append(salida)

    return np.array(X_hist), np.array(X_lluvia_futura), np.array(y)

X_hist, X_lluvia, y = crear_ventanas_con_lluvia_futura(datos_entrada, ventana=ventana)

print("Forma de X_hist:", X_hist.shape)
print("Forma de X_lluvia:", X_lluvia.shape)
print("Forma de y:", y.shape)

#Muestro un ejemplo de las ventanas creadas
print("\nEjemplo de ventana de entrada (X_hist):")
print(X_hist[0])                                             #Primera ventana de entrada (últimos 7 días)

print("\nEjemplo de lluvia futura asociada:")
print(X_lluvia[0])                                           #Lluvia del día que quiero predecir

print("\nEjemplo de salida correspondiente (y):")
print(y[0])                                                  #Predicción del nivel y caudal para el día siguiente


#Divido los datos en entrenamiento y prueba. Utilizo el 80% de los datos para entrenamiento y el 20% para prueba.
train_size = int(len(X_hist) * 0.8)

X_hist_train, X_hist_test = X_hist[:train_size], X_hist[train_size:]
X_lluvia_train, X_lluvia_test = X_lluvia[:train_size], X_lluvia[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("\nTamaño de entrenamiento:")
print("X_hist_train:", X_hist_train.shape)
print("X_lluvia_train:", X_lluvia_train.shape)
print("y_train:", y_train.shape)

print("\nTamaño de prueba:")
print("X_hist_test:", X_hist_test.shape)
print("X_lluvia_test:", X_lluvia_test.shape)
print("y_test:", y_test.shape)


#Normalizo y escalo los datos utilizando MinMaxScaler
scaler_hist = MinMaxScaler()
scaler_lluvia = MinMaxScaler()
scaler_y = MinMaxScaler()

#Para escalar los datos del historial, primero necesito convertir las matrices 3D de X_hist_train y X_hist_test a 2D,
#ya que MinMaxScaler espera una matriz 2D. Luego, después de escalar, puedo volver a darles la forma original.
X_hist_train_2d = X_hist_train.reshape(-1, X_hist_train.shape[2])
X_hist_test_2d = X_hist_test.reshape(-1, X_hist_test.shape[2])

#Ajusto el scaler del historial a los datos de entrenamiento y transformo tanto los datos de entrenamiento como los de prueba
X_hist_train_scaled_2d = scaler_hist.fit_transform(X_hist_train_2d)
X_hist_test_scaled_2d = scaler_hist.transform(X_hist_test_2d)

#Vuelvo a darles la forma original
X_hist_train_scaled = X_hist_train_scaled_2d.reshape(X_hist_train.shape)
X_hist_test_scaled = X_hist_test_scaled_2d.reshape(X_hist_test.shape)

#Escalo la lluvia futura utilizando otro scaler independiente
#Aquí no hace falta reshape porque X_lluvia_train y X_lluvia_test ya son matrices 2D de una sola columna
X_lluvia_train_scaled = scaler_lluvia.fit_transform(X_lluvia_train)
X_lluvia_test_scaled = scaler_lluvia.transform(X_lluvia_test)

#Escalo también las salidas (nivel y caudal)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

#Muestro ejemplos de los datos escalados
print("\nEjemplo de ventana de entrada escalada (X_hist_train_scaled):")
print(X_hist_train_scaled[0])                                #Primera ventana de entrada escalada

print("\nEjemplo de lluvia futura escalada:")
print(X_lluvia_train_scaled[0])                              #Lluvia futura escalada

print("\nEjemplo de salida escalada correspondiente (y_train_scaled):")
print(y_train_scaled[0])                                     #Predicción del nivel y caudal escalada para el día siguiente


#Creamos el modelo de predicción utilizando un modelo de red neuronal LSTM con dos entradas:
#1. El historial de los últimos 7 días
#2. La lluvia prevista para el día que quiero predecir
input_hist = Input(shape=(ventana, X_hist_train.shape[2]), name="historial")
x = LSTM(64)(input_hist)

input_lluvia = Input(shape=(1,), name="lluvia_futura")

#Concateno la salida del LSTM con la lluvia futura y paso el resultado por capas densas
x = Concatenate()([x, input_lluvia])
x = Dense(32, activation="relu")(x)
output = Dense(2)(x)

modelo = Model(inputs=[input_hist, input_lluvia], outputs=output)

modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
modelo.summary()


#Entrenamos el modelo utilizando EarlyStopping para evitar el sobreajuste y quedarme con el mejor modelo
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

#Entrenamos el modelo con los datos de entrenamiento y validación
modelo_entrenado = modelo.fit(
    [X_hist_train_scaled, X_lluvia_train_scaled],
    y_train_scaled,
    validation_data=([X_hist_test_scaled, X_lluvia_test_scaled], y_test_scaled),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


#Muestro la gráfica de evolución del error durante el entrenamiento
plt.figure(figsize=(10, 5))

plt.plot(modelo_entrenado.history['loss'], label='Loss entrenamiento')
plt.plot(modelo_entrenado.history['val_loss'], label='Loss validación')

plt.xlabel('Épocas')
plt.ylabel('Error')
plt.title('Evolución del error durante el entrenamiento')
plt.legend()

plt.show()


#Evaluamos el modelo en los datos de prueba
loss, mae = modelo.evaluate([X_hist_test_scaled, X_lluvia_test_scaled], y_test_scaled, verbose=0)

print(f"\nPérdida (loss) en el conjunto de prueba: {loss:.4f}")
print(f"Error absoluto medio (MAE) en el conjunto de prueba: {mae:.4f}")


#Hacemos predicciones con el modelo entrenado
y_pred_scaled = modelo.predict([X_hist_test_scaled, X_lluvia_test_scaled])

#Desescalamos las predicciones para obtener los valores originales de nivel y caudal
y_pred = scaler_y.inverse_transform(y_pred_scaled)

mae_nivel = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
mae_caudal = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

rmse_nivel = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
rmse_caudal = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))

print(f"\nMAE nivel: {mae_nivel:.4f} m")
print(f"MAE caudal: {mae_caudal:.4f} m3/s")
print(f"RMSE nivel: {rmse_nivel:.4f} m")
print(f"RMSE caudal: {rmse_caudal:.4f} m3/s")

print("\nPrimeras 5 predicciones vs. valores reales:")
for i in range(5):
    print(f"Predicción: {y_pred[i]} | Valor real: {y_test[i]}")

n = 100  # número de puntos a mostrar

plt.figure(figsize=(12,5))

# Nivel
plt.subplot(1,2,1)
plt.plot(y_test[:n, 0], label='Real', linewidth=2)
plt.plot(y_pred[:n, 0], label='Predicción', linestyle='--')
plt.title('Nivel del río (zoom)')
plt.xlabel('Tiempo')
plt.ylabel('Nivel (m)')
plt.legend()

# Caudal
plt.subplot(1,2,2)
plt.plot(y_test[:n, 1], label='Real', linewidth=2)
plt.plot(y_pred[:n, 1], label='Predicción', linestyle='--')
plt.title('Caudal del río (zoom)')
plt.xlabel('Tiempo')
plt.ylabel('Caudal (m³/s)')
plt.legend()

plt.tight_layout()
plt.show()


#Predigo el nivel y caudal del río para el día siguiente utilizando los últimos 7 días de datos
#y una lluvia prevista que indico manualmente
ultima_ventana = datos_entrada[-ventana:]   #Últimos 7 días de datos
lluvia_pred = 2.5                           #Lluvia prevista para mañana (ejemplo)

#Escalo la ventana histórica
ultima_ventana_scaled_2d = scaler_hist.transform(ultima_ventana)
ultima_ventana_scaled = ultima_ventana_scaled_2d.reshape(1, ventana, len(entradas))

#Escalo la lluvia prevista con el mismo scaler usado en entrenamiento
lluvia_prevista_scaled = scaler_lluvia.transform(np.array([[lluvia_pred]]))

#Hago la predicción escalada
pred_manana_scaled = modelo.predict([ultima_ventana_scaled, lluvia_prevista_scaled])

#Desescaleo la predicción para obtener los valores originales
pred_manana = scaler_y.inverse_transform(pred_manana_scaled)

print(f"\nPredicción para el día siguiente (nivel_m, caudal_m3s): {pred_manana[0]}")


#Muestro gráficamente la comparación entre valores reales y predichos en el conjunto de prueba
plt.figure(figsize=(12, 5))

#Nivel
plt.subplot(1, 2, 1)
plt.plot(y_test[:, 0], label='Real')
plt.plot(y_pred[:, 0], label='Predicción')
plt.title('Nivel del río')
plt.legend()

#Caudal
plt.subplot(1, 2, 2)
plt.plot(y_test[:, 1], label='Real')
plt.plot(y_pred[:, 1], label='Predicción')
plt.title('Caudal del río')
plt.legend()

plt.show()