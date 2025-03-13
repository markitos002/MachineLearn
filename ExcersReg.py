import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

datos = pd.read_csv('Datos/CasasCalifornia.csv')

datos.info()
#print(datos.head())  # Imprime las primeras filas del DataFrame para verificar las columnas
#datos.head()

"""
print(datos["ocean_proximity"].value_counts())
print(datos.describe())
print(datos.hist(figsize=(15,8), bins=30, edgecolor='black'))
plt.show()"

"""
"""
#codigo del curso
sb.scatterplot(x="latitude", y="longitude", data=datos, hue="median_house_value", palette="coolwarm" , size=datos["population"])
plt.show()


# Normalizar la columna population para que los valores sean adecuados para el parámetro size
datos['population_normalized'] = datos['population'] / datos['population'].max()

# Generar un gráfico de dispersión con la longitud y latitud, es decir, la ubicación de las casas
sb.scatterplot(x="latitude", y="longitude", data=datos, hue="median_house_value", palette="coolwarm", size='population_normalized', sizes=(20, 200))
plt.show()

# Para filtrar desde los datos
sb.scatterplot(x="latitude", y="longitude", data=datos[(datos.median_income > 14)], hue="median_house_value", palette="coolwarm")
plt.show()"

"""

#para quitar los datos vacios, es decir los que tienen como valor NA
datos_na=datos.dropna()

#convertir la columna catergorica a numerica
#dummies /One-hot encoding asigna un valor binario a cada categoría de la variable categórica
#y crea una nueva columna para cada categoría.
dummies = pd.get_dummies(datos_na['ocean_proximity'], dtype=int)

# Unir los datos originales con los datos dummies
datos_na = datos_na.join(dummies)

# Eliminar la columna original de ocean_proximity, pues ya no la necesitamos, axis es para eliminar columnas
datos_na = datos_na.drop(columns=['ocean_proximity'], axis=1)

print(datos_na.head())
print(datos_na.info())

#si deseamos guardar los datos en un archivo CSV
#datos_na.to_csv('Datos/SetDatosCalif.csv', index=False) # Guardar los datos en un archivo CSV

#analisis de correlacion
print(datos_na.corr())

#en este analisis de correlacion es solo para la columna median_house_value, sort values es para ordenar los valores de forma descendente
print(datos_na.corr()['median_house_value'].sort_values(ascending=False))

#sb.heatmap(datos_na.corr(), annot=True, cmap='coolwarm')
#plt.show()

sb.scatterplot(x=datos_na["median_house_value"], y=datos_na["median_income"])
plt.show()

#agregar nueva caracteristica
datos_na["bedrooms_ratio"] = datos_na["total_bedrooms"] / datos_na["total_rooms"]
#grafica cuadro de correlacion con la nueva caracteristica
sb.heatmap(datos_na.corr(), annot=True, cmap='coolwarm')
plt.show()

#separar las caracteristicas de la etiqueta
X = datos_na.drop(columns=['median_house_value'], axis=1)
Y = datos_na['median_house_value']

#separar los datos en dos partes 1 de entrenamiento y 1 prueba
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.linear_model import LinearRegression

#crear el modelo
modelo = LinearRegression()

#entrenar el modelo
modelo.fit(X_train, Y_train)

#prediccion
prediccion = modelo.predict(X_test)
print(prediccion)
#persentar mejor los datos con pandas
print(pd.DataFrame(prediccion))

#vamos a comparar los datos de prediccion con los datos de prueba
print(pd.DataFrame({'Real': Y_test, 'Prediccion': prediccion}))



#calcular la precision del modelo
#precision = modelo.score(X_test, Y_test)
#print(precision)


#procedimiento para mejorar el modelo "sobreajuste" - overfitting
print(modelo.score(X_train, Y_train))
print(modelo.score(X_test, Y_test))

#error
from sklearn.metrics import mean_squared_error
error = mean_squared_error(Y_test, prediccion)
print(error)
print(np.sqrt(error)) #raiz cuadrada del error

#uso de scaler para comprimir los datos, para que el modelo sea mas preciso al evaluar el modelo sin tener en cuenta la magnitud de los datos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = LinearRegression()
modelo.fit(X_train, Y_train)

prediccion = modelo.predict(X_test)
print(modelo.score(X_test, Y_test))
print(np.sqrt(mean_squared_error(Y_test, prediccion)))