import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
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

sb.heatmap(datos_na.corr(), annot=True, cmap='coolwarm')
plt.show()

