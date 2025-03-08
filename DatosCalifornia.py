import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos = pd.read_csv('SetDatosCalif.csv')

datos.info()
#print(datos.head())  # Imprime las primeras filas del DataFrame para verificar las columnas
#datos.head()

"""
print(datos["ocean_proximity"].value_counts())
print(datos.describe())
print(datos.hist(figsize=(15,8), bins=30, edgecolor='black'))
plt.show()"

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
sb.scatterplot(x="latitude", y="longitude", data=datos[(datos.median_income > 14)], hue="median_house_value", palette="coolwarm", size='population_normalized', sizes=(20, 200))
plt.show()

#para quitar los datos vacios, es decir los que tienen como valor NA
datos_na=datos.dropna()

print(datos_na.info())

