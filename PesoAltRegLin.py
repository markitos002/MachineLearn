import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos = pd.read_csv('Datos/datosAlt.csv')

datos.info()
print(datos.head())  # Imprime las primeras filas del DataFrame para verificar las columnas
#datos.head()

sb.scatterplot(x="peso", y="altura", data=datos, hue="altura", palette="coolwarm")

plt.show()

X = datos['peso']
Y = datos['altura']

X_procesada = X.values.reshape(-1,1)
Y_procesada = Y.values.reshape(-1,1)

modelo = LinearRegression()
modelo.fit(X_procesada,Y_procesada) #entrenamiento del modelo, aca le entrego los datos de entrenamiento
#print(modelo.coef_) #pendiente de la recta 

#prediccion
prediccion = modelo.predict([[1.23]])
print(prediccion)

modelo.score(X_procesada,Y_procesada) #calculo de la precision del modelo
print(modelo.score(X_procesada,Y_procesada))

altura = 1.72
prediccion = modelo.predict([[altura]])
print(f" {altura} es equivalente a {prediccion} kilos")