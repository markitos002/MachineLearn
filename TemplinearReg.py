import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos = pd.read_csv('Datos/datosTemp.csv')

datos.info()
print(datos.head())  # Imprime las primeras filas del DataFrame para verificar las columnas
#datos.head()

# Crear el scatterplot
sb.scatterplot(x="celsius", y="fahrenheit", data=datos, hue="fahrenheit", palette="coolwarm")

# Mostrar el gráfico
plt.show()

#caracteristicas de los datos (X) y etiquetas (Y)
X = datos['celsius']
Y = datos['fahrenheit']

print(X)
print(Y)

#X.values.reshape(-1,1) # esta transformacion es necesaria para que el modelo de regresion lineal pueda trabajar con los datos
#Y.values.reshape(-1,1)

X_procesada = X.values.reshape(-1,1)
Y_procesada = Y.values.reshape(-1,1)

modelo = LinearRegression()
modelo.fit(X_procesada,Y_procesada) #entrenamiento del modelo, aca le entrego los datos de entrenamiento
#print(modelo.coef_) #pendiente de la recta

#prediccion
prediccion = modelo.predict([[123]])
print(prediccion)

modelo.score(X_procesada,Y_procesada) #calculo de la precision del modelo
print(modelo.score(X_procesada,Y_procesada))

celsius = 258
prediccion = modelo.predict([[celsius]])
print(f" {celsius} grados celsius son equivalentes a {prediccion} grados fahrenheit")
