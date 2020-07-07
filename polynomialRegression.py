import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("polynomial_regression.csv",sep=";")
print(df)

y=df.araba_max_hiz.values.reshape(-1,1)
x=df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hiz")
plt.show()

#linear regression => y=b0+b1*x
#multiple linear regression=> b0+b1*x1+b2*x2

#%% linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#değerlerimizi fit edelim
lr.fit(x,y)
#%% predict
"""
Burada grafiğe baktığımız zaman pek verimli görünmediğini
fark edeceğiz.Bunun sebebi polynoimal yöntemi kullanmadık
"""
y_head=lr.predict(x)
plt.plot(x,y_head,color="red",label="linear")
plt.show()
print("10 milyon tl lik araba hız tahmini:",lr.predict([[10000]]))

#polynoimal regression => y=b0+b1*x+b2*x^2+b3*x^3
from sklearn.preprocessing import PolynomialFeatures
#degree değerini ne kadar yüksek veririsek o kadar verim alırız
#yani degree arttıkça hata payı azalır
polynoimal_regression=PolynomialFeatures(degree=4)
x_polyniomal=polynoimal_regression.fit_transform(x)

#%% fit
linear_regression2=LinearRegression()
linear_regression2.fit(x_polyniomal,y)


y_head2=linear_regression2.predict(x_polyniomal)
plt.plot(x,y_head2,color="black",label="poly")
plt.legend()
plt.show()
