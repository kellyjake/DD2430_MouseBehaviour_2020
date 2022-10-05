import numpy as np
import matplotlib.pyplot as plt

exp = lambda x,b,p : (x**p)*(b**x)

x = np.linspace(3,5,100)

for i in range(5,15):
    y = exp(x,i,8)
    y /= y[-1]
    plt.plot(x,y,label=i)

plt.legend()
plt.show()