import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
L=0.5
h=0.5
a=0.0025
e0=8.85*(10**(-12))

#Axes
N=25
xrmin=-L
xrmax=L
Dxr=(xrmax-xrmin)/N
Xr=np.linspace(xrmin,xrmax,N)
#xx=np.zeros(N)
#
# for i in range (0,N-1):
#     xx[i]=0.5*(Xr[i]+Xr[i+1])

N2=203
xmin=-2
xmax=2
X=np.linspace(xmin,xmax,N2)

#initialising arrays
AA=np.zeros((N,N))
# l=np.zeros(N)
# l0=np.zeros(N)

#calculating A matrix
for i in range(0,N):
    for j in range(0,N):
        if i==j:
            AA[i][j]=Dxr*(np.log((Dxr/2+np.sqrt(a**2+(Dxr/2)**2))/(-Dxr/2+np.sqrt(a**2+(Dxr/2)**2)))-2/(3*2*h))/(4*math.pi)
        else:
            R1=abs(Xr[j]-Xr[i])
            R2=np.sqrt((Xr[j]-Xr[i])**2+(2*h)**2)
            AA[i][j]=Dxr*(Dxr/R1 -2*Dxr/(3*R2))/(4*math.pi)

l0=np.dot(np.linalg.inv(AA),np.ones(N))
l=l0/(Dxr*sum(l0))

#plots
fig1, ax1 = plt.subplots()
test=ax1.plot(Xr,l)

plt.show()
