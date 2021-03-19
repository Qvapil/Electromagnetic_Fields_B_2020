import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
a=1
M0=1
levels=np.linspace(-0.9,0.9,num=18)

#Axes
N=100

xmin=-3*a
xmax=3*a
xx=np.linspace(xmin,xmax,N)

ymin=-3*a
ymax=3*a
yy=np.linspace(ymin,ymax,N)

X,Y=np.meshgrid(xx,yy)

# #function for Az
# def Az(x,y):
#     r=np.sqrt(x**2+y**2)
#     theta=np.arctan(y/x)
#     phimin=0
#     phimax=2*math.pi
#     Nphi=200
#     dphi=(phimax-phimin)/Nphi
#     phi=np.linspace(phimin,phimax,Nphi)
#     term1=0
#     for i in phi:
#         term1+=dphi*np.cos(i)*np.log(a/np.sqrt(r**2+a**2-2*r*a*np.cos(theta-i)))
#     return -term1

fig1,ax1=plt.subplots()
p1=ax1.pcolormesh(X,Y,Az(X,Y))
plt.show()
