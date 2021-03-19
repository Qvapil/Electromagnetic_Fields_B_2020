import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
a=0.1
h=a/2
I1=1
I2=1
levels=[0.01,0.025,0.05,0.075,0.1,0.15,0.175,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.25,1.5,2,3,5]

#Axes
N=100

xmin=-3*a
xmax=3*a
xx=np.linspace(xmin,xmax,N)

zmin=-(h+2*a)
zmax=h+2*a
zz=np.linspace(zmin,zmax,N)

X,Z=np.meshgrid(xx,zz)

#function for Potential
def A_phi(x,y,z):
    phimin=0
    phimax=2*math.pi
    Nphi=200
    dphi=(phimax-phimin)/Nphi
    phi=np.linspace(phimin,phimax,Nphi)
    term1=0
    term2=0
    for i in phi:
        R1=np.sqrt(x**2+y**2+(z+h)**2+a**2-2*a*(x*np.cos(i)+y*np.sin(i)))
        R2=np.sqrt(x**2+y**2+(z-h)**2+a**2-2*a*(x*np.cos(i)+y*np.sin(i)))
        term1+=a*np.cos(i)*dphi/R1
        term2+=a*np.cos(i)*dphi/R2
    return I1*term1/(4*math.pi)+I2*term2/(4*math.pi)


#surface plot of potential
fig1,ax1=plt.subplots()
p1=ax1.pcolormesh(X,Z,abs(A_phi(X,0,Z)))

#contour lines of potential
fig2, ax2 = plt.subplots()
p2= ax2.contour(X,Z,abs(A_phi(X,0,Z)),levels)

plt.show()
