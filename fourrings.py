import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
a=0.1
d=1
h=1
e0=8.8*(10**(-12))
levels=np.array([0.01, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7.5])

#Axes
xmin=0
xmax=2*d
ymin=0
ymax=2*h
zmin=-2
zmax=2

N=200
dx=(xmax-xmin)/N
dy=(ymax-ymin)/N
dz=(zmax-zmin)/N

xx=np.arange(xmin,xmax,dx)
yy=np.arange(ymin,ymax,dy)
zz=np.arange(zmin,zmax,dz)

X,Y=np.meshgrid(xx,yy)
XX,Z=np.meshgrid(xx,zz)

#calculating potential at positive x,y
def potential_positive(x,y):
    min=0
    max=2*math.pi
    Ni=200
    di=(max-min)/Ni
    ii=np.arange(min,max,di)

    int1=0
    int2=0
    int3=0
    int4=0

    for i in ii:
        int1=int1+di/np.sqrt((x-d)**2+(y-h)**2+a**2-2*a*(x-d)*np.cos(i))
        int2=int2+di/np.sqrt((x-d)**2+(y+h)**2+a**2-2*a*(x-d)*np.cos(i))
        int3=int3+di/np.sqrt((x+d)**2+(y-h)**2+a**2-2*a*(x+d)*np.cos(i))
        int4=int4+di/np.sqrt((x+d)**2+(y+h)**2+a**2-2*a*(x+d)*np.cos(i))

    return a*int1-a*int2-a*int3+a*int4

#potential in xy plane
def potential(x,y):
    condlist=[np.logical_and(x>0,y>0),np.logical_or(x<=0,y<=0)]
    choicelist=[potential_positive(x,y),0]
    return np.select(condlist,choicelist)

phi=potential(X,Y)


#calculating electric field at positive x,y
def Ex_positive(x,y):
    min=0
    max=2*math.pi
    Ni=200
    di=(max-min)/Ni
    ii=np.arange(min,max,di)

    int1=0
    int2=0
    int3=0
    int4=0

    for i in ii:
        R1=np.sqrt((x-d)**2+(y-h)**2+a**2-2*a*(x-d)*np.cos(i))
        R2=np.sqrt((x-d)**2+(y+h)**2+a**2-2*a*(x-d)*np.cos(i))
        R3=np.sqrt((x+d)**2+(y-h)**2+a**2-2*a*(x+d)*np.cos(i))
        R4=np.sqrt((x+d)**2+(y+h)**2+a**2-2*a*(x+d)*np.cos(i))

        int1=int1+di*(x-d-a*np.cos(i))/R1**3
        int2=int2+di*(x-d-a*np.cos(i))/R2**3
        int3=int3+di*(x+d-a*np.cos(i))/R3**3
        int4=int4+di*(x+d-a*np.cos(i))/R4**3

    return a*int1-a*int2-a*int3+a*int4

def Ey_positive(x,y):
    min=0
    max=2*math.pi
    Ni=200
    di=(max-min)/Ni
    ii=np.arange(min,max,di)

    int1=0
    int2=0
    int3=0
    int4=0

    for i in ii:
        R1=np.sqrt((x-d)**2+(y-h)**2+a**2-2*a*(x-d)*np.cos(i))
        R2=np.sqrt((x-d)**2+(y+h)**2+a**2-2*a*(x-d)*np.cos(i))
        R3=np.sqrt((x+d)**2+(y-h)**2+a**2-2*a*(x+d)*np.cos(i))
        R4=np.sqrt((x+d)**2+(y+h)**2+a**2-2*a*(x+d)*np.cos(i))

        int1=int1+di*(y-h)/R1**3
        int2=int2+di*(y+h)/R2**3
        int3=int3+di*(y-h)/R3**3
        int4=int4+di*(y+h)/R4**3

    return a*int1-a*int2-a*int3+a*int4

#electric field
def Ex_func(x,y):
    condlist=[np.logical_and(x>0,y>0),np.logical_or(x<=0,y<=0)]
    choicelist=[Ex_positive(x,y),0]
    return np.select(condlist,choicelist)

def Ey_func(x,y):
    condlist=[np.logical_and(x>0,y>0),np.logical_or(x<=0,y<=0)]
    choicelist=[Ey_positive(x,y),0]
    return np.select(condlist,choicelist)

Ex=Ex_func(X,Y)
Ey=Ey_func(X,Y)

#calculating charge density
def density(x,z):
    min=0
    max=2*math.pi
    Ni=400
    di=(max-min)/Ni
    ii=np.arange(min,max,di)

    integ=0

    for i in ii:
        R1=np.sqrt((x-d)**2+h**2+z**2+a**2-2*a*((x-d)*np.cos(i)+z*np.sin(i)))
        R2=np.sqrt((x-d)**2+h**2+z**2+a**2-2*a*((x-d)*np.cos(i)+z*np.sin(i)))
        R3=np.sqrt((x+d)**2+h**2+z**2+a**2-2*a*((x+d)*np.cos(i)+z*np.sin(i)))
        R4=np.sqrt((x+d)**2+h**2+z**2+a**2-2*a*((x+d)*np.cos(i)+z*np.sin(i)))

        integ=integ+di*(-1/R1**3 -1/R2**3 +1/R3**3 +1/R4**3)

    return integ

sigma=density(XX,Z)



#potential contour lines
fig1, ax1 = plt.subplots()
CS = ax1.contour(X,Y,phi,levels)
ax1.clabel(CS, inline=True, fontsize=7)
ax1.set_aspect('equal','box')
ax1.set_title('Normalised Electric Potential Φ(x,y)/(λ/4*pi*e0) on xy plane')
ax1.set_xlabel('x(m)')
ax1.set_ylabel('y(m)')

#streamplot + coloured contour lines
fig2, ax2 = plt.subplots()
CS=ax2.contour(X,Y,phi,levels,cmap=cm.autumn)
q=ax2.streamplot(X,Y,Ex,Ey,density=2)
ax2.set_aspect('equal','box')
ax2.set_title('Electric Field in the xy plane')
ax2.set_xlabel('x(m)')
ax2.set_ylabel('y(m)')

#density contour lines
fig3, ax3 = plt.subplots()
levels2=np.array([-10.5,-9.5,-8,-6,-5,-4,-3,-2,-1,-0.7,-0.3,-0.1])
CS = ax3.contour(XX,Z,sigma,levels2,cmap=cm.copper)
ax3.set_aspect('equal','box')
ax3.clabel(CS, inline=True, fontsize=7)
ax3.set_title('Charge Density in xz plane')
ax3.set_xlabel('x(m)')
ax3.set_ylabel('z(m)')

plt.show()
