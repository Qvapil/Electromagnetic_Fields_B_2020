import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
L=0.5
h=0.5
levels=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5])

#Axes
xmin=-2
xmax=2
zmin=-2
zmax=2

N=403
dx=(xmax-xmin)/N
dz=(zmax-zmin)/N

xx=np.arange(xmin,xmax,dx)
zz=np.arange(zmin,zmax,dz)

X,Z=np.meshgrid(xx,zz)

#functions to calculate potential
def phi_term1(x,z):
    R1=np.sqrt((x-L)**2+(z-h)**2)
    R2=np.sqrt((x+L)**2+(z-h)**2)
    return np.log((x-L+R1)/(x+L+R2))

def phi_term2(x,z):
    R3=np.sqrt((x-L)**2+(z+h)**2)
    R4=np.sqrt((x+L)**2+(z+h)**2)
    return np.log((x-L+R3)/(x+L+R4))

def potential1(x,z):
    condlist=[z>=0,z<0]
    choicelist=[phi_term1(x,z)/(4*math.pi)-2*phi_term2(x,z)/(12*math.pi),phi_term1(x,z)/(12*math.pi)]
    return np.select(condlist,choicelist)

def potential2(x,z):
    condlist=[z>=0,z<0]
    choicelist=[phi_term1(x,z)/(20*math.pi)+2*phi_term2(x,z)/(60*math.pi),phi_term1(x,z)/(12*math.pi)]
    return np.select(condlist,choicelist)

phi1=potential1(X,Z)
phi2=potential2(X,Z)

#functions to calculate electric field
def electric_field_term1(x,z):
    R1=np.sqrt((x-L)**2+(z-h)**2)
    R2=np.sqrt((x+L)**2+(z-h)**2)
    return (x-L)/(R1*((z-h)**2))-(x+L)/(R2*((z-h)**2))

def electric_field_term2(x,z):
    R3=np.sqrt((x-L)**2+(z+h)**2)
    R4=np.sqrt((x+L)**2+(z+h)**2)
    return (x-L)/(R3*((z+h)**2))-(x+L)/(R4*((z+h)**2))

def Ex_func1(x,z):
    condlist=[z>=0,z<0]
    choicelist=[x*electric_field_term1(x,z)/(4*math.pi)-2*x*electric_field_term2(x,z)/(12*math.pi),x*electric_field_term1(x,z)/(12*math.pi)]
    return np.select(condlist,choicelist)

def Ex_func2(x,z):
    condlist=[z>=0,z<0]
    choicelist=[x*electric_field_term1(x,z)/(20*math.pi)+2*x*electric_field_term2(x,z)/(60*math.pi),x*electric_field_term1(x,z)/(12*math.pi)]
    return np.select(condlist,choicelist)

def Ez_func1(x,z):
    condlist=[z>=0,z<0]
    choicelist=[(z-h)*electric_field_term1(x,z)/(4*math.pi)-2*(z+h)*electric_field_term2(x,z)/(12*math.pi),(z-h)*electric_field_term1(x,z)/(12*math.pi)]
    return np.select(condlist,choicelist)

def Ez_func2(x,z):
    condlist=[z>=0,z<0]
    choicelist=[(z-h)*electric_field_term1(x,z)/(20*math.pi)+2*(z+h)*electric_field_term2(x,z)/(60*math.pi),(z-h)*electric_field_term1(x,z)/(12*math.pi)]
    return np.select(condlist,choicelist)

Ex1=Ex_func1(X,Z)
Ex2=Ex_func2(X,Z)
Ez1=Ez_func1(X,Z)
Ez2=Ez_func2(X,Z)

#PLOTS

#contour lines of potential
fig1, ax1 = plt.subplots()
CS1 = ax1.contour(X,Z,-phi1,levels)
ax1.clabel(CS1, inline=True, fontsize=7)
ax1.set_aspect('equal','box')
ax1.set_title('Normalised Electric Potential Φ(x,z)/(λ/$ε_0$) on xz plane\nfor $ε_1$=$ε_0$ and $ε_2$=5$ε_0$')
ax1.set_xlabel('x(m)')
ax1.set_ylabel('z(m)')

fig2, ax2 = plt.subplots()
CS2 = ax2.contour(X,Z,-phi2,levels)
ax2.clabel(CS2, inline=True, fontsize=7)
ax2.set_aspect('equal','box')
ax2.set_title('Normalised Electric Potential Φ(x,z)/(λ/$ε_0$) on xz plane\nfor $ε_1$=5$ε_0$ and $ε_2$=$ε_0$')
ax2.set_xlabel('x(m)')
ax2.set_ylabel('z(m)')

#surface plot of potential
fig3, ax3 = plt.subplots()
s3=ax3.pcolormesh(X,Z,-phi1,cmap=cm.jet)
ax3.set_aspect('equal','box')
ax3.set_title('Normalised Electric Potential Φ(x,z)/(λ/$ε_0$) on xz plane\nfor $ε_1$=$ε_0$ and $ε_2$=5$ε_0$')
cb3=fig3.colorbar(s3)
ax3.set_xlabel('x(m)')
ax3.set_ylabel('z(m)')

fig4, ax4 = plt.subplots()
s4=ax4.pcolormesh(X,Z,-phi2,cmap=cm.jet)
ax4.set_aspect('equal','box')
ax4.set_title('Normalised Electric Potential Φ(x,z)/(λ/$ε_0$) on xz plane\nfor $ε_1$=5$ε_0$ and $ε_2$=$ε_0$')
cb4=fig4.colorbar(s4)
ax4.set_xlabel('x(m)')
ax4.set_ylabel('z(m)')


# strpoints=np.array([[-0.51,-0.42,-0.3,-0.19,-0.08,0.001,0.001,0.08,0.19,0.3,0.42,0.51,-0.51,-0.42,-0.3,-0.19,-0.08,0.001,0.001,0.08,0.19,0.3,0.42,0.51],[0.53,0.55,0.55,0.57,0.57,0.57,0.55,0.51,0.55,0.55,0.53,0.53,0.47,0.47,0.47,0.45,0.45,0.43,0.43,0.43,0.47,0.47,0.47,0.47]])
fig5,ax5=plt.subplots()
CS5=ax5.contour(X,Z,-phi1,levels,cmap=cm.autumn)
sl5=ax5.streamplot(X,Z,-Ex1,-Ez1,color=np.log10(np.sqrt(Ex1**2+Ez1**2)),cmap=cm.gnuplot2,density=1.2)
ax5.set_aspect('equal','box')
ax5.set_title('Electric Field on xz plane\nfor $ε_1$=$ε_0$ and $ε_2$=5$ε_0$')
ax5.set_xlabel('y(m)')
ax5.set_ylabel('z(m)')

fig6,ax6=plt.subplots()
CS6=ax6.contour(X,Z,-phi2,levels,cmap=cm.autumn)
sl6=ax6.streamplot(X,Z,-Ex2,-Ez2,color=np.log10(np.sqrt(Ex1**2+Ez1**2)),cmap=cm.gnuplot2,density=1.2)
ax6.set_aspect('equal','box')
ax6.set_title('Electric Field on xz plane\nfor $ε_1$=5$ε_0$ and $ε_2$=$ε_0$')
ax6.set_xlabel('y(m)')
ax6.set_ylabel('z(m)')

plt.show()
