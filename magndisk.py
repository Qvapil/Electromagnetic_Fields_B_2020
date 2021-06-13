import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
a=1
M0=1
levels=2.7*np.linspace(-0.9,0.9,num=18)

#Axes
N=100

xmin=-3*a
xmax=3*a
xx=np.linspace(xmin,xmax,N)

ymin=-3*a
ymax=3*a
yy=np.linspace(ymin,ymax,N)

X,Y=np.meshgrid(xx,yy)

#function for Az
def Az(x,y):
    r=np.sqrt(x**2+y**2)
    theta=np.arctan2(y,x)
    phimin=0
    phimax=2*math.pi
    Nphi=200
    dphi=(phimax-phimin)/Nphi
    phi=np.linspace(phimin,phimax,Nphi)
    res=0
    for i in phi:
        res+=dphi*np.cos(i)*np.log(a/np.sqrt(r**2+a**2-2*r*a*np.cos(theta-i)))
    return -a*res

#functions for B
def Bx(x,y):
    r=np.sqrt(x**2+y**2)
    theta=np.arctan2(y,x)
    phimin=0
    phimax=2*math.pi
    Nphi=200
    dphi=(phimax-phimin)/Nphi
    phi=np.linspace(phimin,phimax,Nphi)
    res=0
    for i in phi:
        res+=dphi*np.cos(i)*(y-a*np.sin(i))/(r**2+a**2-2*r*a*np.cos(theta-i))
    return a*M0*res/(2*math.pi)

def By(x,y):
    r=np.sqrt(x**2+y**2)
    theta=np.arctan2(y,x)
    phimin=0
    phimax=2*math.pi
    Nphi=200
    dphi=(phimax-phimin)/Nphi
    phi=np.linspace(phimin,phimax,Nphi)
    res=0
    for i in phi:
        res+=dphi*np.cos(i)*(x-a*np.cos(i))/(r**2+a**2-2*r*a*np.cos(theta-i))
    return -a*M0*res/(2*math.pi)

#functions for H
def Hx(x,y):
    return Bx(x,y)

def Hy(x,y):
    r=np.sqrt(x**2+y**2)
    condlist=[r<a,r>a]
    choicelist=[By(x,y)-M0,By(x,y)]
    return np.select(condlist,choicelist)

#PLOTS
#surface plot of Az
fig1,ax1=plt.subplots()
p1=ax1.pcolormesh(X,Y,Az(X,Y))
ax1.set_aspect('equal','box')
ax1.set_title('Normalised Magnetic Potential $A_z$(y,z)/($M_0μ_0$/2π)')
cb1=fig1.colorbar(p1)
ax1.set_xlabel('x(m)')
ax1.set_ylabel('y(m)')

#contour lines of Az
fig2,ax2=plt.subplots()
p2=ax2.contour(X,Y,Az(X,Y),levels)
ax2.clabel(p2, inline=True, fontsize=7)
c2=plt.Circle((0,0),a,fill=False)
ax2.add_artist(c2)
ax2.set_aspect('equal','box')
ax2.set_title('Normalised Magnetic Potential $A_z$(x,y)/($M_0μ_0$/2π)')
ax2.set_xlabel('x(m)')
ax2.set_ylabel('y(m)')

#streamplot of B
fig3,ax3=plt.subplots()
p3=ax3.streamplot(X,Y,Bx(X,Y),By(X,Y),density=1.2,color=np.log10(np.sqrt(Bx(X,Y)**2+By(X,Y)**2)),cmap=cm.gist_heat)
c3=plt.Circle((0,0),a,fill=False)
ax3.add_artist(c3)
ax3.set_aspect('equal','box')
ax3.set_title('Normalised Magnetic Induction B(x,y)/$μ_0$')
ax3.set_xlabel('x(m)')
ax3.set_ylabel('y(m)')

#streamplot of H
fig4,ax4=plt.subplots()
p4=ax4.streamplot(X,Y,Hx(X,Y),Hy(X,Y),density=1.2,color=np.log10(np.sqrt(Hx(X,Y)**2+Hy(X,Y)**2)),cmap=cm.gist_heat)
c4=plt.Circle((0,0),a,fill=False)
ax4.add_artist(c4)
ax4.set_aspect('equal','box')
ax4.set_title('Magnetic Field H(x,y)')
ax4.set_xlabel('x(m)')
ax4.set_ylabel('y(m)')

plt.show()
