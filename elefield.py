import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
a=10
b=5
d=3
L=3
D=2.5
V0=1
e0=8.8*(10**(-12))
levels=np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 2, 3, 5])

#Axes
ymin=-15
ymax=15
zmin=-15
zmax=15

N=400
dy=(ymax-ymin)/N
dz=(zmax-zmin)/N

yy=np.arange(ymin,ymax,dy)
zz=np.arange(zmin,zmax,dz)

Y,Z=np.meshgrid(yy,zz)

#utility function to calculate potential inside the cavity
def inside_cavity(y,z):
    thetamin=-L/d
    thetamax=L/d
    Ni=200
    dtheta=(thetamax-thetamin)/Ni
    theta=np.arange(thetamin,thetamax,dtheta)
    int1=0
    int2=0
    dd=(b**2)/d
    for i in theta:
        int1=int1+dtheta/np.sqrt((y-d*np.sin(i))**2+(z-d*np.cos(i)-D)**2)
        int2=int2+dtheta/np.sqrt((y-dd*np.sin(i))**2+(z-dd*np.cos(i)-D)**2)
    return d*int1-b*int2+V0

#electric potential function
#since I am mapping just the yz plane I'm keeping x=0 for code simplicity
def potential(y,z):
    r=np.sqrt(y**2+z**2)
    rT=np.sqrt(y**2+(z-D)**2)
    condlist=[rT<=b,np.logical_and(r<=a,rT>b),r>a]
    choicelist=[inside_cavity(y,z), V0, V0*a/r]
    return np.select(condlist,choicelist)

phi=potential(Y, Z)

#utility funtions to calculate the electric field inside the cavity
def Ey_inside_cavity(y,z):
    thetamin=-L/d
    thetamax=L/d
    Ni=200
    dtheta=(thetamax-thetamin)/Ni
    theta=np.arange(thetamin,thetamax,dtheta)
    int1=0
    int2=0
    dd=(b**2)/d
    for i in theta:
        int1=int1+dtheta*(y-d*np.sin(i))/(np.sqrt((y-d*np.sin(i))**2+(z-d*np.cos(i)-D)**2))**3
        int2=int2+dtheta*(y-dd*np.sin(i))/(np.sqrt((y-dd*np.sin(i))**2+(z-dd*np.cos(i)-D)**2))**3
    return d*int1-b*int2

def Ez_inside_cavity(y,z):
    thetamin=-L/d
    thetamax=L/d
    Ni=200
    dtheta=(thetamax-thetamin)/Ni
    theta=np.arange(thetamin,thetamax,dtheta)
    int1=0
    int2=0
    dd=(b**2)/d
    for i in theta:
        int1=int1+dtheta*(z-d*np.cos(i)-D)/(np.sqrt((y-d*np.sin(i))**2+(z-d*np.cos(i)-D)**2))**3
        int2=int2+dtheta*(z-dd*np.cos(i)-D)/(np.sqrt((y-dd*np.sin(i))**2+(z-dd*np.cos(i)-D)**2))**3
    return d*int1-b*int2

#functions for electric field
def Ey_func(y,z):
    r=np.sqrt(y**2+z**2)
    rT=np.sqrt(y**2+(z-D)**2)
    condlist=[rT<b,np.logical_and(r<=a,rT>=b),r>a]
    choicelist=[Ey_inside_cavity(y,z), 0, V0*a*y/((r)**3)]
    return np.select(condlist,choicelist)

def Ez_func(y,z):
    r=np.sqrt(y**2+z**2)
    rT=np.sqrt(y**2+(z-D)**2)
    condlist=[rT<b,np.logical_and(r<=a,rT>=b),r>a]
    choicelist=[Ez_inside_cavity(y,z), 0, V0*a*z/((r)**3)]
    return np.select(condlist,choicelist)


Ey=Ey_func(Y,Z)
Ez=Ez_func(Y,Z)


def density(i):
    y=b*np.sin(i)
    z=b*np.cos(i)+D
    rb=np.sqrt(y**2+(z-D)**2)

    return -e0*(y*Ey_func(y,z)+(z-D)*Ez_func(y,z))/rb



th=np.linspace(0,2*math.pi,10000)
sigma=density(th)


#contour lines of potential
fig1, ax1 = plt.subplots()
CS = ax1.contour(Y,Z,phi,levels)
ax1.clabel(CS, inline=True, fontsize=7)
ax1.set_aspect('equal','box')
ax1.set_title('Normalised Electric Potential Φ(y,z)/(λ/4*pi*e0) on yz plane')
ax1.set_xlabel('y(m)')
ax1.set_ylabel('z(m)')

#surface plot of potential
fig2,ax2=plt.subplots()
ss=ax2.pcolormesh(Y,Z,phi,cmap=cm.jet)
ax2.set_aspect('equal','box')
ax2.set_title('Normalised Electric Potential Φ(y,z)/(λ/4*pi*e0)')
cb=fig2.colorbar(ss)
ax2.set_xlabel('y(m)')
ax2.set_ylabel('z(m)')

#streamplot of field + light coloured contour lines
fig3,ax3=plt.subplots()
CS=ax3.contour(Y,Z,phi,levels,cmap=cm.Reds)
q=ax3.streamplot(Y,Z,Ey,Ez,density=2.5)
ax3.set_aspect('equal','box')
ax3.set_title('Electric Field in the yz plane')
ax3.set_xlabel('y(m)')
ax3.set_ylabel('z(m)')

#plot of charge density
fig4, ax4 = plt.subplots()
ax4.plot(th,sigma,'.')
ax4.set_title('Charge density of cavity border in yz plane')
ax4.set_xlabel('Polar angle theta (radians)')
ax4.set_ylabel('Charge density sigma')

plt.show()
