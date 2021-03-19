import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
d=2
h=1
a=0.1
I=1

#Axes
N=100

xmin=0
xmax=4
xx=np.linspace(xmin,xmax,N)

ymin=0
ymax=4
yy=np.linspace(ymin,ymax,N)

zmin=-2
zmax=2
zz=np.linspace(zmin,zmax,N)

X,Z=np.meshgrid(xx,zz)
XX,Y=np.meshgrid(xx,yy)
YY,ZZ=np.meshgrid(yy,zz)

#functions for distances
def R1(x,y,z):
    return np.sqrt((x-d)**2+(y-h)**2+z**2)

def R2(x,y,z):
    return np.sqrt((x-d)**2+(y+h)**2+z**2)

def R3(x,y,z):
    return np.sqrt((x+d)**2+(y-h)**2+z**2)

def R4(x,y,z):
    return np.sqrt((x+d)**2+(y+h)**2+z**2)


#functions for potential
def Ax(x,y,z):
    res=1/R1(x,y,z)**3-1/R2(x,y,z)**3+1/R3(x,y,z)**3-1/R4(x,y,z)**3
    return z*I*(a**2)*res/4

Ay=np.zeros((N,N))

def Az(x,y,z):
    res=-(x-d)/R1(x,y,z)**3+(x-d)/R2(x,y,z)**3-(x+d)/R3(x,y,z)**3+(x+d)/R4(x,y,z)**3
    return I*(a**2)*res/4

#functions for magnetic field
def Hx(x,y,z):
    res=(y-h)*(x-d)/R1(x,y,z)**5-(y+h)*(x-d)/R2(x,y,z)**5+(y-h)*(x+d)/R3(x,y,z)**5-(y+h)*(x+d)/R4(x,y,z)**5
    return 3*I*(a**2)*res/4

def Hy(x,y,z):
    Hy1=1/(R1(x,y,z)**3)*(3*(y-h)**2/R1(x,y,z)**2-1)
    Hy2=1/(R2(x,y,z)**3)*(-3*(y+h)**2/R2(x,y,z)**2+1)
    Hy3=1/(R3(x,y,z)**3)*(3*(y-h)**2/R3(x,y,z)**2-1)
    Hy4=1/(R4(x,y,z)**3)*(-3*(y+h)**2/R4(x,y,z)**2+1)
    return I*(a**2)/4*(Hy1+Hy2+Hy3+Hy4)

#functions for current density on yz plane
def Ky_yz(y,z):
    r1=R1(0,y,z)
    r2=R2(0,y,z)
    r3=R3(0,y,z)
    r4=R4(0,y,z)
    return I*(a**2)/4*3*(-(y-h)*z/r1**5+(y+h)*z/r2**5-(y-h)*z/r3**5+(y+h)*z/r4**5)

def Kz_yz(y,z):
    r1=R1(0,y,z)
    r2=R2(0,y,z)
    r3=R3(0,y,z)
    r4=R4(0,y,z)
    term1=3*(y-h)**2/r1**5-1/r1**3
    term2=-3*(y+h)**2/r2**5+1/r2**3
    term3=3*(y-h)**2/r3**5-1/r3**3
    term4=-3*(y+h)**2/r4**5+1/r4**3
    return I*(a**2)/4*(term1+term2+term3+term4)

#functions for current density on xz plane
def Kx_xz(x,z):
    r1=R1(x,0,z)
    r2=R2(x,0,z)
    r3=R3(x,0,z)
    r4=R4(x,0,z)
    return -I*(a**2)/4*3*h*z*(1/r1**5+1/r2**5+1/r3**5+1/r4**5)

def Kz_xz(x,z):
    r1=R1(x,0,z)
    r2=R2(x,0,z)
    r3=R3(x,0,z)
    r4=R4(x,0,z)
    return I*(a**2)/4*3*(h*(x-d)/r1**5+h*(x-d)/r2**5+h*(x+d)/r3**5+h*(x+d)/r4**5)


#PLOTS
#streamplot of magnetic potential on xz plane
fig1, ax1 = plt.subplots()
p1=ax1.streamplot(X,Z,Ax(X,1,Z),Az(X,1,Z),color=np.log10(np.sqrt(Ax(X,1,Z)**2+Az(X,1,Z)**2)),cmap=cm.jet)
ax1.set_aspect('equal','box')
c1=fig1.colorbar(p1.lines)
c1.set_label('$log_{10}$|A/$μ_0$|')
ax1.set_title('Normalised Magnetic Potential A/$μ_0$ on xz plane for y=1')
ax1.set_xlabel('x(m)')
ax1.set_ylabel('z(m)')

# #quiver plot of magnetic potential on xz plane
# #use N=30
# Ax_norm=Ax(X,1,Z)/np.sqrt(Ax(X,1,Z)**2+Az(X,1,Z)**2)
# Az_norm=Az(X,1,Z)/np.sqrt(Ax(X,1,Z)**2+Az(X,1,Z)**2)
#
# fig2, ax2 = plt.subplots()
# plt.quiver(X,Z,Ax_norm,Az_norm)
# ax2.set_aspect('equal','box')
# ax2.set_title('Normalised Magnetic Potential A/$μ_0$ on xz plane for y=1')
# ax2.set_xlabel('x(m)')
# ax2.set_ylabel('z(m)')

#streamplot of magnetic potential on xy plane
fig3, ax3 = plt.subplots()
p3=ax3.streamplot(XX,Y,Ax(XX,Y,2),Ay,color=np.log10(np.sqrt(Ax(XX,Y,2)**2+Ay**2)),cmap=cm.jet)
ax3.set_aspect('equal','box')
c3=fig3.colorbar(p3.lines)
c3.set_label('$log_{10}$|A/$μ_0$|')
ax3.set_title('Normalised Magnetic Potential A/$μ_0$ on xy plane for z=2')
ax3.set_xlabel('x(m)')
ax3.set_ylabel('y(m)')

# #quiver plot of magnetic potential on xz plane
# #use N=30
# Ax_norm2=Ax(XX,Y,2)/np.sqrt(Ax(XX,Y,2)**2+Ay**2)
# Ay_norm=Ay/np.sqrt(Ax(XX,Y,2)**2+Ay**2)
#
# fig4, ax4 = plt.subplots()
# plt.quiver(XX,Y,Ax_norm2,Ay)
# ax4.set_aspect('equal','box')
# ax4.set_title('Normalised Magnetic Potential A/$μ_0$ on xy plane for z=2')
# ax4.set_xlabel('x(m)')
# ax4.set_ylabel('y(m)')

#streamplot of magnetic field on xy plane
fig5, ax5 = plt.subplots()
p5=ax5.streamplot(XX,Y,Hx(XX,Y,0),Hy(XX,Y,0),color=np.log10(np.sqrt(Hx(XX,Y,0)**2+Hy(XX,Y,0)**2)),cmap=cm.jet,density=1.2)
ax5.set_aspect('equal','box')
c5=fig5.colorbar(p5.lines)
c5.set_label('$log_{10}$|H|')
ax5.set_title('Magnetic Field H on xy plane for z=0')
ax5.set_xlabel('x(m)')
ax5.set_ylabel('y(m)')

#streamplot of current density on yz plane
fig6, ax6 = plt.subplots()
p6=ax6.streamplot(YY,ZZ,Ky_yz(YY,ZZ),Kz_yz(YY,ZZ),color=np.log10(np.sqrt(Ky_yz(YY,ZZ)**2+Kz_yz(YY,ZZ)**2)),cmap=cm.jet,density=1.2)
ax6.set_aspect('equal','box')
c6=fig6.colorbar(p6.lines)
c6.set_label('$log_{10}$|K|')
ax6.set_title('Current density K on yz plane for x=0')
ax6.set_xlabel('y(m)')
ax6.set_ylabel('z(m)')

#streamplot of current density on xz plane
fig7, ax7 = plt.subplots()
p7=ax7.streamplot(X,Z,Kx_xz(X,Z),Kz_xz(X,Z),color=np.log10(np.sqrt(Kx_xz(X,Z)**2+Kz_xz(X,Z)**2)),cmap=cm.jet,density=1.2)
ax7.set_aspect('equal','box')
c7=fig7.colorbar(p7.lines)
c7.set_label('$log_{10}$|K|')
ax7.set_title('Current density K on xz plane for y=0')
ax7.set_xlabel('x(m)')
ax7.set_ylabel('z(m)')

plt.show()
