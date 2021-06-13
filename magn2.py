import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
from scipy.special import ellipk
from scipy.special import ellipe

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

amax=0.25
aa=np.linspace(0,amax,N)

#function for Hz for x,y=0
def Hz0(z,h):
    R1=np.sqrt(a**2+(z+h)**2)
    R2=np.sqrt(a**2+(z-h)**2)
    return a**2/2*(I1/R1**3+I2/R2**3)

#first derivative of Hz
def Hz1(z,h):
    R1=np.sqrt(a**2+(z+h)**2)
    R2=np.sqrt(a**2+(z-h)**2)
    term1=-3*a**2*(z+h)/R1**5
    term2=-3*a**2*(z-h)/R2**5
    return I1/2*term1+I2/2*term2

#second derivative of Hz
def Hz2(z,h):
    R1=np.sqrt(a**2+(z+h)**2)
    R2=np.sqrt(a**2+(z-h)**2)
    term1=3*a**2*(4*z**2+8*h*z+4*h**2-a**2)/R1**7
    term2=3*a**2*(4*z**2-8*h*z+4*h**2-a**2)/R2**7
    return I1/2*term1+I2/2*term2

#third derivative of Hz
def Hz3(z,h):
    R1=np.sqrt(a**2+(z+h)**2)
    R2=np.sqrt(a**2+(z-h)**2)
    term1=-15*a**2*(z+h)*(4*z**2+8*h*z+4*h**2-3*a**2)/R1**9
    term2=-15*a**2*(z-h)*(4*z**2-8*h*z+4*h**2-3*a**2)/R2**9
    return I1/2*term1+I2/2*term2

#fourth derivative of Hz
def Hz4(z,h):
    R1=np.sqrt(a**2+(z+h)**2)
    R2=np.sqrt(a**2+(z-h)**2)
    term1=45*a**2*(8*z**4+32*h*z**3+(48*h**2-12*a**2)*z**2-(24*a**2*h-32*h**3)*z+8*h**4-12*a**2*h**2+a**4)/R1**9
    term2=45*a**2*(8*z**4-32*h*z**3+(48*h**2-12*a**2)*z**2+(24*a**2*h-32*h**3)*z+8*h**4-12*a**2*h**2+a**4)/R2**9
    return I1/2*term1+I2/2*term2

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

#functions for Magnetic Field
def Hx(x,y,z,h):
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
        term1+=(z+h)*a*np.cos(i)*dphi/R1**3
        term2+=(z-h)*a*np.cos(i)*dphi/R2**3
    return I1*term1/(4*math.pi)+I2*term2/(4*math.pi)

def Hz(x,y,z,h):
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
        term1+=(-x*np.cos(i)-y*np.sin(i)+a)*a*dphi/R1**3
        term2+=(-x*np.cos(i)-y*np.sin(i)+a)*a*dphi/R2**3
    return I1*term1/(4*math.pi)+I2*term2/(4*math.pi)

#function for L12
# def L12(a,h):
#     phimin=0
#     phimax=2*math.pi
#     Nphi=200
#     dphi=(phimax-phimin)/Nphi
#     phi=np.linspace(phimin,phimax,Nphi)
#     res=0
#     for i in phi:
#         for j in phi:
#             res+=np.cos(i-j)*dphi**2/np.sqrt(2*a**2+h**2-2*a**2*np.cos(i-j))
#     return a**2*res/(4*math.pi)

def L12(a,h):
    k=4*a**2/(h**2+4*a**2)
    return a*((2-k**2)/k*ellipk(k)-2/k*ellipe(k))


#PLOTS
#plot of Hz for h=a/4
hh=a/4
z1=np.linspace(-2*hh,2*hh,N)
fig11,ax11=plt.subplots()
p11=ax11.plot(z1,Hz0(z1,hh))
ax11.set_title('Magnetic field $H_z$ on z axis for h=a/4')
ax11.set_xlabel('z(m)')
ax11.set_ylabel('$H_z$')

#plot of 1st derivative of Hz for h=a/4
fig12,ax12=plt.subplots()
p12=ax12.plot(z1,Hz1(z1,hh))
ax12.set_title('1st derivative of Magnetic field $H_z$ on z axis for h=a/4')
ax12.set_xlabel('z(m)')

#plot of 2nd derivative of Hz for h=a/4
fig13,ax13=plt.subplots()
p13=ax13.plot(z1,Hz2(z1,hh))
ax13.set_title('2nd derivative of Magnetic field $H_z$ on z axis for h=a/4')
ax13.set_xlabel('z(m)')

#plot of 3rd derivative of Hz for h=a/4
fig14,ax14=plt.subplots()
p14=ax14.plot(z1,Hz3(z1,hh))
ax14.set_title('3rd derivative of Magnetic field $H_z$ on z axis for h=a/4')
ax14.set_xlabel('z(m)')

#plot of 4th derivative of Hz for h=a/4
fig15,ax15=plt.subplots()
p15=ax15.plot(z1,Hz4(z1,hh))
ax15.set_title('4th derivative of Magnetic field $H_z$ on z axis for h=a/4')
ax15.set_xlabel('z(m)')

#plot of Hz for h=a/2
hh=a/2
z1=np.linspace(-2*hh,2*hh,N)
fig21,ax21=plt.subplots()
p21=ax21.plot(z1,Hz0(z1,hh))
ax21.set_title('Magnetic field $H_z$ on z axis for h=a/2')
ax21.set_xlabel('z(m)')
ax21.set_ylabel('$H_z$')

#plot of 1st derivative of Hz for h=a/2
fig22,ax22=plt.subplots()
p22=ax22.plot(z1,Hz1(z1,hh))
ax22.set_title('1st derivative of Magnetic field $H_z$ on z axis for h=a/2')
ax22.set_xlabel('z(m)')

#plot of 2nd derivative of Hz for h=a/2
fig23,ax23=plt.subplots()
p23=ax23.plot(z1,Hz2(z1,hh))
ax23.set_title('2nd derivative of Magnetic field $H_z$ on z axis for h=a/2')
ax23.set_xlabel('z(m)')

#plot of 3rd derivative of Hz for h=a/4
fig24,ax24=plt.subplots()
p24=ax24.plot(z1,Hz3(z1,hh))
ax24.set_title('3rd derivative of Magnetic field $H_z$ on z axis for h=a/2')
ax24.set_xlabel('z(m)')

#plot of 4th derivative of Hz for h=a/2
fig25,ax25=plt.subplots()
p25=ax25.plot(z1,Hz4(z1,hh))
ax25.set_title('4th derivative of Magnetic field $H_z$ on z axis for h=a/2')
ax25.set_xlabel('z(m)')

#plot of Hz for h=a
hh=a
z1=np.linspace(-2*hh,2*hh,N)
fig31,ax31=plt.subplots()
p31=ax31.plot(z1,Hz0(z1,hh))
ax31.set_title('Magnetic field $H_z$ on z axis for h=a')
ax31.set_xlabel('z(m)')
ax31.set_ylabel('$H_z$')

#plot of 1st derivative of Hz for h=a
fig32,ax32=plt.subplots()
p32=ax32.plot(z1,Hz1(z1,hh))
ax32.set_title('1st derivative of Magnetic field $H_z$ on z axis for h=a')
ax32.set_xlabel('z(m)')

#plot of 2nd derivative of Hz for h=a
fig33,ax33=plt.subplots()
p33=ax33.plot(z1,Hz2(z1,hh))
ax33.set_title('2nd derivative of Magnetic field $H_z$ on z axis for h=a')
ax33.set_xlabel('z(m)')

#plot of 3rd derivative of Hz for h=a
fig34,ax34=plt.subplots()
p34=ax34.plot(z1,Hz3(z1,hh))
ax34.set_title('3rd derivative of Magnetic field $H_z$ on z axis for h=a')
ax34.set_xlabel('z(m)')

#plot of 4th derivative of Hz for h=a
fig35,ax35=plt.subplots()
p35=ax35.plot(z1,Hz4(z1,hh))
ax35.set_title('4th derivative of Magnetic field $H_z$ on z axis for h=a')
ax35.set_xlabel('z(m)')

#surface plot of potential
fig4,ax4=plt.subplots()
p4=ax4.pcolormesh(X,Z,abs(A_phi(X,0,Z)))
ax4.set_aspect('equal','box')
ax4.set_title('Normalised Magnetic Potential A/$μ_0$ on xz plane for y=0')
ax4.set_xlabel('x(m)')
ax4.set_ylabel('z(m)')

#contour lines of potential
fig5,ax5=plt.subplots()
p5=ax5.contour(X,Z,abs(A_phi(X,0,Z)),levels)
ax5.clabel(p5, inline=True, fontsize=7)
ax5.set_aspect('equal','box')
ax5.set_title('Normalised Magnetic Potential A/$μ_0$ on xz plane for y=0')
ax5.set_xlabel('x(m)')
ax5.set_ylabel('z(m)')

#streamplot of magnetic field for h=a/4
fig61, ax61 = plt.subplots()
p61=ax61.streamplot(X,Z,Hx(X,0,Z,a/4),Hz(X,0,Z,a/4),density=1.2)
ax61.set_aspect('equal','box')
ax61.set_title('Magnetic Field H on xz plane for y=0\nh=a/4')
ax61.set_xlabel('x(m)')
ax61.set_ylabel('z(m)')

#streamplot of magnetic field for h=a/2
fig62, ax62 = plt.subplots()
p62=ax62.streamplot(X,Z,Hx(X,0,Z,a/2),Hz(X,0,Z,a/2),density=1.2)
ax62.set_aspect('equal','box')
ax62.set_title('Magnetic Field H on xz plane for y=0\nh=a/2')
ax62.set_xlabel('x(m)')
ax62.set_ylabel('z(m)')

#streamplot of magnetic field for h=a
fig63, ax63 = plt.subplots()
p63=ax63.streamplot(X,Z,Hx(X,0,Z,a),Hz(X,0,Z,a),density=1.2)
ax63.set_aspect('equal','box')
ax63.set_title('Magnetic Field H on xz plane for y=0\nh=a')
ax63.set_xlabel('x(m)')
ax63.set_ylabel('z(m)')

#plot of L12 for h=a/4
fig71,ax71=plt.subplots()
p71=ax71.plot(aa,L12(aa,a/4))
ax71.set_title('Normalised L12/$μ_0$ for h=a/4')
ax71.set_xlabel('a(m)')
ax71.set_ylabel('L12')

#plot of L12 for h=a/2
fig72,ax72=plt.subplots()
p72=ax72.plot(aa,L12(aa,a/2))
ax72.set_title('Normalised L12/$μ_0$ for h=a/2')
ax72.set_xlabel('a(m)')
ax72.set_ylabel('L12')

#plot of L12 for h=a
fig73,ax73=plt.subplots()
p73=ax73.plot(aa,L12(aa,a))
ax73.set_title('Normalised L12/$μ_0$ for h=a')
ax73.set_xlabel('a(m)')
ax73.set_ylabel('L12')

plt.show()
