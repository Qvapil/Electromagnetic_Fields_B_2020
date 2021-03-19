import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

#constants
a=0.005
L=0.99
h=1
D=1.5
sigma0=1/160
I=500
N=5

#Axes for plot
xmin=-5
xmax=5
Nx=100
dx=(xmax-xmin)/Nx
X=np.linspace(xmin,xmax,Nx)

#length of pieces
Dl1=L/N*np.ones(N)
Dl2=D/N*np.ones(N)
Dl3=L/N*np.ones(N)
Dl=np.concatenate((Dl1,Dl2,Dl3))

#x coordinate of start of pieces (approximate, ignoring a)
x1=np.ones(N)*(-D/2)
x3=np.ones(N)*(D/2)
x2=np.linspace(-D/2,D/2,N+1)
x=np.concatenate((x1,x2,x3))

#x coordinate of center of pieces
xx1=np.zeros(N)
xx2=np.zeros(N)
xx3=np.zeros(N)

for i in range(0,N):
    xx1[i]=0.5*(x[i]+x[i+1])
    xx2[i]=0.5*(x[i+N]+x[i+1+N])
    xx3[i]=0.5*(x[i+2*N]+x[i+1+2*N])
xx=np.concatenate((xx1,xx2,xx3))

#z coordinate of start of pieces (approximate, ignoring a)
z1=np.linspace(-h-L,-h,N+1)
z3=np.linspace(-h,-h-L,N+1)
z2=np.ones(N-1)*(-h)
z=np.concatenate((z1,z2,z3))

#z coordinate of center of pieces
zz1=np.zeros(N)
zz2=np.zeros(N)
zz3=np.zeros(N)

for i in range(0,N):
    zz1[i]=0.5*(z[i]+z[i+1])
    zz2[i]=0.5*(z[i+N]+z[i+1+N])
    zz3[i]=0.5*(z[i+2*N]+z[i+1+2*N])
zz=np.concatenate((zz1,zz2,zz3))

#VDF matrix
VDF=np.zeros((3*N,3*N))

for i in range(0,3*N):
    for j in range(0,3*N):
        if i==j:
            R1=Dl[i]/2+np.sqrt(a**2+(Dl[i]/2)**2)
            R2=-Dl[i]/2+np.sqrt(a**2+(Dl[i]/2)**2)
            Riim=abs(2*zz[i])
            VDF[i][j]=(np.log(R1/R2)+Dl[i]/Riim)/(4*math.pi*sigma0)
        else:
            R1=np.sqrt((zz[i]-zz[j])**2+(xx[i]-xx[j])**2)
            R2=np.sqrt((zz[i]+zz[j])**2+(xx[i]-xx[j])**2)
            VDF[i][j]=Dl[j]*(1/R1+1/R2)/(4*math.pi*sigma0)

#charge at each piece
I0=np.dot(np.linalg.inv(VDF),np.ones(3*N))
Itot=0
for i in range (0, 3*N):
    Itot=Itot+I0[i]*Dl[i]

Ic=np.multiply(I0,I/Itot)

#potential of rod and ground resistance
phi_rod=I/Itot
Rg=phi_rod/I

#potential at ground level
def potential(x):
    res=0
    for i in range(0,3*N):
        R=np.sqrt((x-xx[i])**2+zz[i]**2)
        res=res+Ic[i]*Dl[i]*2/R
    return res/(4*math.pi*sigma0)

phi=potential(X)
phi0=potential(0)

print('For N={}:\nPotential at the rod: {}\nGround resistance: {}\nPotential at (0,0,0): {}'.format(N,phi_rod, Rg, phi0))

#plots
fig1, ax1 = plt.subplots()
potplot=ax1.plot(X,phi)
ax1.set_title('Electric potential Φ at ground level (y=0, z=0)\n for N={}'.format(N))
ax1.set_xlabel('x(m)')
ax1.set_ylabel('Φ')

fig2,ax2=plt.subplots()
iplot=ax2.plot(Ic)
ax2.set_title('Current $I_c$ at each piece of the conductor\nfor N={}'.format(N))
ax2.set_xlabel('Piece number (from bottom left to bottom right)')
ax2.set_ylabel('$I_c$')
plt.show()
