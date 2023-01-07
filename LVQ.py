# Paqueterias
from skimage import color , io
import numpy as np 
import numpy.matlib
import matplotlib.pyplot as plt


# Main 
nfil = 60
ncol = 100
ncapas = 3

som = np.random.rand(nfil,ncol,ncapas)
plt.ion()
plt.figure()
plt.subplot(1,2,1)
plt.imshow(som)

#Generacion de patrones
pat = np.random.rand(100,ncapas)

# Creacion del mallado para visualizacion y variables 
x = np.linspace(0, 100, ncol)
y = np.linspace(0, 100, nfil)
x,y = np.meshgrid(x,y)
epochs = 5
alpha0 = 0.75
sgm0 = 20
decay = 0.05

for t in range(3):
    alpha = alpha0*np.exp(-t*decay)
    sgm = sgm0*np.exp(-t*decay)
    ven = np.ceil(sgm*3)
    
    # SOM
    for i in range(100): #Son 100 porque son 100 patrones
    
        # Calcula las distancias a cada neurona para saber a cual se acerca mas
        vector = pat[i]
        columna = som.reshape(nfil*ncol,3)
        d = 0
        for n in range(3):
            d += ((vector[n]-columna[:,n])**2)
        dist = np.sqrt(d)
        
        ind = np.argmin(dist)
        bmfil, bmcol = np.unravel_index(ind, [nfil,ncol])
        g = np.exp(-(((x-bmcol)**2)+((y-bmfil)**2))/(2*sgm*sgm))
        
        ffil = int(np.max([0, bmfil-ven]))
        tfil = int(np.min([bmfil+ven, nfil]))
        fcol = int(np.max([0, bmcol-ven]))
        tcol = int(np.min([bmcol+ven, ncol]))
        
        vecindad = som[ffil:tfil, fcol:tcol]
        T = np.ones(np.shape(vecindad))
        G = np.ones(np.shape(vecindad))
        for e in range(3):
            T[:,:,e] *= vector[e]
            G[:,:,e] = g[ffil:tfil,fcol:tcol]
        
        vecindad += (alpha*G*(T-vecindad))
        som[ffil:tfil, fcol:tcol, :] = vecindad
        
        plt.subplot(1,2,2)
        plt.imshow(som)
        plt.pause(0.5)
        plt.show()
            
        
imagen2 = io.imread('perro.jpg')/255.0
imagen3 = np.zeros(imagen2.shape)
for ii in range(imagen2.shape[0]):
    for jj in range(imagen2.shape[1]):
        pixel1 = imagen2[ii,jj,:]
        columna = som.reshape(nfil*ncol,3)
        d = 0
        for n in range(3):
            d = d+ (pixel1[n]-columna[:,n])**2
        Dista = np.sqrt(d)
        ind = np.argmin(Dista)
        bmfil, bmcol=np.unravel_index(ind,[nfil, ncol])
        imagen3[ii,jj,:] = som[bmfil, bmcol,:]
                    
plt.figure()
plt.imshow(imagen2)

plt.figure()
plt.imshow(imagen3)

