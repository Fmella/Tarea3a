# coding=utf-8

import numpy as np
import scipy.sparse as sc_s
import scipy.sparse.linalg as sc_sl

import json
import sys

# Se lee el archivo de la configuracion
with open(sys.argv[1]) as file:
    data = json.load(file)

# Problem Setup
H = data["height"]
W = data["width"]
L = data["lenght"]
h = 0.1

# Condiciones Dirichlet
TOP = data["ambient_temperature"]   # Aquarium Top
HA = data["heater_a"]   # Aquarium Bottom heater A
HB = data["heater_b"]   # Aquarium Bottom heater B

# Condiciones Neumann du/dn
F = -data["window_loss"]     # Laterales acuario
BOTTOM = 0                  # Aquarium Bottom

nl = int(L / h) + 1
nw = int(W / h) + 1
nh = int(H / h)

N = nl * nw * nh

def getK(i, j, k):
    return nl * nw * k + nl * j + i

def getIJK(K):
    face = K % (nl * nw)
    k = K // (nl * nw)
    j = face // nl
    i = face % nl
    return (i, j, k)

"""
print(nl, nw, nh, N)
print(getK(0, 0, 0), getIJK(0))
print(getK(1, 0, 0), getIJK(1))
print(getK(0, 1, 0), getIJK(2))
print(getK(0, 0, 1), getIJK(3))
print(getK(1, 1, 1), getIJK(4))
"""

# In this matrix we will write all the coefficients of the unknowns
A = sc_s.lil_matrix((N, N))

# In this vector we will write all the right side of the equations
b = np.zeros((N,))

# We iterate over each point inside the domain
# Each point has an equation associated
# The equation is different depending on the point location inside the domain
for i in range(nl):
    for j in range(nw):
        for k in range(nh):

            # We will write the equation associated with row k
            K = getK(i, j, k)

            # We obtain indices of the other coefficients
            K_up = getK(i, j, k + 1)
            K_down = getK(i, j, k - 1)
            K_north = getK(i, j + 1, k)
            K_south = getK(i, j - 1, k)
            K_east = getK(i + 1, j, k)
            K_west = getK(i - 1, j, k)

            # Depending on the location of the point, the equation is different

            # Bottom
            if k == 0:

                # Inicio Bordes Adyacentes a calefactores

                # Heater A Borde Adyacente south
                if ((nl - 1) // 5) <= i <= 2 * ((nl - 1) // 5) and j == ((nw - 1) // 3) - 1:
                    A[K, K_up] = 2
                    A[K, K_south] = 1
                    A[K, K_east] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = -HA

                # Heater A Borde Adyacente north
                elif ((nl - 1) // 5) <= i <= 2 * ((nl - 1) // 5) and j == 2 * ((nw - 1) // 3) + 1:
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_east] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = -HA

                # Heater A Borde Adyacente west
                elif i == ((nl - 1) // 5) - 1 and ((nw - 1) // 3) <= j <= 2 * ((nw - 1) // 3):
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_south] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = -HA

                # Heater A Borde Adyacente east
                elif i == 2 * ((nl - 1) // 5) + 1 and ((nw - 1) // 3) <= j <= 2 * ((nw - 1) // 3):
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_south] = 1
                    A[K, K_east] = 1
                    A[K, K] = -6
                    b[K] = -HA

                # Heater B Borde Adyacente south
                elif 3 * ((nl - 1) // 5) <= i <= 4 * ((nl - 1) // 5) and j == ((nw - 1) // 3) - 1:
                    A[K, K_up] = 2
                    A[K, K_south] = 1
                    A[K, K_east] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = -HB

                # Heater B Borde Adyacente north
                elif 3 * ((nl - 1) // 5) <= i <= 4 * ((nl - 1) // 5) and j == 2 * ((nw - 1) // 3) + 1:
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_east] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = -HB

                # Heater B Borde Adyacente west
                elif i == 3 * ((nl - 1) // 5) - 1 and ((nw - 1) // 3) <= j <= 2 * ((nw - 1) // 3):
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_south] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = -HB

                # Heater B Borde Adyacente east
                elif i == 4 * ((nl - 1) // 5) + 1 and ((nw - 1) // 3) <= j <= 2 * ((nw - 1) // 3):
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_south] = 1
                    A[K, K_east] = 1
                    A[K, K] = -6
                    b[K] = -HB

                # Fin Bordes Adyacentes a calefactores
                # Inicio Bordes Acuario

                # Acuario Borde south
                elif 0 < i < nl - 1 and j == 0:
                    A[K, K_up] = 2
                    A[K, K_north] = 2
                    A[K, K_east] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = -2 * h * F

                # Acuario Borde north
                elif 0 < i < nl - 1 and j == nw - 1:
                    A[K, K_up] = 2
                    A[K, K_south] = 2
                    A[K, K_east] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = -2 * h * F

                # Acuario Borde west
                elif i == 0 and 0 < j < nw - 1:
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_south] = 1
                    A[K, K_east] = 2
                    A[K, K] = -6
                    b[K] = -2 * h * F

                # Acuario Borde east
                elif i == nl - 1 and 0 < j < nw - 1:
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_south] = 1
                    A[K, K_west] = 2
                    A[K, K] = -6
                    b[K] = -2 * h * F

                # Fin Bordes Acuario
                # Inicio Esquinas Acuario

                # Acuario Esquina south west
                elif i == 0 and j == 0:
                    A[K, K_up] = 2
                    A[K, K_north] = 2
                    A[K, K_east] = 2
                    A[K, K] = -6
                    b[K] = -4 * h * F

                # Acuario Esquina south east
                elif i == nl - 1 and j == 0:
                    A[K, K_up] = 2
                    A[K, K_north] = 2
                    A[K, K_west] = 2
                    A[K, K] = -6
                    b[K] = -4 * h * F

                # Acuario Esquina north west
                elif i == 0 and j == nw - 1:
                    A[K, K_up] = 2
                    A[K, K_south] = 2
                    A[K, K_east] = 2
                    A[K, K] = -6
                    b[K] = -4 * h * F

                # Acuario Esquina north east
                elif i == nl - 1 and j == nw - 1:
                    A[K, K_up] = 2
                    A[K, K_south] = 2
                    A[K, K_west] = 2
                    A[K, K] = -6
                    b[K] = -4 * h * F

                # Fin Esquinas Acuario

                # Resto del fondo (aislado)
                else:
                    A[K, K_up] = 2
                    A[K, K_north] = 1
                    A[K, K_south] = 1
                    A[K, K_east] = 1
                    A[K, K_west] = 1
                    A[K, K] = -6
                    b[K] = 0

            # Sobre Heater A
            elif k == 1 and ((nl - 1) // 5) <= i <= 2 * ((nl - 1) // 5) and ((nw - 1) // 3) <= j <= 2 * ((nw - 1) // 3):
                A[K, K_up] = 1
                A[K, K_north] = 1
                A[K, K_south] = 1
                A[K, K_east] = 1
                A[K, K_west] = 1
                A[K, K] = -6
                b[K] = -HA

            # Sobre Heater B
            elif k == 1 and 3 * ((nl - 1) // 5) <= i <= 4 * ((nl - 1) // 5) and ((nw - 1) // 3) <= j <= 2 * ((nw - 1) // 3):
                A[K, K_up] = 1
                A[K, K_north] = 1
                A[K, K_south] = 1
                A[K, K_east] = 1
                A[K, K_west] = 1
                A[K, K] = -6
                b[K] = -HB

            # Interior
            elif 0 < i < nl - 1 and 0 < j < nw - 1 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_north] = 1
                A[K, K_south] = 1
                A[K, K_east] = 1
                A[K, K_west] = 1
                A[K, K] = -6
                b[K] = 0

            # West lateral
            elif i == 0 and 0 < j < nw - 1 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_north] = 1
                A[K, K_south] = 1
                A[K, K_east] = 2
                A[K, K] = -6
                b[K] = -2 * h * F

            # East lateral
            elif i == nl - 1 and 0 < j < nw - 1 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_north] = 1
                A[K, K_south] = 1
                A[K, K_west] = 2
                A[K, K] = -6
                b[K] = -2 * h * F

            # South lateral
            elif j == 0 and 0 < i < nl - 1 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_north] = 2
                A[K, K_east] = 1
                A[K, K_west] = 1
                A[K, K] = -6
                b[K] = -2 * h * F

            # North lateral
            elif j == nw - 1 and 0 < i < nl - 1 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_south] = 2
                A[K, K_east] = 1
                A[K, K_west] = 1
                A[K, K] = -6
                b[K] = -2 * h * F

            # Top
            elif k == nh - 1 and 0 < i < nl - 1 and 0 < j < nw - 1:
                A[K, K_down] = 1
                A[K, K_north] = 1
                A[K, K_south] = 1
                A[K, K_east] = 1
                A[K, K_west] = 1
                A[K, K] = -6
                b[K] = -TOP

            # Esquina south west laterales
            elif i == 0 and j == 0 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_north] = 2
                A[K, K_east] = 2
                A[K, K] = -6
                b[K] = -4 * h * F

            # Esquina south east laterales
            elif i == nl - 1 and j == 0 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_north] = 2
                A[K, K_west] = 2
                A[K, K] = -6
                b[K] = -4 * h * F

            # Esquina north west laterales
            elif i == 0 and j == nw - 1 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_south] = 2
                A[K, K_east] = 2
                A[K, K] = -6
                b[K] = -4 * h * F

            # Esquina north east laterales
            elif i == nl - 1 and j == nw - 1 and 0 < k < nh - 1:
                A[K, K_up] = 1
                A[K, K_down] = 1
                A[K, K_south] = 2
                A[K, K_west] = 2
                A[K, K] = -6
                b[K] = -4 * h * F

            # Esquina top south
            elif k == nh - 1 and j == 0 and 0 < i < nl - 1:
                A[K, K_down] = 1
                A[K, K_north] = 2
                A[K, K_east] = 1
                A[K, K_west] = 1
                A[K, K] = -6
                b[K] = -TOP - 2 * h * F

            # Esquina top north
            elif k == nh - 1 and j == nw - 1 and 0 < i < nl - 1:
                A[K, K_down] = 1
                A[K, K_south] = 2
                A[K, K_east] = 1
                A[K, K_west] = 1
                A[K, K] = -6
                b[K] = -TOP - 2 * h * F

            # Esquina top west
            elif k == nh - 1 and i == 0 and 0 < j < nw - 1:
                A[K, K_down] = 1
                A[K, K_north] = 1
                A[K, K_south] = 1
                A[K, K_east] = 2
                A[K, K] = -6
                b[K] = -TOP - 2 * h * F

            # Esquina top east
            elif k == nh - 1 and i == nl - 1 and 0 < j < nw - 1:
                A[K, K_down] = 1
                A[K, K_north] = 1
                A[K, K_south] = 1
                A[K, K_west] = 2
                A[K, K] = -6
                b[K] = -TOP - 2 * h * F

            # Esquina top south west
            elif k == nh - 1 and i == 0 and j == 0:
                A[K, K_down] = 1
                A[K, K_north] = 2
                A[K, K_east] = 2
                A[K, K] = -6
                b[K] = -TOP - 4 * h * F

            # Esquina top south east
            elif k == nh - 1 and i == nl - 1 and j == 0:
                A[K, K_down] = 1
                A[K, K_north] = 2
                A[K, K_west] = 2
                A[K, K] = -6
                b[K] = -TOP - 4 * h * F

            # Esquina top north west
            elif k == nh - 1 and i == 0 and j == nw - 1:
                A[K, K_down] = 1
                A[K, K_south] = 2
                A[K, K_east] = 2
                A[K, K] = -6
                b[K] = -TOP - 4 * h * F

            # Esquina top north east
            elif k == nh - 1 and i == nl - 1 and j == nw - 1:
                A[K, K_down] = 1
                A[K, K_south] = 2
                A[K, K_west] = 2
                A[K, K] = -6
                b[K] = -TOP - 4 * h * F


# Se convierte la matrix para resolver el sistema
A = A.tocsr()

# Se resuelve el sistema
x = sc_sl.spsolve(A, b)

# Se vuelve a la solucion al dominio en 3d
u = np.zeros((nl, nw, nh+1))

# Se escriben los temperaturas en la matriz 3d
for K in range(0, N):
    i, j, k = getIJK(K)
    u[i, j, k] = x[K]

# Se colocan los valores de los Heater A y B correspondientes
u[((nl-1)//5):2*((nl-1)//5), ((nw-1)//3):2*((nw-1)//3), 0] = HA
u[3*((nl-1)//5):4*((nl-1)//5), ((nw-1)//3):2*((nw-1)//3), 0] = HB

# Se coloca la temperatura de la superficie del agua
u[:, :, nh] = TOP


# Se guarda en un archivo dependiendo de la configuracion
filename = data["filename"]
np.save(filename, u)