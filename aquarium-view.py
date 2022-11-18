# coding=utf-8

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np

import json
import sys

import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import scene_graph as sg


# Se lee el archivo de la configuracion
with open(sys.argv[1]) as file:
    data = json.load(file)

# Se carga la data segun el setup
temperaturas = np.load(data["filename"])

# Usando h de aquarium-solver se pueden calcular las dimensiones del acuario
h = 0.1

dimensiones = np.zeros(3)
dimensiones[0], dimensiones[1], dimensiones[2] = temperaturas.shape
dimensiones -= 1
dimensiones *= h
dimensiones = dimensiones.astype('int32')

# Modifica la escala de las medidas en el programa
escala = 0.5
escalado = escala * h

# Data
# Peces tipo A
t_a = data["t_a"]
n_a = data["n_a"]

# Peces tipo B
t_b = data["t_b"]
n_b = data["n_b"]

# Peces tipo C
t_c = data["t_c"]
n_c = data["n_c"]

# Zona donde se guardan los puntos para los peces tipo A, B y C
zonaA = []
zonaB = []
zonaC = []

# Zonas donde los peces pueden posicionarse, para evitar que salgan del acuario
# Se tiene un rango para establecer la distancia a las paredes permitida
rango = 3
zonaPecesA = []
zonaPecesB = []
zonaPecesC = []


# Se encuentran los puntos que cumplen las condiciones de temperatura de cada zona
for i in range(len(temperaturas)):
    for j in range(len(temperaturas[0])):
        for k in range(len(temperaturas[0, 0])):

            # Se ubican los puntos (i, j, k) que cumplen los rangos de temperatura
            # y se trasladan para que el centro del acuario este en el origen
            # Si esta suficientemente alejado de las paredes se incluye en la zona para peces
            if t_a - 2 <= temperaturas[i, j, k] <= t_a + 2:
                posicion = [i * escalado - dimensiones[0] * escala / 2,
                            j * escalado - dimensiones[1] * escala / 2,
                            k * escalado]
                zonaA.append(posicion)
                if rango <= i <= len(temperaturas)-rango and rango <= j <= len(temperaturas[0])-rango and 1 <= k:
                    zonaPecesA.append(posicion)

            if t_b - 2 <= temperaturas[i, j, k] <= t_b + 2:
                posicion = [i * escalado - dimensiones[0] * escala / 2,
                            j * escalado - dimensiones[1] * escala / 2,
                            k * escalado]
                zonaB.append(posicion)
                if rango <= i <= len(temperaturas)-rango and rango <= j <= len(temperaturas[0])-rango and 1 <= k:
                    zonaPecesB.append(posicion)

            if t_c - 2 <= temperaturas[i, j, k] <= t_c + 2:
                posicion = [i * escalado - dimensiones[0] * escala / 2,
                            j * escalado - dimensiones[1] * escala / 2,
                            k * escalado]
                zonaC.append(posicion)
                if rango <= i <= len(temperaturas)-rango and rango <= j <= len(temperaturas[0])-rango and 1 <= k:
                    zonaPecesC.append(posicion)

# Se utiliza para crear un shape de un conjunto de puntos de cierto color
def createColorZonePoints(zone, color):

    vertices = []
    for vertice in zone:
        vertices += [vertice[0], vertice[1], vertice[2], color[0], color[1], color[2]]

    indices = list(range(len(zone)))

    return bs.Shape(vertices, indices)

# Se utiliza para formar los bordes del acuario escalando las medidas
def createAquariumShape(dimensiones):

    # Se trasladan los vertices para que el centro del acuario quede en el origen
    vertices =[
        # vertices                                                                  #color
        -dimensiones[0]*escala/2, -dimensiones[1]*escala/2, 0,                      0, 0, 0,
        dimensiones[0]*escala/2,  -dimensiones[1]*escala/2, 0,                      0, 0, 0,
        dimensiones[0]*escala/2,   dimensiones[1]*escala/2, 0,                      0, 0, 0,
        -dimensiones[0]*escala/2,  dimensiones[1]*escala/2, 0,                      0, 0, 0,
        -dimensiones[0]*escala/2, -dimensiones[1]*escala/2, dimensiones[2]*escala,  0, 0, 0,
        dimensiones[0]*escala/2,  -dimensiones[1]*escala/2, dimensiones[2]*escala,  0, 0, 0,
        dimensiones[0]*escala/2,   dimensiones[1]*escala/2, dimensiones[2]*escala,  0, 0, 0,
        -dimensiones[0]*escala/2,  dimensiones[1]*escala/2, dimensiones[2]*escala,  0, 0, 0
    ]

    indices = [0, 1, 1, 2, 2, 3, 3, 0,
               4, 5, 5, 6, 6, 7, 7, 4,
               0, 4, 1, 5, 2, 6, 3, 7]

    return bs.Shape(vertices, indices)

def createTextureFishBody(image_filename):

    vertices = [
        # vertices             # textura
        -53/75,  0.0, -28/75,  23/200, 86/100,
         16/75,  0.0, -29/75,  91/200, 86/100,
         43/75,  0.0, -20/75, 118/200, 78/100,
        -75/75,  0.0,  -5/75,   1/200, 64/100,
        -53/75, -0.2,      0,  23/200, 60/100,
         18/75, -0.2,      0,  93/200, 59/100,
        -53/75,  0.2,      0,  23/200, 60/100,
         18/75,  0.2,      0,  93/200, 59/100,
         75/75,  0.0,  -6/75, 148/200, 63/100,
         75/75,  0.0,   5/75, 148/200, 56/100,
        -52/75,  0.0,  32/75,  25/200, 28/100,
        -27/75,  0.0,  39/75,  48/200, 20/100,
         19/75,  0.0,  29/75,  93/200, 30/100,
        # aletas
         19/75,  0.0, -39/75,  89/200, 100/100,
         49/75,  0.0, -29/75, 137/200, 95/100,
        -15/75,  0.0,  57/75,  60/200,  0/100,
         22/75,  0.0,  43/75,  106/200, 14/100
    ]

    indices =[
        # lateral izquierdo
        3, 0, 4, 4, 0, 1, 4, 1, 5, 5, 1, 2, 5, 2, 8, 5, 8, 9,
        3, 4, 10, 4, 11, 10, 4, 5, 11, 5, 12, 11, 12, 5, 9,
        # lateral derecho
        9, 8, 7, 8, 2, 7, 7, 2, 1, 7, 1, 6, 6, 1, 0, 6, 0, 3,
        9, 7, 12, 12, 7, 11, 11, 7, 6, 11, 6, 10, 10, 6, 3,
        # aletas
        1, 13, 2, 2, 13, 14,
        11, 16, 15, 11, 12, 16
    ]

    textureFilename = image_filename

    return bs.Shape(vertices, indices, textureFilename)


def createTextureFishTail(image_filename):

    vertices = [
        # vertices           # textura
          0.0,  0.0,  -6/75, 150/200, 64/100,
         7/75,  0.0,  -9/75, 155/200, 67/100,
        22/75,  0.0, -30/75, 171/200, 88/100,
        45/75,  0.0, -40/75, 194/200,    1.0,
        34/75,  0.0,  -8/75, 180/200, 67/100,
        33/75,  0.0,   7/75, 180/200, 53/100,
        50/75,  0.0,  41/75,     1.0, 17/100,
        28/75,  0.0,  35/75, 176/200, 24/100,
         6/75,  0.0,   8/75, 155/200, 51/100,
          0.0,  0.0,   6/75, 150/200, 54/100,
        20/75, -0.1, -10/75, 169/200, 69/100,
        21/75, -0.1,  15/75, 169/200, 44/100,
        20/75,  0.1, -10/75, 169/200, 69/100,
        21/75,  0.1,  15/75, 169/200, 44/100
    ]

    indices = [
        # lateral izquierdo
        0, 1, 9, 9, 1, 8, 1, 10, 8, 1, 2, 10, 10, 2, 3, 10, 3, 4,
        8, 10, 11, 11, 10, 5, 5, 10, 4, 8, 11, 7, 7, 11, 6, 11, 5, 6,
        # lateral derecho
        4, 3, 12, 12, 3, 2, 12, 2, 1, 12, 1, 8, 8, 1, 9, 9, 1, 0,
        4, 12, 5, 12, 13, 5, 12, 8, 13, 5, 13, 6, 13, 7, 6, 13, 8, 7
    ]

    textureFilename = image_filename

    return bs.Shape(vertices, indices, textureFilename)

def createNFishes(N, zone, image_filename, escala):

    gpuFishTail = es.toGPUShape(createTextureFishTail(image_filename), GL_REPEAT, GL_NEAREST)
    gpuFishBody = es.toGPUShape(createTextureFishBody(image_filename), GL_REPEAT, GL_NEAREST)

    fishes = sg.SceneGraphNode("fishes")
    tailBase = "fishTail"
    fishBase = "fish"

    for i in range(N):

        # Se buscan posiciones aleatorias en la zona que tiene disponible el pez
        random_num = np.random.randint(len(zone))
        random_pos = zone[random_num]

        # Cola
        newTailNode = sg.SceneGraphNode(tailBase + str(i))
        newTailNode.transform = tr.uniformScale(escala)
        newTailNode.childs += [gpuFishTail]

        # Nodo para mover la cola
        newTailRotationNode = sg.SceneGraphNode(tailBase + "Rotation" + str(i))
        newTailRotationNode.childs += [newTailNode]

        # Nodo que ajusta la cola al cuerpo del pez
        newTailTraslatedNode = sg.SceneGraphNode(tailBase + "Traslated" + str(i))
        newTailTraslatedNode.transform = tr.translate(escala, 0, 0)
        newTailTraslatedNode.childs += [newTailRotationNode]

        # Cuerpo
        newBodyNode = sg.SceneGraphNode(fishBase + "Body" + str(i))
        newBodyNode.transform = tr.uniformScale(escala)
        newBodyNode.childs += [gpuFishBody]

        # Pez completo, se rota aleatoriamente y se mueve a la posicion aleatoria de su zona
        random_rotation = np.random.uniform(0, 2 * np.pi)
        newFishNode = sg.SceneGraphNode(fishBase + str(i))
        newFishNode.transform = tr.matmul([tr.translate(random_pos[0], random_pos[1], random_pos[2]),
                                           tr.rotationZ(random_rotation)])
        newFishNode.childs += [newTailTraslatedNode, newBodyNode]

        # Se agrega al conjunto de peces de su tipo
        fishes.childs += [newFishNode]

    return fishes

def createAquarium(image_filename):

    x0 = -dimensiones[0] * escala / 2
    x1 = dimensiones[0] * escala / 2
    y0 = -dimensiones[1] * escala / 2
    y1 = dimensiones[1] * escala / 2
    z0 = 0
    z1 = dimensiones[2] * escala

    vertices = [
        # piso
        # vertices  # textura
        x0, y0, z0, 0, 1/2,
        x1, y0, z0, 1, 1/2,
        x1, y1, z0, 1, 0,
        x0, y1, z0, 0, 0,
        # ventanas
        # vertices  # textura
        x0, y0, z0, 0, 1,
        x1, y0, z0, 1, 1,
        x1, y0, z1, 1, 1/2 + 0.1,
        x0, y0, z1, 0, 1/2 + 0.1,
        x1, y0, z0, 0, 1,
        x1, y1, z0, 1, 1,
        x1, y1, z1, 1, 1/2 + 0.1,
        x1, y0, z1, 0, 1/2 + 0.1,
        x1, y1, z0, 0, 1,
        x0, y1, z0, 1, 1,
        x0, y1, z1, 1, 1/2 + 0.1,
        x1, y1, z1, 0, 1/2 + 0.1,
        x0, y1, z0, 0, 1,
        x0, y0, z0, 1, 1,
        x0, y0, z1, 1, 1/2 + 0.1,
        x0, y1, z1, 0, 1/2 + 0.1,
    ]

    indices = [
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
        8, 9, 10, 8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19
    ]

    textureFilename = image_filename

    return bs.Shape(vertices, indices, textureFilename)



# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.zonaA = False
        self.zonaB = False
        self.zonaC = False

# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return

    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_A:
        controller.zonaA = not controller.zonaA

    elif key == glfw.KEY_B:
        controller.zonaB = not controller.zonaB

    elif key == glfw.KEY_C:
        controller.zonaC = not controller.zonaC

    elif key == glfw.KEY_ESCAPE:
        sys.exit()

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Aquarium", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Defining shader program
    pipeline = es.SimpleModelViewProjectionShaderProgram()
    textPipeline = es.SimpleTextureModelViewProjectionShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(pipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Enabling transparencies
    glEnable(GL_BLEND)
    glEnable(GL_CULL_FACE)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))
    gpuZoneA = es.toGPUShape(createColorZonePoints(zonaA, [224/255, 177/255, 19/255]))
    gpuZoneB = es.toGPUShape(createColorZonePoints(zonaB, [24/255, 142/255, 58/255]))
    gpuZoneC = es.toGPUShape(createColorZonePoints(zonaC, [196/255, 135/255, 155/255]))
    gpuAquariumShape = es.toGPUShape(createAquariumShape(dimensiones))
    gpuAquarium = es.toGPUShape(createAquarium("acuario.png"), GL_REPEAT, GL_NEAREST)

    # Se crean las shapes de los grupos de peces
    fishesA = createNFishes(n_a, zonaPecesA, "pezA.png", 0.062)
    fishesB = createNFishes(n_b, zonaPecesB, "pezB.png", 0.065)
    fishesC = createNFishes(n_c, zonaPecesC, "pezC.png", 0.068)

    # Velocidades para las colas de los peces
    colasA = np.random.uniform(1, 5, n_a)
    colasB = np.random.uniform(1, 5, n_b)
    colasC = np.random.uniform(1, 5, n_c)

    t0 = glfw.get_time()
    camera_theta = np.pi

    camera_min = max(dimensiones[0], dimensiones[1])/2 + 0.1
    R = camera_min + 2
    AtZ = 1.5 * escala
    pointSize = 0.6

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        glPointSize(pointSize)

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_E) == glfw.PRESS) and pointSize < 5:
            pointSize += 0.2

        if (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS) and pointSize > 0.2:
            pointSize -= 0.2

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta += dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta -= dt

        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS) and R > camera_min:
            R -= dt * 8

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS) and R < camera_min * 2:
            R += dt * 8

        if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS) and AtZ < dimensiones[2] * escala:
            AtZ += 0.1

        if (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS) and AtZ > 0:
            AtZ -= 0.1

        # Setting up the view transform
        camX = R * np.sin(camera_theta)
        camY = R * np.cos(camera_theta)
        camZ = 1.5 * escala
        viewPos = np.array([camX, camY, camZ])
        AtPos = np.array([0, 0, AtZ])

        view = tr.lookAt(
            viewPos,
            AtPos,
            np.array([0, 0, 1])
        )

        # Setting up the projection transform
        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Drawing shapes

        # Peces
        glUseProgram(textPipeline.shaderProgram)

        # Movimiento de cola peces A
        for i in range(n_a):
            fishTailRotationNode = sg.findNode(fishesA, "fishTailRotation" + str(i))
            fishTailRotationNode.transform = tr.rotationZ(np.cos(colasA[i] * t1))

        # Movimiento de cola peces B
        for i in range(n_b):
            fishTailRotationNode = sg.findNode(fishesB, "fishTailRotation" + str(i))
            fishTailRotationNode.transform = tr.rotationZ(np.cos(colasB[i] * t1))

        # Movimiento de cola peces C
        for i in range(n_c):
            fishTailRotationNode = sg.findNode(fishesC, "fishTailRotation" + str(i))
            fishTailRotationNode.transform = tr.rotationZ(np.cos(colasC[i] * t1))

        glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        sg.drawSceneGraphNode(fishesA, textPipeline, "model")
        sg.drawSceneGraphNode(fishesB, textPipeline, "model")
        sg.drawSceneGraphNode(fishesC, textPipeline, "model")

        # Zonas y forma de acuario
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
#        pipeline.drawShape(gpuAxis, GL_LINES)
        pipeline.drawShape(gpuAquariumShape, GL_LINES)

        if controller.zonaA:
            pipeline.drawShape(gpuZoneA, GL_POINTS)

        if controller.zonaB:
            pipeline.drawShape(gpuZoneB, GL_POINTS)

        if controller.zonaC:
            pipeline.drawShape(gpuZoneC, GL_POINTS)

        # Acuario
        glUseProgram(textPipeline.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        textPipeline.drawShape(gpuAquarium)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()