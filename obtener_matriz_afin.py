import cv2
import numpy as np

def detectar_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Poición del cursor (x, y): ({x}, {y})")
        param["clicks"] += 1
        param["puntos"].append([x, y])

def obtener_puntos(imagen_vis):
    params = {
        "clicks": 0,
        "puntos": []
    }
    cv2.namedWindow("Imagen")
    cv2.setMouseCallback("Imagen", detectar_click, param=params)

    while True:
        cv2.imshow("Imagen", imagen_vis)
        if cv2.waitKey(1) & 0xFF == ord('q') or params["clicks"] >= 3:
            break 

    cv2.destroyWindow("Imagen")
    cv2.waitKey(1)
    return params["puntos"]

# -----------------------------
# Cargar imágenes
# -----------------------------
imagen = cv2.imread("img/cameraman.png", cv2.IMREAD_GRAYSCALE)
imagen_procesada = cv2.imread("img/cameraman_processed.png", cv2.IMREAD_GRAYSCALE)

h, w = imagen.shape

# -----------------------------
# Seleccionar puntos
# -----------------------------
print("Selecciona 3 puntos en la imagen ORIGINAL")
puntos_origen = obtener_puntos(imagen)

print("Selecciona los mismos 3 puntos en la imagen PROCESADA")
puntos_destino = obtener_puntos(imagen_procesada)

# -----------------------------
# Estimar matriz
# -----------------------------
M = cv2.getAffineTransform(np.float32(puntos_origen),
                           np.float32(puntos_destino))

print("\nMatriz estimada:")
print(M)

# -----------------------------
# Generar imagen estimada
# -----------------------------
imagen_estimada = cv2.warpAffine(imagen, M, (w, h))

# -----------------------------
# Mostrar resultados
# -----------------------------
cv2.imshow("Original", imagen)
cv2.imshow("Procesada", imagen_procesada)
cv2.imshow("Estimada", imagen_estimada)

cv2.waitKey(0)
cv2.destroyAllWindows()


