# =====================================================
# BLOQUE 1 - PARÁMETROS GLOBALES Y CONFIGURACIÓN INICIAL
# =====================================================

import cv2
import numpy as np
import time

# ------------------------------
# Parámetros generales de la cámara (WEBCAM)
# ------------------------------

CAMERA_INDEX = 0          # Cámara que usaremos (0 = webcam principal)
FRAME_WIDTH = 1280        # Ancho de la imagen procesada
FRAME_HEIGHT = 720        # Alto de la imagen procesada

# -----------------------------
# CONFIGURACIÓN GLOBAL DEL SISTEMA
# -----------------------------
CANNY_LOW = 50
CANNY_HIGH = 150
# ------------------------------
# Región de interés (ROI)
# ------------------------------

ROI_X1, ROI_Y1 = 320, 180      # Esquina superior izquierda de la ROI
ROI_X2, ROI_Y2 = 960, 540   # Esquina inferior derecha de la ROI

# ------------------------------
# Parámetros de Canny (detección de bordes)
# ------------------------------

CANNY_LOW = 80         # Umbral inferior de Canny
    # =====================================================
# BLOQUE 2 - FUNCIONES DE CÁMARA
# =====================================================

def inicializar_camara(index: int = CAMERA_INDEX):
    """
    Abre la cámara y configura la resolución.
    Devuelve el objeto VideoCapture de OpenCV.
    """
    # Creamos un objeto de captura de video asociado al índice de la cámara.
    cap = cv2.VideoCapture(index)

    # Verificamos que la cámara se haya abierto correctamente.
    if not cap.isOpened():
        # Si algo falla, lanzamos un error claro.
        raise RuntimeError("No se pudo abrir la cámara. Revisa conexión o índice.")

    # Intentamos fijar la resolución deseada.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Pequeña pausa para que la cámara se estabilice (exposición, enfoque, etc.).
    time.sleep(0.5)

    return cap


def liberar_camara(cap):
    """
    Libera la cámara y cierra cualquier ventana de OpenCV.
    No cierra nada si aún no se abrieron ventanas.
    """
    cap.release()
    cv2.destroyAllWindows()
    # =====================================================
# BLOQUE 3 - OBTENER FRAME Y RECORTAR ROI
# =====================================================

def obtener_frame(cap):
    """
    Lee un frame de la cámara y lo redimensiona
    a FRAME_WIDTH x FRAME_HEIGHT.
    """
    # ret indica si la lectura fue exitosa (True/False).
    # frame contiene la imagen capturada.
    ret, frame = cap.read()

    # Si no se pudo leer un frame, lanzamos un error para detectarlo.
    if not ret:
        raise RuntimeError("No se pudo leer un frame de la cámara.")

    # Redimensionamos el frame a la resolución estándar definida arriba.
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    return frame


def recortar_roi(frame):
    """
    Recorta la región de interés (ROI) donde se encuentra el objeto
    sobre la banda transportadora.
    Usa las coordenadas globales ROI_X1, ROI_Y1, ROI_X2, ROI_Y2.
    """
    # En OpenCV (y NumPy), primero se indexa filas (eje Y) y luego columnas (eje X):
    # frame[ y_inicial : y_final , x_inicial : x_final ]
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    return roi
# =====================================================
# BLOQUE 4 - PREPROCESAMIENTO DE LA ROI
# =====================================================

def preprocesar_roi(roi):
    """
    A partir de la ROI en color (BGR, formato por defecto de OpenCV),
    genera:
    - imagen en escala de grises
    - imagen en HSV
    - versión suavizada en gris (para Canny y otras operaciones)

    Devuelve: gray, gray_blur, hsv
    """
    # Convertimos de BGR (formato interno de OpenCV) a escala de grises.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Convertimos de BGR a HSV (matiz, saturación, valor) para análisis de color.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Aplicamos un filtro Gaussiano para suavizar y reducir ruido.
    # El kernel (5, 5) define el tamaño de la ventana de convolución.
    # El último parámetro (0) deja que OpenCV calcule automáticamente sigma.
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray, gray_blur, hsv
# =====================================================
# BLOQUE 5 - DETECCIÓN DE BORDES CON CANNY
# =====================================================

def detectar_bordes_canny(gray_blur):
    """
    Aplica el detector de bordes Canny a la imagen en escala de grises suavizada.

    Parámetros:
        gray_blur: imagen en escala de grises ya suavizada con GaussianBlur.

    Usa los umbrales globales:
        CANNY_LOW  - umbral inferior
        CANNY_HIGH - umbral superior

    Devuelve:
        edges: imagen binaria (uint8) del mismo tamaño que gray_blur,
               donde 255 indica píxel perteneciente a un borde y 0 indica no-borde.
    """
    edges = cv2.Canny(gray_blur, CANNY_LOW, CANNY_HIGH)
    return edges
# =====================================================
# MAIN DE PRUEBA 2 - VISUALIZACIÓN CON CANNY
# =====================================================

if __name__ == "__main__":
    print("Iniciando prueba de visión RACIB (HD + Canny)...")

    cap = inicializar_camara()

    try:
        while True:
            # 1) Frame completo
            frame = obtener_frame(cap)

            # 2) Copia para dibujar la ROI
            frame_viz = frame.copy()
            cv2.rectangle(
                frame_viz,
                (ROI_X1, ROI_Y1),
                (ROI_X2, ROI_Y2),
                (0, 255, 0),  # verde
                2
            )

            # 3) Recortar ROI
            roi = recortar_roi(frame)

            # 4) Preprocesar ROI
            gray, gray_blur, hsv = preprocesar_roi(roi)

            # 5) Bordes con Canny
            edges = detectar_bordes_canny(gray_blur)

            # 6) Mostrar ventanas
            cv2.imshow("RACIB - Frame completo (ROI marcada)", frame_viz)
            cv2.imshow("RACIB - ROI (color BGR)", roi)
            cv2.imshow("RACIB - ROI gris", gray)
            cv2.imshow("RACIB - ROI gris suavizada", gray_blur)
            cv2.imshow("RACIB - ROI bordes Canny", edges)

            # 7) Salir con 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Tecla 'q' presionada. Saliendo...")
                break

    except RuntimeError as e:
        print("Error en ejecución:", e)

    finally:
        print("Liberando cámara y cerrando ventanas...")
        liberar_camara(cap)
        print("Fin de la prueba.")