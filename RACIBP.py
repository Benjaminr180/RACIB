# =====================================================
# BLOQUE 1 - PARÁMETROS GLOBALES Y CONFIGURACIÓN INICIAL
# =====================================================

import cv2
import numpy as np
import time

# ------------------------------
# Parámetros generales de la cámara (WEBCAM)
# ------------------------------

CAMERA_INDEX = 0        # Cámara que usaremos No cambiar este parametro
FRAME_WIDTH = 640       # Ancho de la imagen procesada
FRAME_HEIGHT = 480      # Alto de la imagen procesada

# ------------------------------
# Región de interés (ROI)
# ------------------------------

ROI_X1, ROI_Y1 = 160, 120   # Esquina superior izquierda de la ROI
ROI_X2, ROI_Y2 = 480, 360   # Esquina inferior derecha de la ROI

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