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
CANNY_HIGH = 160       # Umbral superior de Canny
