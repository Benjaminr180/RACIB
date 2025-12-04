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
# MAIN PROVISIONAL DE PRUEBA
# =====================================================

if __name__ == "__main__":
    print("Iniciando prueba de visión RACIB (Bloques 1–4, webcam)...")

    # 1) Inicializar cámara
    cap = inicializar_camara()

    try:
        while True:
            # 2) Obtener frame completo
            frame = obtener_frame(cap)

            # 3) Dibujar la ROI sobre una copia del frame para visualización
            frame_viz = frame.copy()
            cv2.rectangle(
                frame_viz,
                (ROI_X1, ROI_Y1),
                (ROI_X2, ROI_Y2),
                (0, 255, 0),  # color verde del rectángulo
                2             # grosor de línea
            )

            # 4) Recortar ROI
            roi = recortar_roi(frame)

            # 5) Preprocesar ROI (gris, gris suavizada, HSV)
            gray, gray_blur, hsv = preprocesar_roi(roi)

            # 6) Para visualizar algo del HSV, mostramos solo el canal V (brillo)
            v_channel = hsv[:, :, 2]

            # 7) Mostrar ventanas
            cv2.imshow("RACIB - Frame completo (ROI marcada)", frame_viz)
            cv2.imshow("RACIB - ROI (color BGR)", roi)
            cv2.imshow("RACIB - ROI gris", gray)
            cv2.imshow("RACIB - ROI gris suavizada", gray_blur)
            cv2.imshow("RACIB - ROI canal V (HSV - brillo)", v_channel)

            # 8) Salir con la tecla 'q'
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