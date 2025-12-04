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

ROI_X1, ROI_Y1 = 320, 180      # Esquina superior izquierda de la ROI
ROI_X2, ROI_Y2 = 960, 540   # Esquina inferior derecha de la ROI

# ------------------------------
# Parámetros de Canny (detección de bordes)
# ------------------------------

CANNY_LOW = 80          # Umbral inferior de Canny (puede ajustarse con pruebas)
CANNY_HIGH = 160        # Umbral superior de Canny

# ------------------------------
# Parámetros para detección de reflejos especulares
# ------------------------------
# Trabajaremos sobre el canal V (brillo) y S (saturación) del espacio HSV.

HIGHLIGHT_V_THRESHOLD = 220   # Umbral mínimo de brillo (0-255) para considerar "muy brillante"
HIGHLIGHT_S_MAX = 80          # Máxima saturación para considerarlo reflejo "blanco/plateado"
# ------------------------------
# Parámetros para histogramas de color
# ------------------------------
# Número de bins para histogramas 1D (por canal).
# 32 bins es un buen compromiso entre detalle y estabilidad.
HIST_BINS = 32

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
        raise RuntimeError("No se pudo iniciar la cámara. Revisa conexión o índice.")

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
# BLOQUE 6 - DETECCIÓN DE BRILLO / REFLEJOS (SPECULAR HIGHLIGHT)
# =====================================================

def detectar_reflejos_specular(hsv_roi):
    """
    Detecta regiones muy brillantes (reflejos especulares) en la ROI usando el espacio HSV.

    Parámetros:
        hsv_roi: imagen de la ROI en espacio de color HSV (salida de preprocesar_roi).

    Estrategia:
        - Usar el canal V (brillo) para localizar píxeles muy luminosos.
        - Opcionalmente restringir a saturación baja (S <= HIGHLIGHT_S_MAX) para favorecer
          reflejos casi blancos/plateados típicos de superficies metálicas.

    Devuelve:
        mask_reflejos: imagen binaria (uint8) del mismo tamaño que hsv_roi[:,:,0],
                       donde 255 indica píxel con reflejo especular y 0 indica lo contrario.
        ratio_reflejo: fracción de píxeles de la ROI marcados como reflejo (entre 0.0 y 1.0).
    """
    # Separar canales H, S, V
    h, s, v = cv2.split(hsv_roi)

    # 1) Máscara de brillo alto: V >= HIGHLIGHT_V_THRESHOLD
    # Creamos una máscara binaria donde 255 = muy brillante, 0 = no brillante.
    mask_v_alto = cv2.inRange(v, HIGHLIGHT_V_THRESHOLD, 255)

    # 2) Máscara de baja saturación: S <= HIGHLIGHT_S_MAX
    # Esto ayuda a descartar colores intensos (plásticos coloreados)
    # y a quedarnos con zonas más "blancas/plateadas".
    mask_s_baja = cv2.inRange(s, 0, HIGHLIGHT_S_MAX)

    # 3) Combinar ambas condiciones: (brillo alto) AND (saturación baja)
    mask_reflejos = cv2.bitwise_and(mask_v_alto, mask_s_baja)

    # 4) Calcular razón de píxeles con reflejo
    num_reflejos = np.count_nonzero(mask_reflejos)
    total_pixeles = mask_reflejos.size

    if total_pixeles > 0:
        ratio_reflejo = num_reflejos / float(total_pixeles)
    else:
        ratio_reflejo = 0.0

    return mask_reflejos, ratio_reflejo     
# =====================================================
# BLOQUE 7 - HISTOGRAMAS DE COLOR (HSV Y BGR)
# =====================================================

def calcular_histogramas_hsv(hsv_roi, bins: int = HIST_BINS):
    """
    Calcula histogramas normalizados (L1) para cada canal H, S, V de la ROI en HSV.

    Parámetros:
        hsv_roi: imagen de la ROI en espacio HSV.
        bins: número de bins para cada histograma.

    Devuelve:
        h_hist, s_hist, v_hist: histogramas 1D normalizados (suma = 1.0) por canal.
    """
    # Histograma del canal H (matiz), rango típico 0-180 en OpenCV
    h_hist = cv2.calcHist([hsv_roi], [0], None, [bins], [0, 180])
    # Histograma del canal S (saturación), rango 0-256
    s_hist = cv2.calcHist([hsv_roi], [1], None, [bins], [0, 256])
    # Histograma del canal V (brillo), rango 0-256
    v_hist = cv2.calcHist([hsv_roi], [2], None, [bins], [0, 256])

    # Normalización L1: la suma de cada histograma será 1.0
    h_hist = cv2.normalize(h_hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    s_hist = cv2.normalize(s_hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    v_hist = cv2.normalize(v_hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

    return h_hist, s_hist, v_hist


def calcular_histogramas_bgr(roi_bgr, bins: int = HIST_BINS):
    """
    Calcula histogramas normalizados (L1) para cada canal B, G, R de la ROI en BGR.

    Parámetros:
        roi_bgr: imagen de la ROI en BGR (la ROI original recortada).
        bins: número de bins para cada histograma.

    Devuelve:
        b_hist, g_hist, r_hist: histogramas 1D normalizados (suma = 1.0) por canal.
    """
    # Canal B
    b_hist = cv2.calcHist([roi_bgr], [0], None, [bins], [0, 256])
    # Canal G
    g_hist = cv2.calcHist([roi_bgr], [1], None, [bins], [0, 256])
    # Canal R
    r_hist = cv2.calcHist([roi_bgr], [2], None, [bins], [0, 256])

    # Normalización L1: la suma de cada histograma será 1.0
    b_hist = cv2.normalize(b_hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    g_hist = cv2.normalize(g_hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    r_hist = cv2.normalize(r_hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

    return b_hist, g_hist, r_hist


def extraer_caracteristicas_color_basicas(hsv_roi):
    """
    Calcula estadísticas simples de color a partir de la ROI en HSV.

    Parámetros:
        hsv_roi: imagen de la ROI en espacio HSV.

    Devuelve:
        features: diccionario con estadísticas básicas:
            - mean_h, mean_s, mean_v
            - std_h, std_s, std_v
    """
    h, s, v = cv2.split(hsv_roi)

    mean_h = float(np.mean(h))
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))

    std_h = float(np.std(h))
    std_s = float(np.std(s))
    std_v = float(np.std(v))

    features = {
        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v,
        "std_h": std_h,
        "std_s": std_s,
        "std_v": std_v,
    }

    return features
# =====================================================
# BLOQUE 8 - RATIO DE BORDES + CLASIFICACIÓN BÁSICA PET / METAL
# =====================================================

# Umbrales iniciales para clasificación (se ajustan con pruebas reales)
EDGE_RATIO_MIN = 0.003        # Mínimo de bordes para considerar que hay un objeto
REFLECTION_RATIO_METAL = 0.02 # Si el ratio de reflejo supera esto, sugiere metal
MEAN_V_MIN_PRESENCIA = 40.0   # Brillo promedio mínimo para considerar que hay algo
MEAN_S_MAX_METAL = 80.0       # Saturación promedio máxima típica de metal (colores poco saturados)


def calcular_ratio_bordes(edges):
    """
    Calcula la fracción de píxeles que son bordes en la imagen 'edges'
    (salida de detectar_bordes_canny).

    Parámetros:
        edges: imagen binaria (0 o 255) de bordes.

    Devuelve:
        ratio_bordes: número de píxeles de borde / número total de píxeles.
    """
    num_bordes = np.count_nonzero(edges)
    total_pixeles = edges.size

    if total_pixeles > 0:
        ratio_bordes = num_bordes / float(total_pixeles)
    else:
        ratio_bordes = 0.0

    return ratio_bordes


def clasificar_material(ratio_reflejo, color_feats, ratio_bordes):
    """
    Clasificación heurística simple del material de la ROI:
        - "NINGUNO" : no hay objeto relevante en la ROI
        - "METAL"   : probable residuo metálico
        - "PET"     : probable residuo de plástico/PET

    Parámetros:
        ratio_reflejo: fracción de píxeles marcados como reflejo (salida de detectar_reflejos_specular).
        color_feats: diccionario de extraer_caracteristicas_color_basicas(hsv_roi).
        ratio_bordes: fracción de píxeles que son bordes en la ROI (salida de calcular_ratio_bordes).

    Devuelve:
        etiqueta: string con uno de {"NINGUNO", "METAL", "PET"}.
        info: diccionario con datos internos útiles para depurar/explicar la decisión.
    """
    mean_v = color_feats["mean_v"]
    mean_s = color_feats["mean_s"]

    # 1) ¿Hay algo en la ROI o está casi vacía?
    #    Si hay muy pocos bordes, poco brillo y casi sin reflejos, asumimos "NINGUNO".
    if (ratio_bordes < EDGE_RATIO_MIN and
        ratio_reflejo < REFLECTION_RATIO_METAL * 0.5 and
        mean_v < MEAN_V_MIN_PRESENCIA):
        etiqueta = "NINGUNO"
        motivo = "Pocos bordes, poco brillo y casi sin reflejos: ROI casi vacía."
    else:
        # 2) Caso METAL: suele tener bastantes reflejos blancos/plateados,
        #    y saturación relativamente baja (no tan coloreado).
        if (ratio_reflejo >= REFLECTION_RATIO_METAL and
            mean_s <= MEAN_S_MAX_METAL and
            mean_v >= MEAN_V_MIN_PRESENCIA):
            etiqueta = "METAL"
            motivo = "Reflejo alto y saturación baja: típico de metal brillante."
        else:
            # 3) Si no es 'NINGUNO' ni cumple condiciones de metal, lo consideramos PET.
            etiqueta = "PET"
            motivo = "No cumple criterios de metal, pero hay presencia de objeto: se asume PET."

    info = {
        "ratio_reflejo": ratio_reflejo,
        "ratio_bordes": ratio_bordes,
        "mean_v": mean_v,
        "mean_s": mean_s,
        "motivo": motivo
    }

    return etiqueta, info
# =====================================================
# BLOQUE 8 - RATIO DE BORDES + CLASIFICACIÓN BÁSICA PET / METAL
# =====================================================

# Umbrales iniciales para clasificación (se ajustan con pruebas reales)
EDGE_RATIO_MIN = 0.003        # Mínimo de bordes para considerar que hay un objeto
REFLECTION_RATIO_METAL = 0.02 # Si el ratio de reflejo supera esto, sugiere metal
MEAN_V_MIN_PRESENCIA = 40.0   # Brillo promedio mínimo para considerar que hay algo
MEAN_S_MAX_METAL = 80.0       # Saturación promedio máxima típica de metal (colores poco saturados)


def calcular_ratio_bordes(edges):
    """
    Calcula la fracción de píxeles que son bordes en la imagen 'edges'
    (salida de detectar_bordes_canny).

    Parámetros:
        edges: imagen binaria (0 o 255) de bordes.

    Devuelve:
        ratio_bordes: número de píxeles de borde / número total de píxeles.
    """
    num_bordes = np.count_nonzero(edges)
    total_pixeles = edges.size

    if total_pixeles > 0:
        ratio_bordes = num_bordes / float(total_pixeles)
    else:
        ratio_bordes = 0.0

    return ratio_bordes


def clasificar_material(ratio_reflejo, color_feats, ratio_bordes):
    """
    Clasificación heurística simple del material de la ROI:
        - "NINGUNO" : no hay objeto relevante en la ROI
        - "METAL"   : probable residuo metálico
        - "PET"     : probable residuo de plástico/PET

    Parámetros:
        ratio_reflejo: fracción de píxeles marcados como reflejo (salida de detectar_reflejos_specular).
        color_feats: diccionario de extraer_caracteristicas_color_basicas(hsv_roi).
        ratio_bordes: fracción de píxeles que son bordes en la ROI (salida de calcular_ratio_bordes).

    Devuelve:
        etiqueta: string con uno de {"NINGUNO", "METAL", "PET"}.
        info: diccionario con datos internos útiles para depurar/explicar la decisión.
    """
    mean_v = color_feats["mean_v"]
    mean_s = color_feats["mean_s"]

    # 1) ¿Hay algo en la ROI o está casi vacía?
    #    Si hay muy pocos bordes, poco brillo y casi sin reflejos, asumimos "NINGUNO".
    if (ratio_bordes < EDGE_RATIO_MIN and
        ratio_reflejo < REFLECTION_RATIO_METAL * 0.5 and
        mean_v < MEAN_V_MIN_PRESENCIA):
        etiqueta = "NINGUNO"
        motivo = "Pocos bordes, poco brillo y casi sin reflejos: ROI casi vacía."
    else:
        # 2) Caso METAL: suele tener bastantes reflejos blancos/plateados,
        #    y saturación relativamente baja (no tan coloreado).
        if (ratio_reflejo >= REFLECTION_RATIO_METAL and
            mean_s <= MEAN_S_MAX_METAL and
            mean_v >= MEAN_V_MIN_PRESENCIA):
            etiqueta = "METAL"
            motivo = "Reflejo alto y saturación baja: típico de metal brillante."
        else:
            # 3) Si no es 'NINGUNO' ni cumple condiciones de metal, lo consideramos PET.
            etiqueta = "PET"
            motivo = "No cumple criterios de metal, pero hay presencia de objeto: se asume PET."

    info = {
        "ratio_reflejo": ratio_reflejo,
        "ratio_bordes": ratio_bordes,
        "mean_v": mean_v,
        "mean_s": mean_s,
        "motivo": motivo
    }

    return etiqueta, info
# =====================================================
# MAIN DE PRUEBA 5 - CLASIFICACIÓN EN TIEMPO REAL (METAL / PET / NINGUNO)
# =====================================================

if __name__ == "__main__":
    print("Iniciando prueba 5: Clasificación METAL / PET / NINGUNO...")

    cap = inicializar_camara()
    frame_counter = 0

    try:
        while True:
            # 1) Captura y ROI
            frame = obtener_frame(cap)
            frame_viz = frame.copy()

            # Dibujar ROI
            cv2.rectangle(
                frame_viz,
                (ROI_X1, ROI_Y1),
                (ROI_X2, ROI_Y2),
                (0, 255, 0),
                2
            )

            roi = recortar_roi(frame)

            # 2) Preprocesamiento
            gray, gray_blur, hsv = preprocesar_roi(roi)

            # 3) Canny y ratio de bordes
            edges = detectar_bordes_canny(gray_blur)
            ratio_bordes = calcular_ratio_bordes(edges)

            # 4) Reflejos
            mask_reflejos, ratio_reflejo = detectar_reflejos_specular(hsv)

            # 5) Color
            color_feats = extraer_caracteristicas_color_basicas(hsv)

            # 6) Clasificación
            etiqueta, info = clasificar_material(ratio_reflejo, color_feats, ratio_bordes)

            # 7) Texto grande con la clasificación
            texto_clase = f"CLASIFICACION: {etiqueta}"
            cv2.putText(
                frame_viz,
                texto_clase,
                (30, 70),                     # posición (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,                           # tamaño de letra
                (0, 0, 255) if etiqueta == "METAL" else (0, 255, 0) if etiqueta == "PET" else (255, 255, 0),
                3,
                cv2.LINE_AA
            )

            # 8) Texto pequeño con datos de apoyo
            texto_info1 = f"Reflejo: {info['ratio_reflejo']:.4f}  Bordes: {info['ratio_bordes']:.4f}"
            texto_info2 = f"mean_V: {info['mean_v']:.1f}  mean_S: {info['mean_s']:.1f}"
            cv2.putText(
                frame_viz,
                texto_info1,
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame_viz,
                texto_info2,
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
                cv2.LINE_AA
            )

            # 9) Mostrar ventanas principales
            cv2.imshow("RACIB - Frame completo (clasificacion)", frame_viz)
            cv2.imshow("RACIB - ROI (color BGR)", roi)
            cv2.imshow("RACIB - ROI bordes Canny", edges)
            cv2.imshow("RACIB - ROI reflejos especulares", mask_reflejos)

            # (Opcional) si quieres ver gris o canal V, puedes destapar estas:
            # cv2.imshow("RACIB - ROI gris", gray)
            # v_channel = hsv[:, :, 2]
            # cv2.imshow("RACIB - ROI canal V (HSV - brillo)", v_channel)

            # 10) Log en consola de vez en cuando
            frame_counter += 1
            if frame_counter % 20 == 0:
                print("===== FRAME", frame_counter, "=====")
                print("Etiqueta:", etiqueta)
                print("Info:", info)
                print()

            # 11) Salir con 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Tecla 'q' presionada. Saliendo...")
                break

    except RuntimeError as e:
        print("Error en ejecución:", e)

    finally:
        print("Liberando cámara y cerrando ventanas...")
        liberar_camara(cap)
        print("Fin de la prueba 5.")