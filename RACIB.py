import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Parámetros de LBP
LBP_POINTS = 24          # puntos vecinos
LBP_RADIUS = 3           # radio
LBP_METHOD = 'uniform'   # patrón uniforme

def analizar_color_hsv(hsv):
    # Mascara para tonos típicos de plástico (azules, verdes, rojos)
    lower_plastico = np.array([0, 40, 40])
    upper_plastico = np.array([179, 255, 255])
    mask_plastico = cv2.inRange(hsv, lower_plastico, upper_plastico)

    plastico_pct = (np.sum(mask_plastico > 0) / mask_plastico.size) * 100
    return plastico_pct

def analizar_bordes(gray):
    edges = cv2.Canny(gray, 80, 160)
    bordes_pct = (np.sum(edges > 0) / edges.size) * 100
    return bordes_pct, edges

def analizar_brillo(gray):
    _, thresh_bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    brillo_pct = (np.sum(thresh_bright > 0) / thresh_bright.size) * 100
    return brillo_pct, thresh_bright

def analizar_textura(gray):
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    # Histograma normalizado de la textura
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS+3), range=(0, LBP_POINTS+2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Medida de uniformidad (entre más uniforme, más metálico)
    uniformidad = np.sum(hist[:LBP_POINTS])  
    return uniformidad

def detectar_material(frame):
    # Reducción de ruido
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Conversión
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # --- 1. Análisis de color
    plastico_pct = analizar_color_hsv(hsv)

    # --- 2. Bordes
    bordes_pct, edges = analizar_bordes(gray)

    # --- 3. Brillo
    brillo_pct, bright_mask = analizar_brillo(gray)

    # --- 4. Textura (LBP)
    uniformidad_textura = analizar_textura(gray)

    # --- 5. Regla final
    # Estos umbrales los ajustan ustedes con pruebas reales:
    es_metal = (brillo_pct > 1.5) and (bordes_pct > 5) and (uniformidad_textura > 0.2)
    es_plastico = (plastico_pct > 10) and (bordes_pct < 7) and (brillo_pct < 1.0)

    if es_metal:
        material = "METAL"
    elif es_plastico:
        material = "PLASTICO"
    else:
        material = "DESCONOCIDO"

    return material, {
        "plastico_pct": plastico_pct,
        "bordes_pct": bordes_pct,
        "brillo_pct": brillo_pct,
        "uniformidad_textura": uniformidad_textura,
        "edges": edges,
        "bright_mask": bright_mask
    }


# ---------------------------
# Main loop con cámara
# ---------------------------
cap = cv2.VideoCapture(0)  # 0 = cámara principal; en Raspberry puede ser 0 o 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    material, datos = detectar_material(frame)

    # Mostrar resultado en pantalla
    cv2.putText(frame, f"Material: {material}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Detector de Metal/Plastico", frame)

    # Mostrar ventanas auxiliares
    cv2.imshow("Bordes (Canny)", datos["edges"])
    cv2.imshow("Brillo", datos["bright_mask"])

    if cv2.waitKey(1) == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
