import cv2

# Función que extrae un frame de un video en función del número de frame especificado
def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el archivo de video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"No se pudo leer el frame {frame_number} del video: {video_path}")
    return frame

# Se cargan los videos
video_path = '../../Videos/SupI-20230629182035-20230629182536.mp4'
video_path2 = '../../Videos/SupD-20230629182038-20230629182538.mp4'

# Se extrae un frame específico de cada video
frame_l = extract_frame(video_path, 720)  # Ejemplo de extraer el frame número 1200
frame_r = extract_frame(video_path2, 603)  # Ejemplo de extraer el frame número 2100

# Se ajusta el tamaño para la visualización
frame_show = cv2.resize(frame_l, (640, 480))
frame2_show = cv2.resize(frame_r, (640, 480))

# Se reconvierte a BGR para la visualización
frame_show = cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB)
frame2_show = cv2.cvtColor(frame2_show, cv2.COLOR_BGR2RGB)

# Se muestran los frames
cv2.imshow('Frame', frame_show)
cv2.imshow('Frame2', frame2_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ruta de salida para guardar los frames
output_path = 'Images/Pruebas/cam_sup_l.jpg'
output_path2 = 'Images/Pruebas/cam_sup_r.jpg'

# Se guardan los frames
cv2.imwrite(output_path, frame_l)
cv2.imwrite(output_path2, frame_r)