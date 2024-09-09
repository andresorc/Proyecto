import re
import cv2
import json
import os

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

def get_video_index_from_user(video_side):
    while True:
        try:
            video_index = int(input(f"Ingrese el índice del video {video_side}: "))
            video_path = f'../../Videos/Sup{video_side}-{video_index}.mp4'
            if os.path.exists(video_path):
                return video_path, video_index
            else:
                print(f"No se encontró el archivo {video_path}. Intente nuevamente.")
        except ValueError:
            print("Índice inválido. Por favor, ingrese un número entero.")

def get_frame_number_from_user(video_path, video_side, max_frame_number):
    while True:
        try:
            frame_number = int(input(f"Ingrese el número de frame base para el video {video_side} (máximo {max_frame_number}): "))
            if 0 <= frame_number < max_frame_number:
                return frame_number
            else:
                print(f"El número de frame debe estar entre 0 y {max_frame_number - 1}. Intente nuevamente.")
        except ValueError:
            print("Número de frame inválido. Por favor, ingrese un número entero.")

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# Función para obtener la diferencia de frames entre dos vídeos desde un archivo JSON
def get_frame_difference(video_index, json_path):
    with open(json_path, 'r') as file:
        differences = json.load(file)
    
    key = f"diferencia_{video_index}"
    if key in differences:
        return differences[key]
    else:
        raise ValueError(f"No se encontró la diferencia de frames para: {key}")

# Se cargan las rutas de los videos ingresadas por el usuario
video_path_l, video_index_l = get_video_index_from_user('I')
video_index_r = video_index_l
video_path_r = f'../../Videos/SupD-{video_index_r}.mp4'


# Se obtiene la diferencia de frames entre los dos videos desde el archivo JSON
json_path = 'diferencias_frames.json'
frame_difference = get_frame_difference(video_index_l, json_path)
print(frame_difference)

# Se obtiene el número total de frames para cada video
total_frames_l = get_total_frames(video_path_l)
total_frames_r = get_total_frames(video_path_r)

# Se obtiene el número de frame base ingresado por el usuario para el video izquierdo
base_frame_number = get_frame_number_from_user(video_path_l, 'izquierdo', total_frames_l)

# Se calcula el frame correspondiente en el otro video
frame_number_l = base_frame_number
frame_number_r = base_frame_number - frame_difference

# Verificar que los frames calculados estén dentro de los límites de los videos
if frame_number_r < 0 or frame_number_r >= total_frames_r:
    print(f"El número de frame calculado para el video derecho ({frame_number_r}) está fuera de los límites.")

# Se extrae el frame específico de cada video
frame_l = extract_frame(video_path_l, frame_number_l)
frame_r = extract_frame(video_path_r, frame_number_r)

# Se ajusta el tamaño para la visualización
frame_show = cv2.resize(frame_l, (640, 480))
frame2_show = cv2.resize(frame_r, (640, 480))

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

print("Frames extraídos y guardados con éxito.")

print("Frames extraídos y guardados con éxito.")
