import cv2


#funcion que extrae el frame de un video en funcion del parametro que indica el minuto, el segundo y el frame
def extract_frame(video_path, minute, second, frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el archivo de video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Los FPS del video no pueden ser 0. Verifica el archivo y su formato.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_seconds = total_frames / fps
    total_minutes = total_seconds / 60
    total_minutes = int(total_minutes)
    total_seconds = int(total_seconds)
    total_frames = int(total_frames)
    if minute > total_minutes:
        print("El video no tiene tantos minutos")
        return
    if second > total_seconds:
        print("El video no tiene tantos segundos")
        return
    if frame > total_frames:
        print("El video no tiene tantos frames")
        return
    total_seconds = minute * 60 + second
    total_frames = total_seconds * fps + frame
    print(total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames)
    ret, frame = cap.read()
    cap.release()
    return frame

#Se cargan dos videos
video_path = 'Videos/cam_sup_l.mp4'
video_path2 = 'Videos/cam_sup_r.mp4'

#Se extrae un frame de cada video
#frame = extract_frame(video_path, 4, 0, 5)
#frame2 = extract_frame(video_path2, 3, 56, 7)


frame_l = extract_frame(video_path, 4, 0, 5)#04 : 00 : 00
#mostrar numero de frame
print(frame_l.shape)
frame_r = extract_frame(video_path2, 3, 56, 7)#03 : 56 : 07
#mostrar numero de frame
print(frame_r.shape)

#Se pasa a escala de grises
frame_gray = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
frame2_gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)



#Se aplica una ecualizacion para homogeneizar iluminacion
frame_gray = cv2.equalizeHist(frame_gray)
frame2_gray = cv2.equalizeHist(frame2_gray)


#Se ajusta el tamaño ppara la visualizacion
frame_show= cv2.resize(frame_gray, (640, 480))
frame2_show = cv2.resize(frame2_gray, (640, 480))


#Se reconvierte a BGR para la visualizacion
frame_show = cv2.cvtColor(frame_show, cv2.COLOR_GRAY2BGR)
frame2_show = cv2.cvtColor(frame2_show, cv2.COLOR_GRAY2BGR)

#Se muestran los frames
cv2.imshow('Frame', frame_show)
cv2.imshow('Frame2', frame2_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Ruta de salida
output_path = 'Images/Pruebas/cam_sup_l_corrected_gray.jpg'
output_path2 = 'Images/Pruebas/cam_sup_r_corrected_gray.jpg'

output_path_color = 'Images/Pruebas/cam_sup_l.jpg'
output_path2_color = 'Images/Pruebas/cam_sup_r.jpg'

#Se guardan los frames
cv2.imwrite(output_path_color, frame_l)
cv2.imwrite(output_path2_color, frame_r)


