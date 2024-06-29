import cv2
import numpy as np

# Función para agregar texto con un cuadro negro en una esquina específica
def add_text_box(image, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_type = cv2.LINE_AA

    # Obtener el tamaño del texto
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Coordenadas del cuadro de texto
    text_origin = (pos[0] - text_size[0], pos[1])
    text_bottom_left = (text_origin[0], text_origin[1] + text_size[1] + 10)

    # Crear un fondo negro para el texto
    cv2.rectangle(image, text_origin, (image.shape[1],image.shape[0]), (0, 0, 0), cv2.FILLED)

    # Escribir el texto en rojo
    cv2.putText(image, text, (text_origin[0] + 5, text_origin[1] + text_size[1] + 5),
                font, font_scale, (0, 0, 255), font_thickness, line_type)

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar la imagen de ejemplo (puedes usar tu propia imagen aquí)
    image = cv2.imread(r'C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Results\cam_sup_r_corrected_1.jpg')
    
    #reducir tamaño de la imagen
    image = cv2.resize(image, (640,480))

    # Definir el texto a mostrar
    text = "Numero de objetos detectados: 10"  # Puedes cambiar el número

    # Obtener las dimensiones de la imagen
    height, width, _ = image.shape

    # Coordenadas para la esquina inferior derecha
    bottom_right_corner = (width - 50, height - 50)

    # Agregar el cuadro de texto a la imagen
    add_text_box(image, text, bottom_right_corner)

    # Mostrar la imagen con el cuadro de texto agregado
    cv2.imshow('Imagen con cuadro de texto', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
