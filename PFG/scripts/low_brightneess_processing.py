import cv2 as cv

image = cv.imread('Images/Pruebas/image.png')
cv.imshow('image', image)
q = cv.waitKey(0)

#Convertir a escala de grises
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
q = cv.waitKey(0)

#Mostrar brillo medio
mean_brightness = gray.mean()
print('Mean brightness: ', mean_brightness)

#Subir brillo
if mean_brightness < 100:
    gray = cv.convertScaleAbs(gray, alpha=2.0, beta=0)
    cv.imshow('gray', gray)
    q = cv.waitKey(0)
    print('No se sube el brillo')

#Aplicar clahe
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)


#Umbralización dinámica con Otsu
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow('thresh', thresh)
q = cv.waitKey(0)
