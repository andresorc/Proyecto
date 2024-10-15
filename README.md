Para el correcto funcionamiento de los scripts que componen el sistema de estimación de tamaño de rodaballos, primero se debe crear un entorno virtual en Visual Studio y teniendo como directorio de trabajo la carpeta “PFG”. También hay que instalar todas las dependencias especificadas en PFG/requirements.txt mediante el comando pip install -r requirements.txt. 

El usuario debe ejecutar cada uno de los siguientes scripts y notebooks en el orden que aparece reflejado:
1. scripts\extraer_frames.py: se pide al usuario por terminal que elija el par de vídeos sobre los que se quieren extraer los frames. En principio hay 3 vídeos L y R (o bien I y D), aunque en el servidor del NAS se encuentran más vídeos que habría que sincronizar primero encontrando la diferencia en frames que existe entre un par de ellos. Esta diferencia se debe apuntar en el archivo diferencias_frames.json. En principio no se proporciona la carpeta de vídeos por su peso, pero si alguien quisiese probar el sistema se la podría enviar yo personalmente
2. Codigo_TFM\Depth_estimation\img_correction.py
3. notebooks\automatic_mask_generator_turbots.ipynb: debido a la alta cantidad de recursos que requiere este módulo, se debe ejecutar en el entorno de Google Colab. Este se puede abrir desde Visual Studio Code al pinchar en el icono que aparece al principio del notebook. Después se debe cargar la imagen izquierda (cam_sup_l_corrected_{sufijo}.jpg) que fue corregida en el punto 2, las cual está en el directorio Images\Pruebas\Corrected.
4. scripts\compare_images.py: se pide al usuario el sufijo de la imagen de entrada. Es el sufijo de mayor número si se quiere procesar la última imagen con la que se ha tratado en los scripts anteriores.
5. scripts\tratamiento_aislado.py
6. scripts\contours.py
7. Codigo_TFM\Segmentation\segmentation.py
8. Codigo_TFM\Depth_estimation\depth_estimation.py

For the proper functioning of the scripts that make up the turbot size estimation system, a virtual environment must first be created in Visual Studio, with the "PFG" folder set as the working directory. Additionally, all dependencies specified in PFG/requirements.txt must be installed using the command pip install -r requirements.txt.

The user must execute each of the following scripts and notebooks in the order shown:

1. scripts\extraer_frames.py: The user is prompted via terminal to select the pair of videos from which frames should be extracted. Initially, there are 3 L and R (or I and D) videos, although more videos are available on the NAS server, which should first be synchronized by finding the frame difference between pairs. This difference should be noted in the file diferencias_frames.json. Initially, the video folder is not provided due to its size, but if anyone wants to test the system, I could personally send it to them.

2. Codigo_TFM\Depth_estimation\img_correction.py

3. notebooks\automatic_mask_generator_turbots.ipynb: Due to the high resource requirements of this module, it must be run in the Google Colab environment. It can be opened from Visual Studio Code by clicking the icon that appears at the start of the notebook. Then, the left image (cam_sup_l_corrected_{suffix}.jpg), which was corrected in step 2, should be loaded from the directory Images\Pruebas\Corrected.

4. scripts\compare_images.py: The user is asked for the suffix of the input image. This is the suffix with the highest number if the user wants to process the most recent image worked on in the previous scripts.

5. scripts\tratamiento_aislado.py

6. scripts\contours.py

7. Codigo_TFM\Segmentation\segmentation.py

8. Codigo_TFM\Depth_estimation\depth_estimation.py
