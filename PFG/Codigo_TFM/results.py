import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_length_dir = "Depth_estimation/depth.csv"
csv_results_dir = "results.csv"
#bad_segmentations = [['fig_24_corrected.jpg',3],['fig_21_corrected.jpg',3],['fig_21_corrected.jpg',5],['fig_22_corrected.jpg',3],['fig_22_corrected.jpg',5]]
bad_segmentations = [['fig_24_corrected.jpg',3]]
include_bad_segmentation = 0

df = pd.read_csv(csv_length_dir, sep=';')

print("## Los peces que presentan errores en la detección de la cabeza y la cola son: 'fig_21_corrected.jpg'-pez 3, 'fig_21_corrected.jpg'-pez 5, 'fig_22_corrected.jpg'-pez 3, 'fig_22_corrected.jpg'-pez 5 ##")

if not include_bad_segmentation:
    df= df[(df['imagen'] != bad_segmentations[0][0]) | (df['id_pez'] != bad_segmentations[0][1])]
    #df= df[(df['imagen'] != bad_segmentations[1][0]) | (df['id_pez'] != bad_segmentations[1][1])]
    #df= df[(df['imagen'] != bad_segmentations[2][0]) | (df['id_pez'] != bad_segmentations[2][1])]
    #df= df[(df['imagen'] != bad_segmentations[3][0]) | (df['id_pez'] != bad_segmentations[3][1])]

## Numero de registros descartados
num_no_valid = len(df[df['profundidad de cabeza'] == 'Puntos mal seleccionados'])
# Registros no descartados
valid = df[df['profundidad de cabeza'] != 'Puntos mal seleccionados']
print("\n###############################################\n")
print(f"Numero de registros no válidos: {num_no_valid}")
print("\n###############################################\n")

##########################################################

Q1 = np.quantile(valid['error absoluto (mm)'], .25)
Q3 = np.quantile(valid['error absoluto (mm)'], .75)
atipicos_sup = Q3 + 1.5*(Q3-Q1)
atipicos_inf = Q1 - 1.5*(Q3-Q1)
extremos_sup = Q3 + 3*(Q3-Q1)
extremos_inf = Q1 - 3*(Q3-Q1)

atipicos = valid[(valid['error absoluto (mm)'] > atipicos_sup) | (valid['error absoluto (mm)'] < atipicos_inf)]
print(f"Valores atípicos: \n{atipicos[['imagen','id_pez','color','error absoluto (mm)']]}")
extremos = valid[(valid['error absoluto (mm)'] > extremos_sup) | (valid['error absoluto (mm)'] < extremos_inf)]
print(f"\nValores extremos: \n{extremos[['imagen','id_pez','color','error absoluto (mm)']]}\n")

print(f"Numero de atípicos: {len(atipicos)}")
print(f"Numero de extremos: {len(extremos)}")
print("\n###############################################\n")

########################################################

## Separación por color
df_rojo = valid[valid['color'] == 'rojo']
df_blanco = valid[valid['color'] == 'blanco']
df_verde = valid[valid['color'] == 'verde']

## Estadísticas rojo
median_rojo = df_rojo['error absoluto (mm)'].median()
mean_rojo = df_rojo['error absoluto (mm)'].mean()
std_rojo = df_rojo['error absoluto (mm)'].std()
mape_rojo = round(np.mean(np.asarray(df_rojo['error absoluto (mm)'])/(np.asarray(df_rojo['longitud real (m)']*1000))*100),2)

## Estadísticas blanco
median_blanco = df_blanco['error absoluto (mm)'].median()
mean_blanco = df_blanco['error absoluto (mm)'].mean()
std_blanco = df_blanco['error absoluto (mm)'].std()
mape_blanco = round(np.mean(np.asarray(df_blanco['error absoluto (mm)'])/(np.asarray(df_blanco['longitud real (m)']*1000))*100),2)

## Estadísticas verde
median_verde = df_verde['error absoluto (mm)'].median()
mean_verde = df_verde['error absoluto (mm)'].mean()
std_verde = df_verde['error absoluto (mm)'].std()
mape_verde = round(np.mean(np.asarray(df_verde['error absoluto (mm)'])/(np.asarray(df_verde['longitud real (m)']*1000))*100),2)

print("Separación por color")
print("--------------------")

print("Color rojo:")
print(f"Numero de registros: {len(df_rojo)}")
print(f"MAE: {mean_rojo}")
print(f"MedAE: {median_rojo}")
print(f"Desviación típica: {std_rojo}")
print(f"MAPE: {mape_rojo}\n")

print("Color blanco:")
print(f"Numero de registros: {len(df_blanco)}")
print(f"MAE: {mean_blanco}")
print(f"MedAE: {median_blanco}")
print(f"Desviación típica: {std_blanco}")
print(f"MAPE: {mape_blanco}\n")

print("Color verde:")
print(f"Numero de registros: {len(df_verde)}")
print(f"MAE: {mean_verde}")
print(f"MedAE: {median_verde}")
print(f"Desviación típica: {std_verde}")
print(f"MAPE: {mape_verde}\n")

print("\n###############################################\n")

#########################################################

## Detección outliers
errores_altos = valid[valid['error absoluto (mm)'] > 5]
errores_altos = errores_altos[['imagen','id_pez','color','error absoluto (mm)']]

print("Errores altos (> 0.5cm) *0.5cm es la diferencia de tamaño entre los tipos de peces, por lo que se considera un error alto")
print("--------")
print(errores_altos)
print("\n----------------------------------\n")
print(f"Numero de registros: {len(errores_altos)}")
print("\n###############################################\n")
#########################################################

## Separación por profundidad
shallow_depth = valid[valid['profundidad cuerpo'] < 0.75]
med_depth = valid[(valid['profundidad cuerpo'] >= 0.75) & (valid['profundidad cuerpo'] < 0.9)]
great_depth = valid[valid['profundidad cuerpo'] >=0.9]

## Estadísticas baja profundidad
median_shallow = shallow_depth['error absoluto (mm)'].median()
mean_shallow = shallow_depth['error absoluto (mm)'].mean()
std_shallow = shallow_depth['error absoluto (mm)'].std()
mape_shallow = round(np.mean(np.asarray(shallow_depth['error absoluto (mm)'])/(np.asarray(shallow_depth['longitud real (m)']*1000))*100),2)

## Estadísticas profundidad media
median_med = med_depth['error absoluto (mm)'].median()
mean_med = med_depth['error absoluto (mm)'].mean()
std_med = med_depth['error absoluto (mm)'].std()
mape_med = round(np.mean(np.asarray(med_depth['error absoluto (mm)'])/(np.asarray(med_depth['longitud real (m)']*1000))*100),2)

## Estadísticas alta profundidad
median_great = great_depth['error absoluto (mm)'].median()
mean_great = great_depth['error absoluto (mm)'].mean()
std_great = great_depth['error absoluto (mm)'].std()
mape_great = round(np.mean(np.asarray(great_depth['error absoluto (mm)'])/(np.asarray(great_depth['longitud real (m)']*1000))*100),2)

print("Separación por profundidad")
print("--------------------------")

print("Profundidad baja:")
print(f"Numero de registros: {len(shallow_depth)}")
print(f"MAE: {mean_shallow}")
print(f"MedAE: {median_shallow}")
print(f"Desviación típica: {std_shallow}")
print(f"MAPE: {mape_shallow}\n")

print("Profundidad media:")
print(f"Numero de registros: {len(med_depth)}")
print(f"MAE: {mean_med}")
print(f"MedAE: {median_med}")
print(f"Desviación típica: {std_med}")
print(f"MAPE: {mape_med}\n")

print("Profundidad alta:")
print(f"Numero de registros: {len(great_depth)}")
print(f"MAE: {mean_great}")
print(f"MedAE: {median_great}")
print(f"Desviación típica: {std_great}")
print(f"MAPE: {mape_great}\n")

print("\n###############################################\n")

########################################################

# Separacion por color Y profundidad

shallow_rojo = pd.concat([df_rojo, shallow_depth], axis=1, join="inner")
med_rojo = pd.concat([df_rojo, med_depth], axis=1, join="inner")
great_rojo = pd.concat([df_rojo, great_depth], axis=1, join="inner")

shallow_blanco = pd.concat([df_blanco, shallow_depth], axis=1, join="inner")
med_blanco = pd.concat([df_blanco, med_depth], axis=1, join="inner")
great_blanco = pd.concat([df_blanco, great_depth], axis=1, join="inner")

shallow_verde = pd.concat([df_verde, shallow_depth], axis=1, join="inner")
med_verde = pd.concat([df_verde, med_depth], axis=1, join="inner")
great_verde = pd.concat([df_verde, great_depth], axis=1, join="inner")

# Estadísticas por color Y profundidad

## Estadísticas color rojo
median_shallow_rojo = shallow_rojo['error absoluto (mm)'].median()
mean_shallow_rojo = shallow_rojo['error absoluto (mm)'].mean()
std_shallow_rojo = shallow_rojo['error absoluto (mm)'].std()
mape_shallow_rojo = round(np.mean(np.asarray(shallow_rojo['error absoluto (mm)'])/(np.asarray(shallow_rojo['longitud real (m)']*1000))*100),2)

median_med_rojo = med_rojo['error absoluto (mm)'].median()
mean_med_rojo = med_rojo['error absoluto (mm)'].mean()
std_med_rojo = med_rojo['error absoluto (mm)'].std()
mape_med_rojo = round(np.mean(np.asarray(med_rojo['error absoluto (mm)'])/(np.asarray(med_rojo['longitud real (m)']*1000))*100),2)

median_great_rojo = great_rojo['error absoluto (mm)'].median()
mean_great_rojo = great_rojo['error absoluto (mm)'].mean()
std_great_rojo = great_rojo['error absoluto (mm)'].std()
mape_great_rojo = round(np.mean(np.asarray(great_rojo['error absoluto (mm)'])/(np.asarray(great_rojo['longitud real (m)']*1000))*100),2)

## Estadísticas color blanco
median_shallow_blanco = shallow_blanco['error absoluto (mm)'].median()
mean_shallow_blanco = shallow_blanco['error absoluto (mm)'].mean()
std_shallow_blanco = shallow_blanco['error absoluto (mm)'].std()
mape_shallow_blanco = round(np.mean(np.asarray(shallow_blanco['error absoluto (mm)'])/(np.asarray(shallow_blanco['longitud real (m)']*1000))*100),2)

median_med_blanco = med_blanco['error absoluto (mm)'].median()
mean_med_blanco = med_blanco['error absoluto (mm)'].mean()
std_med_blanco = med_blanco['error absoluto (mm)'].std()
mape_med_blanco = round(np.mean(np.asarray(med_blanco['error absoluto (mm)'])/(np.asarray(med_blanco['longitud real (m)']*1000))*100),2)

median_great_blanco = great_rojo['error absoluto (mm)'].median()
mean_great_blanco = great_blanco['error absoluto (mm)'].mean()
std_great_blanco = great_blanco['error absoluto (mm)'].std()
mape_great_blanco = round(np.mean(np.asarray(great_blanco['error absoluto (mm)'])/(np.asarray(great_blanco['longitud real (m)']*1000))*100),2)

## Estadísticas color verde
median_shallow_verde = shallow_verde['error absoluto (mm)'].median()
mean_shallow_verde = shallow_verde['error absoluto (mm)'].mean()
std_shallow_verde = shallow_verde['error absoluto (mm)'].std()
mape_shallow_verde = round(np.mean(np.asarray(shallow_verde['error absoluto (mm)'])/(np.asarray(shallow_verde['longitud real (m)']*1000))*100),2)

median_med_verde = med_verde['error absoluto (mm)'].median()
mean_med_verde = med_verde['error absoluto (mm)'].mean()
std_med_verde = med_verde['error absoluto (mm)'].std()
mape_med_verde = round(np.mean(np.asarray(med_verde['error absoluto (mm)'])/(np.asarray(med_verde['longitud real (m)']*1000))*100),2)

median_great_verde = great_verde['error absoluto (mm)'].median()
mean_great_verde = great_verde['error absoluto (mm)'].mean()
std_great_verde = great_verde['error absoluto (mm)'].std()
mape_great_verde = round(np.mean(np.asarray(great_verde['error absoluto (mm)'])/(np.asarray(great_verde['longitud real (m)']*1000))*100),2)

print("Separación por profundidad Y color")
print("----------------------------------\n")

print("Color rojo")
print("----------")

print("Profundidad baja:")
print(f"Numero de registros: {len(shallow_rojo)}")
print(f"MAE: {mean_shallow_rojo[0]}")
print(f"MedAE: {median_shallow_rojo[0]}")
print(f"Desviación típica: {std_shallow_rojo[0]}")
print(f"MAPE: {mape_shallow_rojo}\n")

print("Profundidad media:")
print(f"Numero de registros: {len(med_rojo)}")
print(f"MAE: {mean_med_rojo[0]}")
print(f"MedAE: {median_med_rojo[0]}")
print(f"Desviación típica: {std_med_rojo[0]}")
print(f"MAPE: {mape_med_rojo}\n")

print("Profundidad alta:")
print(f"Numero de registros: {len(great_rojo)}")
print(f"MAE: {mean_great_rojo[0]}")
print(f"MedAE: {median_great_rojo[0]}")
print(f"Desviación típica: {std_great_rojo[0]}")
print(f"MAPE: {mape_great_rojo}")

print("\n----------------------------------------\n")
print("Color blanco")
print("----------")

print("Profundidad baja:")
print(f"Numero de registros: {len(shallow_blanco)}")
print(f"MAE: {mean_shallow_blanco[0]}")
print(f"MedAE: {median_shallow_blanco[0]}")
print(f"Desviación típica: {std_shallow_blanco[0]}")
print(f"MAPE: {mape_shallow_blanco}\n")

print("Profundidad media:")
print(f"Numero de registros: {len(med_blanco)}")
print(f"MAE: {mean_med_blanco[0]}")
print(f"MedAE: {median_med_blanco[0]}")
print(f"Desviación típica: {std_med_blanco[0]}")
print(f"MAPE: {mape_med_blanco}\n")

print("Profundidad alta:")
print(f"Numero de registros: {len(great_blanco)}")
print(f"MAE: {mean_great_blanco[0]}")
print(f"MedAE: {median_great_blanco[0]}")
print(f"Desviación típica: {std_great_blanco[0]}")
print(f"MAPE: {mape_great_blanco}")

print("\n----------------------------------------\n")
print("Color verde")
print("----------")

print("Profundidad baja:")
print(f"Numero de registros: {len(shallow_verde)}")
print(f"MAE: {mean_shallow_verde[0]}")
print(f"MedAE: {median_shallow_verde[0]}")
print(f"Desviación típica: {std_shallow_verde[0]}")
print(f"MAPE: {mape_shallow_verde}\n")

print("Profundidad media:")
print(f"Numero de registros: {len(med_verde)}")
print(f"MAE: {mean_med_verde[0]}")
print(f"MedAE: {median_med_verde[0]}")
print(f"Desviación típica: {std_med_verde[0]}")
print(f"MAPE: {mape_med_verde}\n")

print("Profundidad alta:")
print(f"Numero de registros: {len(great_verde)}")
print(f"MAE: {mean_great_verde[0]}")
print(f"MedAE: {median_great_verde[0]}")
print(f"Desviación típica: {std_great_verde[0]}")
print(f"MAPE: {mape_great_verde}")

print("\n###############################################\n")

print("Análisis conjunto")
print("-----------------\n")

print(f"Numero total de registros: {len(valid)}")
print(f"MAE: {valid['error absoluto (mm)'].mean()}")
print(f"MedAE: {valid['error absoluto (mm)'].median()}")
print(f"Desviación típica: {valid['error absoluto (mm)'].std()}")
print(f"MAPE: {round(np.mean(np.asarray(valid['error absoluto (mm)'])/(np.asarray(valid['longitud real (m)']*1000))*100),2)}")

print("\n###############################################\n")

plt.plot(valid['longitud real (m)'], valid['error absoluto (mm)'], 'ro')
plt.xlabel('Tamaño real (m)')
plt.ylabel('Error absoluto (mm)')
plt.title('Separación train/test')
plt.show()

plt.figure()
plt.plot(valid['profundidad cuerpo'], valid['error absoluto (mm)'], 'ro')
plt.xlabel('Profundidad (m)')
plt.ylabel('Error absoluto (mm)')
plt.title('Separación train/test')
plt.show()