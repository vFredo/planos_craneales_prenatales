import os
import matplotlib.pyplot as plt
from collections import Counter


class_labels = ["trans_cerebellum", "trans_thalamic", "trans_ventricular"]

def create_histogram(label_dir):
    class_counter = Counter()

    # Recorrer cada archivo de etiquetas
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])  # El primer número es la clase
                    class_counter[class_id] += 1  # Contar la ocurrencia de la clase

    # Generar el histograma usando los nombres de las clases
    classes = [class_labels[i] for i in class_counter.keys()]
    counts = list(class_counter.values())
    for i in class_counter.keys():
        print(class_labels[i], class_counter[i])

    plt.bar(classes, counts)
    plt.xlabel('Clase')
    plt.ylabel('Cantidad de imágenes')
    plt.title('Distribución de imágenes por clase')
    plt.xticks(rotation=45)  # Rotar las etiquetas de clase para mejor legibilidad
    plt.tight_layout()  # Ajustar el layout para evitar etiquetas cortadas
    plt.show()

create_histogram('backup_dataset/labels/')