# Modelo Liquid Foundation para Generación de Texto

Este repositorio contiene el código para un modelo de generación de texto llamado "Liquid Foundation". El modelo está diseñado con un enfoque en la eficiencia y el rendimiento, incorporando varias técnicas avanzadas y optimizaciones arquitectónicas.

## Arquitectura del Modelo

La arquitectura central del modelo, definida en `simple.py`, se basa en una estructura similar a la de un transformador con varias mejoras clave:

* **Liquid Embedding (Incrustación Líquida):** Una capa de incrustación personalizada (`LiquidEmbedding`) que incorpora capas convolucionales y compresión adaptativa basada en la complejidad de la secuencia. Esto tiene como objetivo mejorar la representación de los tokens de entrada mientras se gestionan los costos computacionales.  La compresión adaptativa permite al modelo ajustar dinámicamente la cantidad de información retenida de la incrustación, optimizando el uso de recursos para secuencias más cortas o menos complejas.  Las capas convolucionales ayudan a capturar características locales y dependencias dentro de la secuencia de entrada.

* **Atención Local Mejorada con Group Query Attention (GQA):** (`EnhancedLocalAttentionWithGQA`) emplea atención local dentro de un tamaño de ventana definido para mayor eficiencia, combinado con GQA para reducir la huella de memoria y mejorar el rendimiento.  La atención local limita el alcance de la atención a una ventana alrededor de cada token, reduciendo la complejidad cuadrática de la atención completa. GQA (Group Query Attention) divide las consultas en grupos y calcula la atención para cada grupo por separado, lo que reduce aún más la complejidad computacional. Además, integra Rotary Position Embeddings (RoPE) para manejar información posicional, lo que permite al modelo comprender el orden de los tokens en la secuencia.

* **Mezcla de Expertos (MoE):** (`MoELayer`) utiliza una capa MoE para enrutar los tokens de entrada a diferentes expertos, lo que permite un procesamiento especializado y potencialmente mejora la capacidad del modelo para capturar patrones diversos en los datos.  Cada experto es una red neuronal independiente que se especializa en un subconjunto del espacio de entrada.  El enrutamiento dinámico de expertos y la regularización de uso se implementan para una mejor asignación de recursos y estabilidad del entrenamiento.  Esto asegura que los expertos se utilicen de manera eficiente y evita que un solo experto domine el proceso.

* **Convolución Deformable:** (`DeformableConv1d`) introduce convoluciones deformables para capturar características más flexibles y dependientes del contexto.  A diferencia de las convoluciones tradicionales, las convoluciones deformables permiten que el modelo aprenda desplazamientos para los puntos de muestreo, lo que le permite adaptarse a diferentes formas y patrones en los datos.

* **Convolución Optimizada con Puerta:** (`OptimizedGatedConvolution`) combina convoluciones deformables con mecanismos de puerta y normalización para una mejor extracción de características y flujo de gradiente.  Los mecanismos de puerta controlan el flujo de información a través de la red, mientras que la normalización ayuda a estabilizar el entrenamiento y mejorar la convergencia.

* **LSTM Mejorado:** (`EnhancedLSTM`) integra un módulo LSTM como memoria externa, lo que potencialmente mejora la capacidad del modelo para manejar dependencias de largo alcance e información contextual.  El LSTM actúa como un búfer de memoria, almacenando información relevante de pasos de tiempo anteriores que puede ser utilizada para informar la generación de texto en pasos de tiempo posteriores.

* **Bloque Transformador Mejorado:** (`ImprovedTransformerBlock`) combina los componentes anteriores en un bloque transformador con normalización previa a la capa (Pre-LN) y optimizaciones adicionales para la estabilidad y el rendimiento.  Pre-LN aplica la normalización de capa antes de la atención y las capas de feedforward, lo que se ha demostrado que mejora la estabilidad del entrenamiento.

## Entrenamiento y Evaluación

El proceso de entrenamiento y evaluación se gestiona mediante `analysis_main.py`. Los aspectos clave incluyen:

* **Carga y Preprocesamiento del Conjunto de Datos:** El código utiliza la biblioteca `datasets` para cargar el conjunto de datos "TIGER-Lab/WebInstructSub". Los pasos de preprocesamiento incluyen limpieza de texto, eliminación de escapes HTML, normalización Unicode, eliminación de URL y normalización de espacios en blanco.  Estos pasos aseguran que los datos estén limpios y consistentes, lo que facilita el entrenamiento del modelo.

* **Tokenización:** Se utiliza el `LEDTokenizer` para la tokenización, con tokens especiales agregados para relleno, fin de secuencia, inicio de secuencia y separación.  La tokenización convierte el texto en una secuencia de IDs numéricas que el modelo puede procesar.

* **Funciones de Pérdida:** El modelo se entrena utilizando una combinación de pérdida de entropía cruzada y pérdida focal, junto con términos de regularización para la reconstrucción y la entropía.  La pérdida de entropía cruzada mide la diferencia entre la distribución de probabilidad predicha y la distribución real, mientras que la pérdida focal se centra en ejemplos difíciles de clasificar.  Los términos de regularización ayudan a prevenir el sobreajuste y promueven un modelo más generalizado.

* **Optimizador y Programador:** Se utiliza el optimizador `AdamW` con un programador de tasa de aprendizaje `CosineAnnealingWarmRestarts`.  El escalado de gradientes también se emplea para el entrenamiento de precisión mixta.  `AdamW` es un optimizador popular para modelos de aprendizaje profundo, y `CosineAnnealingWarmRestarts` ajusta la tasa de aprendizaje de forma cíclica para mejorar la convergencia.

* **Métricas:** El código calcula un conjunto completo de métricas, que incluyen precisión de tokens, precisión de secuencia, precisión top-k, distinct-n, longitud promedio de secuencia, perplejidad, puntuaciones BLEU, ROUGE y METEOR.  Estas métricas proporcionan una evaluación completa del rendimiento del modelo en diversas tareas y aspectos de la generación de texto.

* **Monitoreo de Activación:** La clase `ActivationMonitor` se utiliza para rastrear activaciones y gradientes durante el entrenamiento para la depuración y el análisis.  Esto permite a los desarrolladores comprender el comportamiento interno del modelo e identificar posibles problemas.

* **Estabilidad Numérica:** Se implementan varias comprobaciones y protecciones en todo el código para abordar posibles problemas de estabilidad numérica, como sujetar valores, manejar NaN e infinitos y recorte de gradientes.  Estos mecanismos ayudan a garantizar que el entrenamiento del modelo sea estable y converja a una solución óptima.

## Ejecución del Código

Para ejecutar el código, deberá instalar las dependencias requeridas, incluyendo PyTorch, Transformers, Datasets, NLTK, Scikit-learn y otras bibliotecas enumeradas en el código. También deberá descargar el conjunto de datos "TIGER-Lab/WebInstructSub". El bucle de entrenamiento principal se puede iniciar ejecutando `analysis_main.py`.

## Desarrollo futuro

Las posibles áreas para mayor desarrollo y mejora incluyen:

* **Ajuste de hiperparámetros:** Explorar diferentes configuraciones de hiperparámetros para optimizar el rendimiento del modelo.

* **Aumento de datos:** Aplicar técnicas de aumento de datos para aumentar el tamaño y la diversidad de los datos de entrenamiento.

* **Escalado del modelo:** Investigar los efectos de escalar el tamaño y la arquitectura del modelo en el rendimiento.

* **Arquitecturas alternativas:** Experimentar con diferentes opciones y componentes arquitectónicos.


Este README proporciona una descripción general detallada del modelo Liquid Foundation y su implementación. Para obtener detalles más específicos, consulte el código y los comentarios dentro de `simple.py` y `analysis_main.py`.
