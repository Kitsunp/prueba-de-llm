# Modelo Liquid Foundation para Generación de Texto

Este repositorio contiene el código para un modelo de generación de texto llamado "Liquid Foundation". El modelo está diseñado con un enfoque en la eficiencia y el rendimiento, incorporando varias técnicas avanzadas y optimizaciones arquitectónicas.  Su objetivo es generar texto de alta calidad de manera eficiente, abordando las limitaciones de los modelos tradicionales de generación de texto, como la complejidad computacional y la dificultad para capturar dependencias a largo plazo.

## Arquitectura del Modelo (Detalle Extendido)

La arquitectura central del modelo, definida en `simple.py`, se basa en una estructura similar a la de un transformador, pero con mejoras significativas para optimizar el rendimiento y la eficiencia.  A continuación, se detallan los componentes clave con una explicación exhaustiva de cada capa:

### Liquid Embedding (Incrustación Líquida)

La capa `LiquidEmbedding` es el primer paso en el procesamiento de la entrada de texto.  Su función principal es convertir cada token de entrada en una representación vectorial densa, también conocida como incrustación o embedding.  Esta capa no se limita a una simple búsqueda en una tabla de embeddings, sino que incorpora elementos adicionales para enriquecer la representación de los tokens.

**Incrustación de Tokens y Posiciones:**

Inicialmente, cada token se incrusta utilizando `nn.Embedding(vocab_size, embed_dim)`, que es una capa de embedding estándar.  Esto asigna a cada token un vector de `embed_dim` dimensiones.  Simultáneamente, se agrega información posicional utilizando `nn.Embedding(max_length, embed_dim)`.  Esta incrustación posicional proporciona al modelo información sobre la posición de cada token en la secuencia, lo cual es fundamental para el procesamiento de secuencias, ya que el orden de las palabras es crucial para el significado.

**Capas Convolucionales:**

Después de las incrustaciones iniciales, se aplican dos capas convolucionales 1D (`nn.Conv1d`).  Estas capas convolucionales escanean la secuencia de embeddings, aplicando un filtro que captura características locales y dependencias contextuales entre tokens adyacentes.  La función de activación GELU (`F.gelu`) se aplica después de cada capa convolucional para introducir no linealidad, permitiendo que el modelo aprenda relaciones más complejas entre los tokens.

**Normalización y Dropout:**

La capa `nn.LayerNorm` normaliza las embeddings resultantes, estabilizando el entrenamiento y mejorando la convergencia.  La capa `nn.Dropout` aplica dropout, una técnica de regularización que previene el sobreajuste al desactivar aleatoriamente un porcentaje de las neuronas durante el entrenamiento.

**Compresión Adaptativa y Transformada de Fourier Rápida (FFT):**

Para optimizar la eficiencia, la capa `LiquidEmbedding` implementa un mecanismo de compresión adaptativa.  Primero, se aplica la Transformada de Fourier Rápida (FFT) a la secuencia de embeddings.  La FFT descompone la secuencia en sus componentes de frecuencia, lo que permite al modelo analizar la complejidad de la secuencia.  La complejidad se estima basándose en la magnitud de los componentes de frecuencia.  Secuencias más complejas tendrán componentes de frecuencia más altos.

En función de la complejidad estimada, el modelo ajusta dinámicamente la tasa de compresión.  Para secuencias menos complejas, se aplica una mayor compresión, reduciendo la dimensionalidad de las embeddings y optimizando el uso de recursos.  Para secuencias más complejas, se aplica una menor compresión, preservando más información de la incrustación.

**Proyección y Reconstrucción:**

Después de la compresión, se aplica una capa lineal (`nn.Linear`) para proyectar las embeddings comprimidas de vuelta al espacio original.  Finalmente, se calcula una pérdida de reconstrucción que compara las embeddings reconstruidas con las embeddings originales antes de la compresión.  Esta pérdida de reconstrucción fomenta que el modelo aprenda representaciones comprimidas que conserven la mayor cantidad de información posible.

### Enhanced Local Attention with Group Query Attention (GQA)

La capa `EnhancedLocalAttentionWithGQA` es un componente crucial del modelo que calcula la atención entre los tokens de la secuencia.  Combina atención local con Group Query Attention (GQA) y Rotary Position Embeddings (RoPE) para lograr una atención eficiente y efectiva.

**Atención Local:**

La atención local restringe el cálculo de la atención a una ventana alrededor de cada token.  Esto reduce significativamente la complejidad computacional en comparación con la atención global, que considera todos los pares de tokens en la secuencia.  El tamaño de la ventana es un hiperparámetro que controla el alcance de la atención local.

**Group Query Attention (GQA):**

GQA divide las queries en grupos y calcula la atención para cada grupo por separado.  Esto reduce aún más la complejidad computacional y la huella de memoria, lo que permite escalar el modelo a secuencias más largas.  El número de grupos es un hiperparámetro que controla la granularidad de GQA.

**Rotary Position Embeddings (RoPE):**

RoPE codifica la información posicional directamente en las incrustaciones de queries y keys.  Esto permite que el modelo tenga en cuenta la posición relativa de los tokens al calcular la atención, lo cual es fundamental para el procesamiento de secuencias.  RoPE es una alternativa más eficiente a los embeddings posicionales absolutos, ya que no requiere una matriz de embeddings separada.

**Cálculo de la Atención:**

El cálculo de la atención se realiza utilizando la función `flash_attn_func`, que es una implementación optimizada de la atención.  La función calcula los pesos de atención para cada par de query y key dentro de la ventana local y los grupos de queries.  Estos pesos de atención se utilizan para combinar los valores (values) correspondientes, produciendo una representación contextualizada de cada token.

### Mixture of Experts (MoE) - (Mezcla de Expertos)

La capa `MoELayer` implementa una estrategia de Mixture of Experts (MoE), donde la entrada se enruta a un subconjunto de "expertos", cada uno de los cuales es una red neuronal independiente especializada en un aspecto particular de los datos.

**Routing de Expertos (Enrutamiento):**

El enrutamiento de los tokens de entrada a los expertos se realiza mediante una "puerta" (gate).  La puerta es una capa lineal que produce un conjunto de pesos para cada experto.  Estos pesos se interpretan como probabilidades, y se utilizan para determinar a qué expertos se enruta cada token.  El enrutamiento dinámico permite que el modelo adapte la selección de expertos a los datos de entrada, lo que resulta en un procesamiento más especializado.

**Expertos:**

Cada experto es una red neuronal lineal que transforma la entrada.  El número de expertos y la dimensionalidad de sus salidas son hiperparámetros del modelo.  La idea detrás de MoE es que cada experto se especialice en un subconjunto del espacio de entrada, lo que permite que el modelo capture una gama más amplia de patrones en los datos.

**Agregación de Salidas de Expertos:**

Las salidas de los expertos seleccionados se combinan utilizando los pesos generados por la puerta.  Esta combinación ponderada produce una representación final que integra las contribuciones de los expertos más relevantes para la entrada dada.

**Regularización de Uso de Expertos:**

Para evitar que un solo experto domine el proceso y fomentar la especialización entre los expertos, se aplica una regularización de uso.  Esta regularización penaliza el uso excesivo de un experto, asegurando que todos los expertos contribuyan al aprendizaje y que el modelo pueda capturar una gama más amplia de patrones en los datos.  Además, se calcula la entropía de los pesos de la puerta y se utiliza como una forma de regularización para fomentar una distribución más uniforme del uso de expertos.

### Deformable Convolution (Convolución Deformable) - `DeformableConv1d`

La capa `DeformableConv1d` implementa una convolución deformable unidimensional.  A diferencia de las convoluciones tradicionales, que aplican un filtro fijo a la entrada, las convoluciones deformables permiten que el modelo aprenda desplazamientos para los puntos de muestreo del filtro.  Esto permite que el modelo adapte el filtro a la entrada, capturando características más relevantes y contextuales.

**Cálculo de Offsets (Desplazamientos):**

Los desplazamientos para los puntos de muestreo se calculan mediante una capa convolucional separada (`offset_conv`).  Esta capa toma la entrada y produce un conjunto de desplazamientos para cada punto de muestreo del filtro.  Los desplazamientos se aprenden durante el entrenamiento, lo que permite que el modelo optimice la forma del filtro para la tarea específica.

**Muestreo Deformable:**

Utilizando los desplazamientos calculados, el modelo muestrea la entrada en ubicaciones deformadas.  Esto permite que el filtro capture información de diferentes partes de la entrada, adaptándose a la forma y la estructura de los datos.

**Convolución:**

Después del muestreo deformable, se aplica una convolución estándar con un tamaño de kernel de 1.  Esta convolución combina las muestras deformadas para producir la salida final.

### Optimized Gated Convolution (Convolución Optimizada con Puerta) - `OptimizedGatedConvolution`

La capa `OptimizedGatedConvolution` combina una convolución deformable con un mecanismo de puerta.  Esto permite que el modelo controle el flujo de información a través de la red, aprendiendo qué características son importantes para cada entrada.

**Convolución Deformable:**

Se utiliza una capa `DeformableConv1d` para capturar características deformables de la entrada.

**Mecanismo de Puerta:**

La salida de la convolución deformable se divide en dos partes: una parte "principal" y una parte "puerta".  La parte principal se pasa a través de una función de activación GELU.  La parte puerta se pasa a través de una función sigmoide, produciendo un conjunto de pesos entre 0 y 1.

**Combinación con Puerta:**

La parte principal y la parte puerta se combinan multiplicando elemento por elemento.  Esto permite que la puerta controle la cantidad de información que fluye de la parte principal a la siguiente capa.  Los pesos de la puerta se aprenden durante el entrenamiento, lo que permite que el modelo optimice el flujo de información para la tarea específica.

**Normalización y Dropout:**

Después de la combinación con puerta, se aplica normalización y dropout para estabilizar el entrenamiento y prevenir el sobreajuste.

### Enhanced LSTM (LSTM Mejorado) - `EnhancedLSTM`

La capa `EnhancedLSTM` utiliza un módulo LSTM (Long Short-Term Memory) para modelar dependencias a largo plazo en la secuencia de entrada.  El LSTM es una red neuronal recurrente que mantiene un estado oculto que se actualiza en cada paso de tiempo.  Esto permite que el LSTM almacene información de pasos de tiempo anteriores y la utilice para informar el procesamiento de los pasos de tiempo actuales.

**LSTM Estándar:**

Se utiliza una capa `nn.LSTM` estándar con parámetros configurables, como el tamaño oculto y el número de capas.

**Capa de Salida:**

La salida del LSTM se pasa a través de una capa lineal seguida de una función de activación GELU.  Esto permite que el LSTM produzca una representación más compleja de la secuencia de entrada.

**Conexión Residual:**

Se utiliza una conexión residual para combinar la salida del LSTM con la entrada original.  Esto ayuda a estabilizar el entrenamiento y mejorar el flujo de gradientes.

### Improved Transformer Block (Bloque Transformador Mejorado) - `ImprovedTransformerBlock`

El `ImprovedTransformerBlock` es el bloque básico del modelo.  Combina las capas descritas anteriormente en una estructura de transformador con normalización previa a la capa (Pre-LN).

**Normalización Pre-Capa (Pre-LN):**

En Pre-LN, la normalización de capa se aplica antes de la atención y las capas de feedforward.  Esto se ha demostrado que mejora la estabilidad del entrenamiento, especialmente para modelos profundos.

**Atención, Convolución, MoE y Feedforward:**

El bloque aplica las capas `EnhancedLocalAttentionWithGQA`, `OptimizedGatedConvolution`, `MoELayer` y una red feedforward en secuencia.  Cada capa se sigue de una capa de normalización y dropout.

**Conexiones Residuales:**

Se utilizan conexiones residuales alrededor de cada capa para mejorar el flujo de gradientes y estabilizar el entrenamiento.

**Clipping de Gradientes:**

El bloque implementa clipping de gradientes para evitar que los gradientes se vuelvan demasiado grandes durante el entrenamiento, lo que podría causar inestabilidad.

### Bidirectional Encoder (Codificador Bidireccional) - `BidirectionalEncoder`

El `BidirectionalEncoder` es responsable de codificar la secuencia de entrada en una representación contextualizada.  Utiliza múltiples capas `ImprovedTransformerBlock` para procesar la entrada en ambas direcciones (hacia adelante y hacia atrás).

**Incrustación:**

La entrada se incrusta utilizando la capa `LiquidEmbedding`.

**Capas del Transformador:**

Se aplican múltiples capas `ImprovedTransformerBlock` a la secuencia de embeddings.

**Normalización y Dropout:**

La salida del codificador se normaliza y se le aplica dropout.

### LiquidFoundationModelOptimized

Esta es la clase principal del modelo, que combina el codificador bidireccional con un decodificador y una memoria externa LSTM.

**Codificador:**

Se utiliza un `BidirectionalEncoder` para codificar la secuencia de entrada.

**Decodificador:**

El decodificador utiliza una estructura similar al codificador, pero con atención causal, lo que significa que solo atiende a los tokens anteriores en la secuencia.

**Memoria Externa LSTM:**

Se utiliza un `EnhancedLSTM` como memoria externa para almacenar información contextual de la secuencia de entrada.

**Capa de Salida:**

La capa de salida es una capa lineal que proyecta la salida del decodificador al tamaño del vocabulario.

**Generación:**

El modelo implementa un método `generate` para generar texto.  Este método utiliza un proceso de decodificación autoregresiva, donde el modelo genera un token a la vez, condicionando cada nuevo token a los tokens generados previamente.

## Entrenamiento y Evaluación (Más Detallado)

El proceso de entrenamiento y evaluación, implementado en `analysis_main.py`, es crucial para el rendimiento del modelo.  Se describe a continuación con mayor detalle:

### Carga y Preprocesamiento del Conjunto de Datos

El código utiliza la biblioteca `datasets` para cargar el conjunto de datos "TIGER-Lab/WebInstructSub".  Este conjunto de datos probablemente contiene pares de pregunta-respuesta o instrucciones-respuesta que se utilizan para entrenar el modelo en la generación de texto.

El preprocesamiento de los datos es un paso crucial para garantizar que el modelo reciba datos limpios y consistentes.  Los pasos de preprocesamiento incluyen:

* **Limpieza de texto:** Eliminación de caracteres innecesarios, como espacios en blanco adicionales o caracteres de control.
* **Eliminación de escapes HTML:** Conversión de entidades HTML, como `&amp;`, a sus caracteres correspondientes.
* **Normalización Unicode:** Conversión de diferentes representaciones Unicode del mismo carácter a una forma canónica.
* **Eliminación de URL:** Eliminación de URLs del texto, ya que pueden no ser relevantes para la tarea de generación de texto.
* **Normalización de espacios en blanco:**  Asegurar que haya un solo espacio entre las palabras y eliminar espacios en blanco al principio y al final del texto.

### Tokenización

La tokenización es el proceso de convertir el texto en una secuencia de tokens, que son las unidades básicas que el modelo procesa.  El código utiliza `LEDTokenizer`, un tokenizador pre-entrenado, para este propósito.  Se agregan tokens especiales a la secuencia para indicar el inicio de la secuencia (`BOS`), el final de la secuencia (`EOS`), el relleno (`PAD`) y la separación entre diferentes partes de la entrada (`SEP`).

### Funciones de Pérdida

El modelo se entrena utilizando una combinación de diferentes funciones de pérdida:

* **Entropía cruzada:** Mide la diferencia entre la distribución de probabilidad predicha por el modelo y la distribución real de los tokens en los datos de entrenamiento.
* **Pérdida focal:**  Una variante de la entropía cruzada que se centra en ejemplos difíciles de clasificar, lo que puede mejorar el rendimiento del modelo en casos complejos.
* **Regularización de reconstrucción:**  Un término de regularización que fomenta que el modelo aprenda representaciones que se pueden reconstruir con precisión a partir de sus componentes comprimidos.
* **Regularización de entropía:**  Un término de regularización que fomenta que el modelo utilice todos los expertos de manera uniforme, evitando que un solo experto domine el proceso.

### Optimizador y Programador

El optimizador `AdamW` se utiliza para actualizar los pesos del modelo durante el entrenamiento.  `AdamW` es una variante del optimizador Adam que incluye la regularización de peso L2, lo que ayuda a prevenir el sobreajuste.

El programador de tasa de aprendizaje `CosineAnnealingWarmRestarts` ajusta la tasa de aprendizaje de forma cíclica durante el entrenamiento.  Esto puede ayudar a que el modelo escape de mínimos locales y converja a una mejor solución.

El escalado de gradientes se utiliza para el entrenamiento de precisión mixta, lo que permite que el modelo se entrene con mayor eficiencia utilizando tipos de datos de punto flotante de menor precisión.

### Métricas

El código calcula una amplia gama de métricas para evaluar el rendimiento del modelo:

* **Precisión de tokens:**  La proporción de tokens predichos correctamente.
* **Precisión de secuencia:**  La proporción de secuencias predichas correctamente.
* **Precisión top-k:**  La proporción de veces que el token correcto se encuentra entre las k predicciones principales del modelo.
* **Distinct-n:**  Mide la diversidad del texto generado, calculando la proporción de n-gramas únicos.
* **Longitud promedio de secuencia:**  La longitud promedio de las secuencias generadas.
* **Perplejidad:**  Una medida de qué tan bien el modelo predice la siguiente palabra en una secuencia.  Una perplejidad más baja indica un mejor rendimiento.
* **BLEU:**  Una métrica que compara el texto generado con un conjunto de textos de referencia.
* **ROUGE:**  Un conjunto de métricas que evalúan la calidad de los resúmenes generados.
* **METEOR:**  Una métrica que considera sinónimos y paráfrasis al comparar el texto generado con textos de referencia.

### Monitoreo de Activación

La clase `ActivationMonitor` se utiliza para registrar las activaciones y los gradientes de las diferentes capas del modelo durante el entrenamiento.  Esto proporciona información valiosa para la depuración y el análisis del modelo, permitiendo a los desarrolladores identificar posibles problemas, como gradientes que desaparecen o explotan, o activaciones anómalas.

### Estabilidad Numérica

El código incorpora varias técnicas para garantizar la estabilidad numérica durante el entrenamiento:

* **Sujeción de valores:**  Limitar el rango de valores de las activaciones y los gradientes para evitar valores extremos que puedan causar inestabilidad.
* **Manejo de NaN e infinitos:**  Detectar y manejar valores no finitos (NaN e infinitos) que puedan surgir durante el entrenamiento.
* **Recorte de gradientes:**  Limitar la norma de los gradientes para evitar que se vuelvan demasiado grandes, lo que puede causar inestabilidad en el entrenamiento.


## Ejecución del Código (Más Detallado)

Para ejecutar el código, siga estos pasos:

1. **Clonar el repositorio:**  Clone este repositorio en su máquina local.
2. **Crear un entorno virtual:**  Se recomienda crear un entorno virtual para aislar las dependencias del proyecto.
3. **Instalar las dependencias:**  Instale las bibliotecas necesarias utilizando el archivo `requirements.txt` (si se proporciona) o instalando manualmente las bibliotecas mencionadas en el código, como PyTorch, Transformers, Datasets, NLTK, Scikit-learn, etc.
4. **Descargar el conjunto de datos:**  Descargue el conjunto de datos "TIGER-Lab/WebInstructSub" y colóquelo en la ubicación adecuada según el código.
5. **Ejecutar el script principal:**  Ejecute el script `analysis_main.py` para iniciar el proceso de entrenamiento y evaluación.

## Desarrollo Futuro

Existen varias áreas de desarrollo futuro que podrían mejorar aún más el modelo:

* **Ajuste de hiperparámetros:**  Realizar una búsqueda de hiperparámetros más exhaustiva para encontrar la configuración óptima para el modelo y el conjunto de datos.
* **Aumento de datos:**  Aplicar técnicas de aumento de datos para aumentar el tamaño y la diversidad del conjunto de datos de entrenamiento, lo que podría mejorar la generalización del modelo.
* **Escalado del modelo:**  Explorar el escalado del modelo, aumentando el número de capas, cabezas de atención o expertos, para ver si esto mejora el rendimiento.
* **Arquitecturas alternativas:**  Experimentar con diferentes arquitecturas o componentes, como mecanismos de atención alternativos o diferentes tipos de capas convolucionales.
* **Integración con otras tareas:**  Adaptar el modelo para otras tareas de procesamiento del lenguaje natural, como traducción automática o resumen de texto.


Este README proporciona una descripción detallada y técnica del modelo Liquid Foundation y su implementación.  Para obtener información aún más específica, consulte el código y los comentarios en los archivos `simple.py` y `analysis_main.py`.
