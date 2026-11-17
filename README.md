# 🧠📱 Brain Tumor MRI Classifier — Android (Transfer Learning + LiteRT / TFLite)

Aplicación móvil nativa para Android desarrollada en Java que clasifica imágenes de resonancia magnética (MRI) cerebral en tiempo real mediante **Transfer Learning (MobileNetV2)** y ejecución local *offline* con **Google AI Edge LiteRT / TensorFlow Lite**.

---

## 🏗️ Proyecto Base y Atribución

Esta aplicación parte de la plantilla oficial de clasificación de imágenes de **TensorFlow Examples**:
* 🔗 **Repositorio original:** [tensorflow/examples](https://github.com/tensorflow/examples)
* 📂 **Módulo base:** [`lite/examples/image_classification/android_java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android_java)
* 📄 **Licencia base:** Apache License 2.0 (The TensorFlow Authors).

### ✨ Principales modificaciones respecto al proyecto base:
* 🔄 Sustitución del motor `ImageClassifier` (Task Vision) por inferencia directa mediante `org.tensorflow.lite.Interpreter` en `ImageClassifierHelper.java` para soportar modelos personalizados sin metadatos incrustados.
* ⚡ Migración del motor de inferencia a **Google AI Edge LiteRT (`litert:1.4.0`)** para garantizar compatibilidad con modelos exportados en TensorFlow 2.21.0 (`FULLY_CONNECTED` v12).
* 🎨 Ajuste de la tubería de preprocesamiento de tensores (`[0, 255]`) para alinearlo con la capa interna `mobilenet_v2.preprocess_input` del modelo entrenado.
* 🧬 Sustitución de los modelos por defecto por un modelo propio entrenado en Google Colab (`modelo_tumores.tflite` y `labels.txt`).

---

## 🚀 Características Principales

* 🔒 **Inferencia 100 % Offline y en Tiempo Real:** Procesamiento continuo de fotogramas mediante **CameraX** sin necesidad de conexión a Internet ni envío de imágenes médicas a servidores externos.
* 🩺 **Clasificación Multiclase (4 categorías):**
  1. **`glioma`**: Tumor originado en las células gliales del cerebro.
  2. **`meningioma`**: Tumor que surge en las meninges que recubren el encéfalo.
  3. **`notumor`**: Resonancia magnética cerebral sana sin presencia de masa tumoral.
  4. **`pituitary`**: Tumor localizado en la glándula pituitaria (observable principalmente en cortes sagitales y coronales).
* 🎛️ **Controles Dinámicos en Interfaz:** Panel desplegable inferior que permite ajustar en caliente el umbral mínimo de confianza (*Threshold*), el número máximo de resultados (*Max Results*) y el número de hilos de CPU (*Threads*).

---

## 🧪 Dataset y Entrenamiento (Google Colab)

* 📓 **Cuaderno de entrenamiento:** [Abrir en Google Colab (`Brain_Tumor_Android.ipynb`)][(https://colab.research.google.com/drive/1KsFD3Ji8aICny5hXAaIUc7dkICB4oiaN)](https://colab.research.google.com/drive/10JFQrHe20IXoe5Ih2wBmgDEKeGCuz4tX?usp=sharing)
* 🗂️ **Dataset:** [Brain Tumor Classification DataSet (Sartaj Bhuvaji)](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet), estandarizado en `Training` (80 % entrenamiento / 20 % validación, `seed=123`, `224x224x3`) y `Testing`.
* 🖼️ **Aumentación de Datos:** `RandomContrast(0.1)` y `RandomBrightness(0.1)` aplicados exclusivamente al conjunto de entrenamiento para preservar las coordenadas espaciales del tumor.
* 🧠 **Arquitectura de Transfer Learning:**
  * Base **MobileNetV2** preentrenada en ImageNet (`include_top=False`, pesos congelados).
  * Capa `tf.keras.applications.mobilenet_v2.preprocess_input` integrada en el grafo del modelo (reescala $[0, 255] \rightarrow [-1, 1]$).
  * Cabecera personalizada: `GlobalAveragePooling2D` $\rightarrow$ `Dense(128, relu)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(4, softmax)`.
* 📦 **Exportación:** Conversión a `.tflite` mediante `tf.lite.TFLiteConverter` con cuantización dinámica por defecto (`tf.lite.Optimize.DEFAULT`).

---

## 🛠️ Stack Tecnológico

* 🐍 **Entrenamiento:** Google Colab (GPU), Python 3, TensorFlow / Keras `2.21.0`, `scikit-learn`
* 🤖 **Aplicación Móvil:** Android Studio, Java 8, Gradle (`compileSdk 34`, `minSdk 24`)
* ⚙️ **Motor de Inferencia:** Google AI Edge LiteRT (`com.google.ai.edge.litert:litert:1.4.0`)
* 📷 **Captura de Cámara:** Android CameraX (`1.3.1`)

---

## 🧩 Arquitectura y Decisiones Técnicas Clave

1. **Inferencia Directa con `Interpreter` (`ImageClassifierHelper.java`):** Se desacopló la carga del modelo de `ImageClassifier` (Task Vision) para evitar errores por ausencia de metadatos TFLite, manteniendo la compatibilidad total con la interfaz gráfica (`Category` y `Classifications`).
2. **Migración a LiteRT `1.4.0`:** Resuelve la incompatibilidad del operador `FULLY_CONNECTED` (versión 12) generado por TensorFlow 2.21.0 frente a las librerías heredadas de `tensorflow-lite` 2.16, excluyendo los módulos antiguos en `build.gradle` para evitar conflictos de *namespace*.
3. **Alineación del Rango de Píxeles (`[0, 255]`):** Dado que el modelo exportado incorpora internamente `mobilenet_v2.preprocess_input`, el volcado al `ByteBuffer` mantiene los canales RGB en `[0, 255]` (`NORMALIZE_0_TO_1 = false`), evitando la doble normalización.

---

## 📁 Estructura del Proyecto

```text
├── app/
│   ├── build.gradle                              # Dependencias LiteRT y CameraX (noCompress 'tflite')
│   └── src/main/
│       ├── assets/
│       │   ├── modelo_tumores.tflite             # Modelo MobileNetV2 entrenado y cuantizado
│       │   └── labels.txt                        # Etiquetas en orden alfabético
│       └── java/org/tensorflow/lite/examples/imageclassification/
│           ├── ImageClassifierHelper.java        # Carga de modelo, preprocesamiento e inferencia
│           ├── fragments/CameraFragment.java     # Flujo de vídeo CameraX y control de UI
│           └── fragments/ClassificationResultAdapter.java
├── build.gradle
└── settings.gradle
```

---

## ▶️ Instalación y Ejecución

1. Clona este repositorio:
   ```bash
   git clone https://github.com/lauramarcelasape/brain-tumor-mri-classifier-android
   ```
2. Abre la carpeta raíz directamente en **Android Studio**.
3. Espera a que finalice la sincronización de Gradle (**Sync Project with Gradle Files**).
4. Conecta un dispositivo físico Android con **Depuración USB** activada (probado en OnePlus 7T Pro) y pulsa **Run (`▶`)**.

---

## 👩‍💻 Autora

* **Laura Marcela Sánchez Peláez** — Práctica 6 (PR6), *Deep Learning / Salud Digital*.
