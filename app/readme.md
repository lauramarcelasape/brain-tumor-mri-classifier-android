# 🧠 Clasificación de Tumores Cerebrales en Tiempo Real (Android + TFLite)

Aplicación móvil nativa para **Android** que realiza clasificación de imágenes de resonancia magnética (MRI) cerebral en tiempo real y de forma 100 % *offline* mediante **Transfer Learning (MobileNetV2)** y **Google AI Edge LiteRT / TensorFlow Lite**.

---

## 📋 Descripción del Proyecto

El sistema captura vídeo en directo desde la cámara del dispositivo mediante **CameraX**, preprocesa cada fotograma ($224 \times 224 \times 3$) y ejecuta inferencia local para clasificar resonancias magnéticas en 4 categorías clínicas:

| Clase (`labels.txt`) | Carpeta Original Dataset | Descripción |
| :--- | :--- | :--- |
| **`glioma`** | `glioma_tumor` | Tumor originado en las células gliales del cerebro. |
| **`meningioma`** | `meningioma_tumor` | Tumor que surge en las meninges que recubren el encéfalo. |
| **`notumor`** | `no_tumor` | Resonancia magnética cerebral sana sin presencia de masa tumoral. |
| **`pituitary`** | `pituitary_tumor` | Tumor localizado en la glándula pituitaria (visible en cortes sagitales/coronales). |

---

## 🔬 Entrenamiento del Modelo (Google Colab)

El modelo fue entrenado y exportado en **Google Colab** utilizando **TensorFlow / Keras 2.21.0**:

* **Dataset:** [Brain Tumor Classification DataSet (Sartaj Bhuvaji)](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet), particionado en 80 % entrenamiento y 20 % validación (`224x224`, `BATCH_SIZE = 16`).
* **Data Augmentation:** Transformaciones fotométricas ligeras (`RandomContrast(0.1)` y `RandomBrightness(0.1)`) aplicadas solo en entrenamiento para preservar las coordenadas anatómicas del tumor.
* **Arquitectura (Transfer Learning):**
  * Base congelada: `MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')`.
  * Preprocesamiento integrado en el grafo: `tf.keras.applications.mobilenet_v2.preprocess_input` (escala píxeles de $[0, 255]$ a $[-1, 1]$).
  * Cabecera de clasificación: `GlobalAveragePooling2D` $\rightarrow$ `Dense(128, relu)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(4, softmax)`.
* **Optimización y Exportación:** Entrenado con `Adam(lr=0.001)`, `EarlyStopping(patience=4)` y exportado mediante `TFLiteConverter` con cuantización dinámica (`tf.lite.Optimize.DEFAULT`).

---

## 📱 Arquitectura Android y Stack Tecnológico

* **Lenguaje / IDE:** Java 8 / Android Studio (`compileSdk 34`, `minSdk 24`).
* **Motor de Inferencia:** `com.google.ai.edge.litert:litert:1.4.0` (compatible con operadores `FULLY_CONNECTED` v12 exportados desde TF 2.21.0).
* **Captura de Cámara:** Android **CameraX 1.3.1**.
* **Inferencia directa (`ImageClassifierHelper.java`):**
  * Carga del modelo desde `assets` mediante `MappedByteBuffer` (`noCompress 'tflite'`).
  * Rotación según orientación del sensor y reescalado a $224 \times 224$.
  * Volcado de canales RGB en rango $[0, 255]$ a un `ByteBuffer` directo (`ByteOrder.nativeOrder()`) para alimentar la capa `preprocess_input` integrada en el `.tflite`.

---

## 📂 Estructura Principal del Proyecto

```text
├── app/src/main/
│   ├── assets/
│   │   ├── modelo_tumores.tflite     # Modelo MobileNetV2 cuantizado
│   │   └── labels.txt                # Etiquetas en orden alfabético
│   ├── java/org/tensorflow/lite/examples/imageclassification/
│   │   ├── ImageClassifierHelper.java # Motor de preprocesamiento e inferencia LiteRT
│   │   ├── MainActivity.java
│   │   └── fragments/
│   │       ├── CameraFragment.java
│   │       └── ClassificationResultAdapter.java
│   └── AndroidManifest.xml
├── app/build.gradle                  # Dependencias LiteRT 1.4.0 y CameraX
└── README.md
```

---

## 🚀 Instalación y Ejecución

1. Clona este repositorio:
   ```bash
   git clone <URL_DE_TU_REPOSITORIO>
   ```
2. Abre el proyecto en **Android Studio** y espera a que finalice la sincronización de **Gradle**.
3. Conecta un dispositivo Android físico con **Depuración USB** activada (o utiliza un emulador con cámara configurada).
4. Pulsa **Run (`▶`)** para compilar e instalar la aplicación.
5. *(Opcional)* Desliza hacia arriba el panel inferior de la app para ajustar en tiempo real el umbral de confianza (**Threshold**), el número máximo de resultados (**Max Results**) y los hilos de CPU (**Threads**).

---

## 👩‍💻 Autoría

* **Laura Marcela Sánchez Peláez** — *Deep Learning / Salud Digital (Práctica 6)*