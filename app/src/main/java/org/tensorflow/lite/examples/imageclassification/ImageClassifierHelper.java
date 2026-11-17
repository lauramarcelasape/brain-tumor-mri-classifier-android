package org.tensorflow.lite.examples.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.SystemClock;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.vision.classifier.Classifications;

public class ImageClassifierHelper {
    private static final String TAG = "ImageClassifierHelper";

    // Archivos en app/src/main/assets/
    private static final String MODEL_NAME = "modelo_tumores.tflite";
    private static final String LABELS_NAME = "labels.txt";

    // true = normaliza píxeles a [0, 1] (dividiendo entre 255.0)
    // false = deja píxeles en [0, 255] (si tu modelo ya tiene capa Rescaling interna)
    private static final boolean NORMALIZE_0_TO_1 = true;

    private Interpreter interpreter;
    private List<String> labels;
    private int inputImageWidth;
    private int inputImageHeight;
    private int numThreads = 2;
    private float threshold = 0.5f;
    private int maxResults = 1;
    private final Context context;
    private final ClassifierListener imageClassifierListener;

    public ImageClassifierHelper(
            float threshold,
            int numThreads,
            int maxResults,
            int currentDelegate,
            int currentModel,
            Context context,
            ClassifierListener imageClassifierListener) {
        this.threshold = threshold;
        this.numThreads = numThreads;
        this.maxResults = maxResults;
        this.context = context;
        this.imageClassifierListener = imageClassifierListener;
        setupImageClassifier();
    }

    public static ImageClassifierHelper create(Context context, ClassifierListener listener) {
        return new ImageClassifierHelper(0.5f, 2, 1, 0, 0, context, listener);
    }

    public void setupImageClassifier() {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(numThreads);
            interpreter = new Interpreter(loadModelFile(context, MODEL_NAME), options);

            int[] inputShape = interpreter.getInputTensor(0).shape();
            inputImageHeight = inputShape[1];
            inputImageWidth = inputShape[2];

            labels = loadLabels(context, LABELS_NAME);
        } catch (IOException e) {
            if (imageClassifierListener != null) {
                imageClassifierListener.onError("Error cargando assets: " + e.getMessage());
            }
            Log.e(TAG, "Error inicializando TFLite desde assets", e);
        }
    }

    public void classify(Bitmap image, int imageRotation) {
        if (interpreter == null) {
            setupImageClassifier();
        }
        if (interpreter == null) return;

        long inferenceTime = SystemClock.uptimeMillis();

        Bitmap rotatedBitmap = rotateBitmap(image, imageRotation);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(rotatedBitmap, inputImageWidth, inputImageHeight, true);
        ByteBuffer inputBuffer = convertBitmapToByteBuffer(resizedBitmap);

        float[][] output = new float[1][labels.size()];
        interpreter.run(inputBuffer, output);

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime;

        List<Category> categories = new ArrayList<>();
        for (int i = 0; i < labels.size(); i++) {
            float score = output[0][i];
            if (score >= threshold) {
                categories.add(Category.create(labels.get(i), labels.get(i), score, i));
            }
        }
        Collections.sort(categories, (a, b) -> Float.compare(b.getScore(), a.getScore()));
        if (categories.size() > maxResults) {
            categories = categories.subList(0, maxResults);
        }
        final List<Category> finalCategories = categories;

        Classifications classification = new Classifications() {
            @Override
            public List<Category> getCategories() {
                return finalCategories;
            }

            @Override
            public int getHeadIndex() {
                return 0;
            }
        };
        List<Classifications> results = Collections.singletonList(classification);

        if (imageClassifierListener != null) {
            imageClassifierListener.onResults(results, inferenceTime);
        }
    }

    private Bitmap rotateBitmap(Bitmap source, int angle) {
        if (angle == 0) return source;
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * inputImageWidth * inputImageHeight * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputImageWidth * inputImageHeight];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < inputImageHeight; ++i) {
            for (int j = 0; j < inputImageWidth; ++j) {
                final int val = intValues[pixel++];
                float r = ((val >> 16) & 0xFF);
                float g = ((val >> 8) & 0xFF);
                float b = (val & 0xFF);

                if (NORMALIZE_0_TO_1) {
                    byteBuffer.putFloat(r / 255.0f);
                    byteBuffer.putFloat(g / 255.0f);
                    byteBuffer.putFloat(b / 255.0f);
                } else {
                    byteBuffer.putFloat(r);
                    byteBuffer.putFloat(g);
                    byteBuffer.putFloat(b);
                }
            }
        }
        return byteBuffer;
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        android.content.res.AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabels(Context context, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            if (!line.trim().isEmpty()) {
                labelList.add(line.trim());
            }
        }
        reader.close();
        return labelList;
    }

    public void clearImageClassifier() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }

    public int getNumThreads() { return numThreads; }
    public void setNumThreads(int numThreads) { this.numThreads = numThreads; }
    public int getMaxResults() { return maxResults; }
    public void setMaxResults(int maxResults) { this.maxResults = maxResults; }
    public float getThreshold() { return threshold; }
    public void setThreshold(float threshold) { this.threshold = threshold; }
    public void setCurrentDelegate(int currentDelegate) { }
    public void setCurrentModel(int currentModel) { }

    public interface ClassifierListener {
        void onError(String error);
        void onResults(List<Classifications> results, long inferenceTime);
    }
}