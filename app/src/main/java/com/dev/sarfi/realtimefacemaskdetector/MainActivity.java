package com.dev.sarfi.realtimefacemaskdetector;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.common.util.concurrent.ListenableFuture;
import org.tensorflow.lite.Interpreter;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private PreviewView previewView;
    private TextView resultText;
    private Interpreter tfliteModel;
    private ExecutorService cameraExecutor;
    private static final int REQUEST_CAMERA_PERMISSION = 101;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        EdgeToEdge.enable(this);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        previewView = findViewById(R.id.previewView);
        resultText = findViewById(R.id.resultText);

        // Camera Permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        } else {
            startCamera();
        }

        try {
            tfliteModel = loadModelFile();
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Model Load Failed!", Toast.LENGTH_SHORT).show();
        }

        cameraExecutor = Executors.newSingleThreadExecutor();
    }

    private Interpreter loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("face_mask_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return new Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength));
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Select the camera (front or back)
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)  // Use front camera
                        .build();

                // Preview configuration
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Image analysis configuration
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(100, 100))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, image -> {
                    Bitmap bitmap = imageToBitmap(image);  // Convert ImageProxy to Bitmap
                    if (bitmap != null) {
                        detectMask(bitmap);  // Mask detection logic
                    }
                    image.close();  // Make sure to close the ImageProxy
                });

                // Unbind previous use cases and bind new ones
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

            } catch (ExecutionException | InterruptedException e) {
                Log.e("FaceMaskApp", "Error starting camera: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private Bitmap imageToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private void detectMask(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

        float[][] result = new float[1][1]; // Mask Detection Output
        tfliteModel.run(byteBuffer, result);

        runOnUiThread(() -> {
            if (result[0][0] > 0.5) {
                resultText.setText("✅ Mask Detected");
            } else {
                resultText.setText("❌ No Mask Detected");
            }
        });
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 100, 100, true);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 100 * 100 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[100 * 100];
        resizedBitmap.getPixels(intValues, 0, 100, 0, 0, 100, 100);

        int pixel = 0;
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        if (tfliteModel != null) {
            tfliteModel.close();
        }
    }
}