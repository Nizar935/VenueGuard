package com.nizar.venueguard;

import com.nizar.venueguard.vision.ImageLoader;
import com.nizar.venueguard.vision.ImagePreprocessor;
import com.nizar.venueguard.vision.PreparedImage;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

public final class VenueGuard {

    private static final Path INPUT_IMAGE =
            Path.of("data", "input", "sample.jpg");

    private static final Path OUTPUT_IMAGE =
            Path.of("data", "output", "preprocessed.jpg");

    private static final int MAXIMUM_IMAGE_DIMENSION = 1280;

    private VenueGuard() {
    }

    public static void main(String[] args) throws IOException {
        System.out.println("VenueGuard starting...");
        System.out.println(
                "Java version: "
                        + System.getProperty("java.version")
        );

        ImagePreprocessor preprocessor =
                new ImagePreprocessor(MAXIMUM_IMAGE_DIMENSION);

        try (
                Mat original = ImageLoader.loadColour(INPUT_IMAGE);
                PreparedImage prepared =
                        preprocessor.preprocess(original)
        ) {
            printImageInformation(prepared);
            savePreview(prepared.grayscale());
        }
    }

    private static void printImageInformation(
            PreparedImage prepared
    ) {
        System.out.printf(
                "Original dimensions: %d × %d%n",
                prepared.originalWidth(),
                prepared.originalHeight()
        );

        System.out.printf(
                "Prepared dimensions: %d × %d%n",
                prepared.preparedWidth(),
                prepared.preparedHeight()
        );

        System.out.printf(
                "Scale factor: %.4f%n",
                prepared.scaleFactor()
        );

        System.out.println(
                "Colour channels: "
                        + prepared.colour().channels()
        );

        System.out.println(
                "Grayscale channels: "
                        + prepared.grayscale().channels()
        );
    }

    private static void savePreview(Mat image)
            throws IOException {
        Path absoluteOutputPath =
                OUTPUT_IMAGE.toAbsolutePath().normalize();

        Files.createDirectories(
                absoluteOutputPath.getParent()
        );

        boolean saved = imwrite(
                absoluteOutputPath.toString(),
                image
        );

        if (!saved) {
            throw new IOException(
                    "OpenCV could not save the preview image: "
                            + absoluteOutputPath
            );
        }

        System.out.println(
                "Preprocessed preview saved to: "
                        + absoluteOutputPath
        );

        System.out.println(
                "Image preprocessing completed successfully!"
        );
    }
}
