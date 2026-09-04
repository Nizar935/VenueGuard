package com.nizar.venueguard.vision;

import org.bytedeco.opencv.opencv_core.Mat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public final class ImageLoader {

    private ImageLoader() {
    }

    public static Mat loadColour(Path imagePath) {
        Objects.requireNonNull(
                imagePath,
                "Image path cannot be null"
        );

        Path absolutePath =
                imagePath.toAbsolutePath().normalize();

        if (!Files.isRegularFile(absolutePath)) {
            throw new IllegalArgumentException(
                    "Image file does not exist: "
                            + absolutePath
            );
        }

        Mat image = imread(
                absolutePath.toString(),
                IMREAD_COLOR
        );

        if (image.empty()) {
            image.close();

            throw new IllegalArgumentException(
                    "OpenCV could not decode the image: "
                            + absolutePath
            );
        }

        return image;
    }
}
