package com.nizar.venueguard.vision;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_AREA;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public final class ImagePreprocessor {

    private static final int MINIMUM_IMAGE_DIMENSION = 100;

    private final int maximumImageDimension;

    public ImagePreprocessor(int maximumImageDimension) {
        if (maximumImageDimension < MINIMUM_IMAGE_DIMENSION) {
            throw new IllegalArgumentException(
                    "Maximum image dimension must be at least "
                            + MINIMUM_IMAGE_DIMENSION
            );
        }

        this.maximumImageDimension = maximumImageDimension;
    }

    public PreparedImage preprocess(Mat source) {
        validateSource(source);

        int originalWidth = source.cols();
        int originalHeight = source.rows();

        int longestDimension = Math.max(
                originalWidth,
                originalHeight
        );

        double scaleFactor = Math.min(
                1.0,
                (double) maximumImageDimension / longestDimension
        );

        int preparedWidth = Math.max(
                1,
                (int) Math.round(originalWidth * scaleFactor)
        );

        int preparedHeight = Math.max(
                1,
                (int) Math.round(originalHeight * scaleFactor)
        );

        Mat colour = new Mat();
        Mat grayscale = new Mat();

        try {
            if (scaleFactor < 1.0) {
                resizeImage(
                        source,
                        colour,
                        preparedWidth,
                        preparedHeight
                );
            } else {
                source.copyTo(colour);
            }

            cvtColor(
                    colour,
                    grayscale,
                    COLOR_BGR2GRAY
            );

            return new PreparedImage(
                    colour,
                    grayscale,
                    originalWidth,
                    originalHeight,
                    scaleFactor
            );
        } catch (RuntimeException exception) {
            grayscale.close();
            colour.close();
            throw exception;
        }
    }

    private void validateSource(Mat source) {
        if (source == null) {
            throw new IllegalArgumentException(
                    "Source image cannot be null"
            );
        }

        if (source.empty()) {
            throw new IllegalArgumentException(
                    "Source image cannot be empty"
            );
        }

        if (source.cols() < MINIMUM_IMAGE_DIMENSION
                || source.rows() < MINIMUM_IMAGE_DIMENSION) {
            throw new IllegalArgumentException(
                    "Image must be at least "
                            + MINIMUM_IMAGE_DIMENSION
                            + " × "
                            + MINIMUM_IMAGE_DIMENSION
                            + " pixels. Received: "
                            + source.cols()
                            + " × "
                            + source.rows()
            );
        }

        if (source.channels() != 3) {
            throw new IllegalArgumentException(
                    "Expected a three-channel colour image, but received "
                            + source.channels()
                            + " channels"
            );
        }
    }

    private void resizeImage(
            Mat source,
            Mat destination,
            int width,
            int height
    ) {
        try (Size targetSize = new Size(width, height)) {
            resize(
                    source,
                    destination,
                    targetSize,
                    0.0,
                    0.0,
                    INTER_AREA
            );
        }
    }
}
