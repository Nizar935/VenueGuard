package com.nizar.venueguard.vision;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.Objects;

public final class PreparedImage implements AutoCloseable {

    private final Mat colour;
    private final Mat grayscale;
    private final int originalWidth;
    private final int originalHeight;
    private final double scaleFactor;

    public PreparedImage(
            Mat colour,
            Mat grayscale,
            int originalWidth,
            int originalHeight,
            double scaleFactor
    ) {
        this.colour = Objects.requireNonNull(
                colour,
                "Colour image cannot be null"
        );

        this.grayscale = Objects.requireNonNull(
                grayscale,
                "Grayscale image cannot be null"
        );

        if (colour.empty()) {
            throw new IllegalArgumentException(
                    "Colour image cannot be empty"
            );
        }

        if (grayscale.empty()) {
            throw new IllegalArgumentException(
                    "Grayscale image cannot be empty"
            );
        }

        if (scaleFactor <= 0.0 || scaleFactor > 1.0) {
            throw new IllegalArgumentException(
                    "Scale factor must be greater than 0 and at most 1"
            );
        }

        this.originalWidth = originalWidth;
        this.originalHeight = originalHeight;
        this.scaleFactor = scaleFactor;
    }

    public Mat colour() {
        return colour;
    }

    public Mat grayscale() {
        return grayscale;
    }

    public int originalWidth() {
        return originalWidth;
    }

    public int originalHeight() {
        return originalHeight;
    }

    public int preparedWidth() {
        return colour.cols();
    }

    public int preparedHeight() {
        return colour.rows();
    }

    public double scaleFactor() {
        return scaleFactor;
    }

    @Override
    public void close() {
        grayscale.close();
        colour.close();
    }
}
