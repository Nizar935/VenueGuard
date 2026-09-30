package com.nizar.venueguard.vision;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;

import java.util.Objects;

public final class FaceCropper {

    private FaceCropper() {
    }

    public static Mat crop(
            Mat source,
            FaceDetection detection,
            double paddingRatio
    ) {
        Objects.requireNonNull(source);
        Objects.requireNonNull(detection);

        if (source.empty()) {
            throw new IllegalArgumentException(
                    "Source image cannot be empty"
            );
        }

        if (!Double.isFinite(paddingRatio)
                || paddingRatio < 0
                || paddingRatio > 1) {
            throw new IllegalArgumentException(
                    "Padding ratio must be between 0 and 1"
            );
        }

        double horizontalPadding =
                detection.width() * paddingRatio;

        double verticalPadding =
                detection.height() * paddingRatio;

        int left = clamp(
                (int) Math.floor(
                        detection.x() - horizontalPadding
                ),
                0,
                source.cols() - 1
        );

        int top = clamp(
                (int) Math.floor(
                        detection.y() - verticalPadding
                ),
                0,
                source.rows() - 1
        );

        int right = clamp(
                (int) Math.ceil(
                        detection.x()
                                + detection.width()
                                + horizontalPadding
                ),
                left + 1,
                source.cols()
        );

        int bottom = clamp(
                (int) Math.ceil(
                        detection.y()
                                + detection.height()
                                + verticalPadding
                ),
                top + 1,
                source.rows()
        );

        try (
                Rect region = new Rect(
                        left,
                        top,
                        right - left,
                        bottom - top
                );

                Mat temporaryView =
                        new Mat(source, region)
        ) {
            return temporaryView.clone();
        }
    }

    private static int clamp(
            int value,
            int minimum,
            int maximum
    ) {
        return Math.max(
                minimum,
                Math.min(value, maximum)
        );
    }
}
