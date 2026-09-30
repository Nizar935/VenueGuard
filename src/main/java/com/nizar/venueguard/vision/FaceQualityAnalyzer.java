package com.nizar.venueguard.vision;

import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.BORDER_DEFAULT;
import static org.bytedeco.opencv.global.opencv_core.CV_64F;
import static org.bytedeco.opencv.global.opencv_core.meanStdDev;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_AREA;
import static org.bytedeco.opencv.global.opencv_imgproc.Laplacian;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public final class FaceQualityAnalyzer {

    private static final int MINIMUM_WIDTH = 160;
    private static final int MINIMUM_HEIGHT = 160;

    private static final double MINIMUM_BRIGHTNESS = 40.0;
    private static final double MAXIMUM_BRIGHTNESS = 215.0;

    private static final double MINIMUM_CONTRAST = 20.0;
    private static final double MINIMUM_SHARPNESS = 100.0;

    private static final int ANALYSIS_SIZE = 256;

    public FaceQualityReport analyse(Mat face) {
        validateFace(face);

        try (
                Mat grayscale = new Mat();
                Mat normalised = new Mat();
                Mat laplacian = new Mat();

                Mat intensityMean = new Mat();
                Mat intensityDeviation = new Mat();

                Mat laplacianMean = new Mat();
                Mat laplacianDeviation = new Mat();

                Size analysisDimensions =
                        new Size(ANALYSIS_SIZE, ANALYSIS_SIZE)
        ) {
            cvtColor(
                    face,
                    grayscale,
                    COLOR_BGR2GRAY
            );

            resize(
                    grayscale,
                    normalised,
                    analysisDimensions,
                    0.0,
                    0.0,
                    INTER_AREA
            );

            meanStdDev(
                    normalised,
                    intensityMean,
                    intensityDeviation
            );

            Laplacian(
                    normalised,
                    laplacian,
                    CV_64F,
                    3,
                    1.0,
                    0.0,
                    BORDER_DEFAULT
            );

            meanStdDev(
                    laplacian,
                    laplacianMean,
                    laplacianDeviation
            );

            double brightness =
                    readFirstValue(intensityMean);

            double contrast =
                    readFirstValue(intensityDeviation);

            double laplacianStandardDeviation =
                    readFirstValue(laplacianDeviation);

            double sharpness =
                    laplacianStandardDeviation
                            * laplacianStandardDeviation;

            List<String> problems = evaluate(
                    face,
                    brightness,
                    contrast,
                    sharpness
            );

            return new FaceQualityReport(
                    face.cols(),
                    face.rows(),
                    brightness,
                    contrast,
                    sharpness,
                    problems
            );
        }
    }

    private List<String> evaluate(
            Mat face,
            double brightness,
            double contrast,
            double sharpness
    ) {
        List<String> problems = new ArrayList<>();

        if (face.cols() < MINIMUM_WIDTH
                || face.rows() < MINIMUM_HEIGHT) {
            problems.add("face resolution is too small");
        }

        if (brightness < MINIMUM_BRIGHTNESS) {
            problems.add("image is too dark");
        }

        if (brightness > MAXIMUM_BRIGHTNESS) {
            problems.add("image is overexposed");
        }

        if (contrast < MINIMUM_CONTRAST) {
            problems.add("image contrast is too low");
        }

        if (sharpness < MINIMUM_SHARPNESS) {
            problems.add("image may be blurred");
        }

        return problems;
    }

    private double readFirstValue(Mat matrix) {
        try (
                DoubleIndexer values =
                        matrix.createIndexer()
        ) {
            return values.get(0, 0);
        }
    }

    private void validateFace(Mat face) {
        if (face == null || face.empty()) {
            throw new IllegalArgumentException(
                    "Face image cannot be null or empty"
            );
        }

        if (face.channels() != 3) {
            throw new IllegalArgumentException(
                    "Quality analysis requires a colour image"
            );
        }
    }
}
