package com.nizar.venueguard;

import com.nizar.venueguard.vision.FaceCropper;
import com.nizar.venueguard.vision.FaceDetection;
import com.nizar.venueguard.vision.FaceEmbedding;
import com.nizar.venueguard.vision.FaceQualityAnalyzer;
import com.nizar.venueguard.vision.FaceQualityReport;
import com.nizar.venueguard.vision.FaceRenderer;
import com.nizar.venueguard.vision.ImageLoader;
import com.nizar.venueguard.vision.ImagePreprocessor;
import com.nizar.venueguard.vision.PreparedImage;
import com.nizar.venueguard.vision.SFaceEmbedder;
import com.nizar.venueguard.vision.YuNetFaceDetector;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

public final class VenueGuard {

    private static final Path INPUT_IMAGE =
            Path.of(
                    "data",
                    "input",
                    "sample.jpg"
            );

    private static final Path DETECTION_OUTPUT =
            Path.of(
                    "data",
                    "output",
                    "detected-faces.jpg"
            );

    private static final Path FACE_DETECTION_MODEL =
            Path.of(
                    "models",
                    "face_detection_yunet_2023mar.onnx"
            );

    private static final Path FACE_RECOGNITION_MODEL =
            Path.of(
                    "models",
                    "face_recognition_sface_2021dec.onnx"
            );

    private static final int MAXIMUM_IMAGE_DIMENSION = 480;

    private VenueGuard() {
    }

    public static void main(String[] args) throws IOException {
        System.out.println("VenueGuard starting...");

        ImagePreprocessor preprocessor =
                new ImagePreprocessor(
                        MAXIMUM_IMAGE_DIMENSION
                );

        try (
                Mat original =
                        ImageLoader.loadColour(INPUT_IMAGE);

                PreparedImage prepared =
                        preprocessor.preprocess(original);

                YuNetFaceDetector detector =
                        new YuNetFaceDetector(
                                FACE_DETECTION_MODEL
                        );

                SFaceEmbedder embedder =
                        new SFaceEmbedder(
                                FACE_RECOGNITION_MODEL
                        )
        ) {
            List<FaceDetection> detections =
                    detector.detect(
                            prepared.colour()
                    );

            printResults(
                    prepared,
                    detections
            );

            try (
                    Mat rendered = FaceRenderer.render(
                            prepared.colour(),
                            detections
                    )
            ) {
                saveImage(
                        rendered,
                        DETECTION_OUTPUT
                );
            }

            processFaces(
                    original,
                    prepared.scaleFactor(),
                    detections,
                    embedder
            );
        }
    }

    private static void printResults(
            PreparedImage prepared,
            List<FaceDetection> detections
    ) {
        System.out.printf(
                "Original dimensions: %d × %d%n",
                prepared.originalWidth(),
                prepared.originalHeight()
        );

        System.out.printf(
                "Detection dimensions: %d × %d%n",
                prepared.preparedWidth(),
                prepared.preparedHeight()
        );

        System.out.println(
                "Faces detected: " + detections.size()
        );

        for (
                int index = 0;
                index < detections.size();
                index++
        ) {
            FaceDetection detection =
                    detections.get(index);

            System.out.printf(
                    "Face %d: x=%.1f, y=%.1f, "
                            + "width=%.1f, height=%.1f, "
                            + "confidence=%.2f%%%n",
                    index + 1,
                    detection.x(),
                    detection.y(),
                    detection.width(),
                    detection.height(),
                    detection.confidence() * 100
            );
        }
    }

    private static void processFaces(
            Mat original,
            double detectionScaleFactor,
            List<FaceDetection> detections,
            SFaceEmbedder embedder
    ) throws IOException {
        float originalScale =
                (float) (1.0 / detectionScaleFactor);

        FaceQualityAnalyzer qualityAnalyzer =
                new FaceQualityAnalyzer();

        for (
                int index = 0;
                index < detections.size();
                index++
        ) {
            int faceNumber = index + 1;

            FaceDetection originalDetection =
                    detections
                            .get(index)
                            .scaledBy(originalScale);

            Path cropOutputPath =
                    Path.of(
                            "data",
                            "output",
                            "faces",
                            "face-" + faceNumber + ".jpg"
                    );

            try (
                    Mat faceCrop = FaceCropper.crop(
                            original,
                            originalDetection,
                            0.15
                    )
            ) {
                FaceQualityReport qualityReport =
                        qualityAnalyzer.analyse(faceCrop);

                saveImage(
                        faceCrop,
                        cropOutputPath
                );

                printQualityReport(
                        faceNumber,
                        qualityReport
                );

                if (qualityReport.accepted()) {
                    createAndReportEmbedding(
                            faceNumber,
                            original,
                            originalDetection,
                            embedder
                    );
                } else {
                    System.out.println(
                            "  Embedding skipped because "
                                    + "the face failed quality checks."
                    );
                }
            }
        }
    }

    private static void createAndReportEmbedding(
            int faceNumber,
            Mat original,
            FaceDetection originalDetection,
            SFaceEmbedder embedder
    ) throws IOException {
        Path alignedOutputPath =
                Path.of(
                        "data",
                        "output",
                        "aligned",
                        "face-" + faceNumber + ".jpg"
                );

        try (
                Mat detectionRow =
                        originalDetection.toOpenCvRow();

                FaceEmbedding embedding =
                        embedder.createEmbedding(
                                original,
                                detectionRow
                        )
        ) {
            saveImage(
                    embedding.alignedFace(),
                    alignedOutputPath
            );

            System.out.println();
            System.out.println(
                    "Embedding generated for face "
                            + faceNumber + ":"
            );

            System.out.printf(
                    "  Aligned dimensions: %d × %d%n",
                    embedding.alignedFace().cols(),
                    embedding.alignedFace().rows()
            );

            System.out.printf(
                    "  Embedding matrix: %d × %d%n",
                    embedding.values().rows(),
                    embedding.values().cols()
            );

            System.out.println(
                    "  Embedding values: "
                            + embedding.dimensions()
            );

            System.out.println(
                    "  Status: READY FOR COMPARISON"
            );
        }
    }

    private static void printQualityReport(
            int faceNumber,
            FaceQualityReport report
    ) {
        System.out.println();

        System.out.println(
                "Quality report for face "
                        + faceNumber + ":"
        );

        System.out.printf(
                "  Resolution: %d × %d%n",
                report.width(),
                report.height()
        );

        System.out.printf(
                "  Brightness: %.2f / 255%n",
                report.brightness()
        );

        System.out.printf(
                "  Contrast: %.2f%n",
                report.contrast()
        );

        System.out.printf(
                "  Sharpness: %.2f%n",
                report.sharpness()
        );

        if (report.accepted()) {
            System.out.println(
                    "  Decision: ACCEPTED FOR RECOGNITION"
            );
        } else {
            System.out.println(
                    "  Decision: REJECTED"
            );

            for (String problem : report.problems()) {
                System.out.println(
                        "  Reason: " + problem
                );
            }
        }
    }

    private static void saveImage(
            Mat image,
            Path outputPath
    ) throws IOException {
        Path absoluteOutputPath =
                outputPath.toAbsolutePath().normalize();

        Files.createDirectories(
                absoluteOutputPath.getParent()
        );

        if (!imwrite(
                absoluteOutputPath.toString(),
                image
        )) {
            throw new IOException(
                    "Could not save image: "
                            + absoluteOutputPath
            );
        }

        System.out.println(
                "Image saved to: "
                        + absoluteOutputPath
        );
    }
}
