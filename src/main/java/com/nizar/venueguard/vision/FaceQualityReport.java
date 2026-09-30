package com.nizar.venueguard.vision;

import java.util.List;
import java.util.Objects;

public record FaceQualityReport(
        int width,
        int height,
        double brightness,
        double contrast,
        double sharpness,
        List<String> problems
) {

    public FaceQualityReport {
        Objects.requireNonNull(problems);
        problems = List.copyOf(problems);
    }

    public boolean accepted() {
        return problems.isEmpty();
    }
}
