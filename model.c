#include <stdio.h>
#include <math.h>

float prediction(float *features, int n_features) {
    if (features[0] <= 78.002831) {
        return -194619.561110;
    } else {
        if (features[0] <= 158.476189) {
            if (features[0] <= 153.471748) {
                if (features[0] <= 125.506351) {
                    if (features[2] <= 0.500000) {
                        return 248843.654966;
                    } else {
                        return 286500.280731;
                    }
                } else {
                    if (features[0] <= 132.281731) {
                        return -289382.826215;
                    } else {
                        return 293375.846650;
                    }
                }
            } else {
                return -264704.420109;
            }
        } else {
            if (features[2] <= 0.500000) {
                if (features[1] <= 1.500000) {
                    return 232510.781988;
                } else {
                    if (features[0] <= 179.071709) {
                        return 251789.848865;
                    } else {
                        return 258790.567221;
                    }
                }
            } else {
                if (features[1] <= 1.500000) {
                    if (features[0] <= 187.973213) {
                        return 282674.291717;
                    } else {
                        return 267462.125237;
                    }
                } else {
                    if (features[0] <= 164.458046) {
                        return 303886.472512;
                    } else {
                        return 317257.372473;
                    }
                }
            }
        }
    }
}

int main() {
    float features[3] = {1.0, 2.0, 0.0}; // Example features
    float pred = prediction(features, 3);
    printf("Prediction: %f\n", pred);
    return 0;
}
