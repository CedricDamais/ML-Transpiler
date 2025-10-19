#include <stdio.h>

int simple_tree(float *features, int n_features) {
    if (features[0] > 0) {
        return 0;
    } else {
        if (features[1] > 0) {
            return 0;
        } else {
            return 1;
        }
    }
}


int simple_tree_no_if(float *features, int n_features) {
    int x1_positive = (features[0] > 0);
    int x2_positive = (features[1] > 0);
    
    return (1 - x1_positive) * (1 - x2_positive);
}
