#include <stdlib.h>
#include <stdio.h>
#include "simple_ml_ext.h"

void test_matmul() {
    size_t m = 2;
    size_t n = 3;
    size_t k = 1;
    const float lhs[] = {
       1.0, 2.0, 3.0,
       4.0, 5.0, 6.0,
    };
    const float rhs[] = {
       1.0,
       2.0,
       3.0,
    };
    float out[2];
    float expect[] = {14.0, 32.0};

    matmul(lhs, rhs, m, n, k, out);

    bool same = true;
    for (size_t i = 0; i < m * k; i++) {
        if (expect[i] != out[i]) {
            same = false;
            break;
        }
    }

    if (same) {
        printf("test_matmul passed\n");
    } else {
        printf("test_matmul failed\n");
    }
}



int main() {
    test_matmul();
    return 0;
}
