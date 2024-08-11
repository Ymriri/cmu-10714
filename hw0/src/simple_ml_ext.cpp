#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <assert.h>

namespace py = pybind11;

void print_mat(const float *mat, size_t m, size_t n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf("\t");
        for (int j = 0; j < n; j++) {
            printf("%f ", mat[i * n + j]);
        }
        printf("\n");
    }
    printf("]\n");
}

void my_softmax_regresion(const float *X,
    const unsigned char *y,
    float *theta,
    size_t m, size_t n, size_t k,
    float lr)
{
    // 首先计算min_X@ theta  (batch, numclass)
    float *temp_Z = (float*) malloc(sizeof(float) * m * k);
    // k应该是批次
    for(size_t i=0;i<m*k;i++){
        float now_value = 0.;
        // 计算当前的batch
        size_t r = i/k;
        // 当前的num_calss
        size_t l = i%k;
        // j 是特征的数量
        for(size_t j=0;j<n;j++){
            now_value += X[r*n+j]*theta[j*k+l];
        }
        // 计算e^value
        temp_Z[i] = exp(now_value);

    }

    // 计算 a/sum
    for(size_t i=0;i<m;i++){
        float sum = 0.;
        for(size_t j=0;j<k;j++){
            sum+=temp_Z[i*k+j];
        }
        for(size_t j=0;j<k;j++){
            temp_Z[i*k+j] /= sum;
        }
    }

    // 开始计算梯度

    // Z-I , 只需要计算每列的正确的标签
    for(size_t i=0;i<m;i++){
        // 为什么是size_t(y[i]) 因为y[i]是unsigned char类型
        temp_Z[i*k + size_t(y[i])] -= 1.;
    }

    // 计算 mini_x.T @ (Z-I) (feature*batch) * (batch*numclass) = feature*numclass
    float *dW = (float*) malloc(sizeof(float) * n * k);

    for( size_t i=0;i<n*k;i++){
        float now_value = 0.;
        size_t r = i/k;
        size_t l = i%k;
        for(size_t j=0;j<m;j++){
            now_value += X[j*n+r]*temp_Z[j*k+l];
        }
        dW[i] = now_value;
    }

    // updat theta value
    for (size_t i = 0; i < n * k; i++) {
        theta[i] -= lr * dW[i] / m;
    }

    free(temp_Z);
    free(dW);

}
void softmax_regression(
    const float *X,
    const unsigned char *y,
    float *theta,
    size_t m, size_t n, size_t k,
    float lr
) {
    /// Z = softmax(X @ T)
    float *Z = (float*)(malloc(sizeof(float) * m * k));
    for (size_t i = 0; i < m * k; i++) {
        size_t r = i / k;
        size_t c = i % k;

        float dot = 0.;
        for (size_t j = 0; j < n; j++) {
            dot += X[r*n+j] * theta[j*k+c];
        }
        Z[i] = exp(dot);
    }
    for (size_t i = 0; i < m; i++) {
        float sum = 0.;
        for (size_t j = 0; j < k; j++) {
            sum += Z[i*k+j];
        }
        for (size_t j = 0; j < k; j++) {
            Z[i*k+j] /= sum;
        }
    }

    /// (Z - I)
    for (size_t i = 0; i < m; i++) {
        Z[i*k + size_t(y[i])] -= 1.;
    }

    /// X_T @ (Z - I)
    float *dW = (float *)(malloc(sizeof(float) * n * k));
    for (size_t i = 0; i < n * k; i++) {
        size_t r = i / k;
        size_t c = i % k;

        float dot = 0.;
        for (size_t j = 0; j < m; j++) {
            dot += X[j*n+r] * Z[j*k+c];
        }
        dW[i] = dot;
    }

    for (size_t i = 0; i < n * k; i++) {
        theta[i] -= lr * dW[i] / m;
    }

    // TODO(zhangfan): we could've reuse the same buffer aross minibatches.
    free(Z);
    free(dW);
}

void softmax_regression_epoch_cpp(
    const float *X,
    const unsigned char *y,
    float *theta,
    size_t m, size_t n, size_t k,
    float lr,
    size_t batch) {
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    assert(m % batch == 0);
    size_t num_batches = m / batch;

    for (size_t i = 0; i < num_batches; i++) {
        const float *mini_X = X + i * batch * n;
        const unsigned char *mini_y = y + i * batch;
        // 拿到当前批次的数据，然后计算softmax_regression , update theta
        my_softmax_regresion(mini_X, mini_y, theta, batch, n, k, lr);
    }
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
