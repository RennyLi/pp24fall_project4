#include "ops.hpp"
const float epsilon = 1e-20;

void gemm(const float* A, const float* B, float* Out, size_t batch, size_t mn, size_t k)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < k; i++) {
            Out[b * k + i] = 0.0f;
            for (size_t j = 0; j < mn; j++) {
                Out[b * k + i] += A[b * mn + j] * B[j * k + i];
            }
        }
    }
    // END YOUR CODE HERE <-
}

void add_bias(float* A, float* B, const float* bias, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < out_dim; i++) {
            B[b * out_dim + i] = A[b * out_dim + i] + bias[i];
        }
    }
    // END YOUR CODE HERE <-
}

void Relu(float* A, float* B, size_t size)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < size; i++) {
        B[i] = std::max(0.0f, A[i]);
    }
    // END YOUR CODE HERE <-
}

void Softmax(float* A, float* B, size_t batch, size_t out_dim) {
    for (size_t i = 0; i < batch; ++i) {
        float max_val = A[i * out_dim];
        // find the maximum value
        for (size_t j = 1; j < out_dim; ++j) {
            if (A[i * out_dim + j] > max_val) {
                max_val = A[i * out_dim + j];
            }
        }

        // calculate exp and summarize
        float sum = 0.0f;
        for (size_t j = 0; j < out_dim; ++j) {
            B[i * out_dim + j] = std::exp(A[i * out_dim + j] - max_val);
            sum += B[i * out_dim + j];
        }

        // back to 1
        for (size_t j = 0; j < out_dim; ++j) {
            B[i * out_dim + j] /= (sum + epsilon);
        }
    }
}

void vector_to_one_hot_matrix(const unsigned char* A, float* B, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    std::fill(B, B + batch * out_dim, 0.0f);
    for (size_t b = 0; b < batch; b++) {
        B[b * out_dim + A[b]] = 1.0f;
    }
    // END YOUR CODE HERE <-
}

void cross_entropy_loss(const float* predictions, const float* labels, float* loss, size_t batch, size_t class_num) {
    for (size_t i = 0; i < batch; ++i) {
        loss[i] = 0.0f;
        for (size_t j = 0; j < class_num; ++j) {
            loss[i] -= labels[i * class_num + j] * std::log(predictions[i * class_num + j] + epsilon);
        }
    }
}

void cross_entropy_loss_grad(const float* A, const float* B, float* Grad, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < out_dim; i++) {
            Grad[b * out_dim + i] = A[b * out_dim + i] - B[b * out_dim + i];
        }
    }
    // END YOUR CODE HERE <-
}

void update_bias(float* Bias, const float* Output_Grad, size_t batch, float lr, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < out_dim; i++) {
        float grad = 0.0f;
        for (size_t b = 0; b < batch; b++) {
            grad += Output_Grad[b * out_dim + i];
        }
        Bias[i] -= lr * grad / batch;
    }
    // END YOUR CODE HERE <-
}

void input_grad(const float* Weight, const float* Output_Grad, float* Input, float* Input_Grad, size_t batch, size_t in_dim, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < in_dim; i++) {
            Input_Grad[b * in_dim + i] = 0.0f;
            for (size_t j = 0; j < out_dim; j++) {
                Input_Grad[b * in_dim + i] += Output_Grad[b * out_dim + j] * Weight[i * out_dim + j];
            }
        }
    }
    // END YOUR CODE HERE <-
}

void update_weight(float* Weight, const float* Output_Grad, const float* Input, size_t batch, float lr, size_t in_dim, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < in_dim; i++) {
        for (size_t j = 0; j < out_dim; j++) {
            float grad = 0.0f;
            for (size_t b = 0; b < batch; b++) {
                grad += Output_Grad[b * out_dim + j] * Input[b * in_dim + i];
            }
            Weight[i * out_dim + j] -= lr * grad / batch;
        }
    }
    // END YOUR CODE HERE <-
}

void relu_grad(const float* A, float* Grad, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < batch * out_dim; i++) {
        Grad[i] = A[i] > 0 ? Grad[i] : 0.0f;
    }
    // END YOUR CODE HERE <-
}

float mean_acc(const unsigned char* result, const unsigned char* labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE HERE ->
    size_t correct = 0;
    for (size_t i = 0; i < images_num; i++) {
        if (result[i] == labels_array[i]) correct++;
    }
    return static_cast<float>(correct) / images_num;
    // return 0.0f;
    // END YOUR CODE HERE <-
}

void argmax(const float* A, unsigned char* B, size_t num_classes, size_t images_num)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < images_num; i++) {
        float max_val = A[i * num_classes];
        size_t max_idx = 0;
        for (size_t j = 1; j < num_classes; j++) {
            if (A[i * num_classes + j] > max_val) {
                max_val = A[i * num_classes + j];
                max_idx = j;
            }
        }
        B[i] = max_idx;
    }
    // END YOUR CODE HERE <-
}
