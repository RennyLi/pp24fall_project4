#include "ops.hpp"
#include <cmath>
const float epsilon = 1e-20;

void gemm(const float* A, const float* B, float* Out, size_t batch, size_t mn, size_t k)
{
    #pragma acc parallel loop collapse(2) present(A,B,Out)
    for (size_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < k; i++) {
            float tmp = 0.0f;
            for (size_t j = 0; j < mn; j++) {
                tmp += A[b * mn + j] * B[j * k + i];
            }
            Out[b * k + i] = tmp;
        }
    }
}

void add_bias(float* A, float* B, const float* bias, size_t batch, size_t out_dim)
{
    #pragma acc parallel loop collapse(2) present(A,B,bias)
    for (size_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < out_dim; i++) {
            B[b * out_dim + i] = A[b * out_dim + i] + bias[i];
        }
    }
}

void Relu(float* A, float* B, size_t size)
{
    #pragma acc parallel loop present(A,B)
    for (size_t i = 0; i < size; i++) {
        B[i] = A[i] > 0.0f ? A[i] : 0.0f;
    }
}

void Softmax(float* A, float* B, size_t batch, size_t out_dim)
{
    #pragma acc parallel loop present(A,B)
    for (size_t i = 0; i < batch; ++i) {
        float max_val = A[i * out_dim];
        for (size_t j = 1; j < out_dim; ++j) {
            max_val = A[i * out_dim + j] > max_val ? A[i * out_dim + j] : max_val;
        }
        float sum = 0.0f;
        for (size_t j = 0; j < out_dim; ++j) {
            float val = expf(A[i * out_dim + j] - max_val);
            B[i * out_dim + j] = val;
            sum += val;
        }
        for (size_t j = 0; j < out_dim; ++j) {
            B[i * out_dim + j] /= (sum + epsilon);
        }
    }
}

void vector_to_one_hot_matrix(const unsigned char* A, float* B, size_t batch, size_t out_dim)
{
    // 先清0
    #pragma acc parallel loop present(B)
    for (size_t idx = 0; idx < batch * out_dim; idx++) {
        B[idx] = 0.0f;
    }

    #pragma acc parallel loop present(A,B)
    for (size_t b = 0; b < batch; b++) {
        B[b * out_dim + A[b]] = 1.0f;
    }
}

void cross_entropy_loss(const float* A, const float* B, float* Loss, size_t batch, size_t out_dim)
{
    #pragma acc parallel loop present(A,B,Loss)
    for (size_t i = 0; i < batch; ++i) {
        float l = 0.0f;
        for (size_t j = 0; j < out_dim; ++j) {
            l -= B[i * out_dim + j] * logf(A[i * out_dim + j] + epsilon);
        }
        Loss[i] = l;
    }
}

void cross_entropy_loss_grad(const float* A, const float* B, float* Grad, size_t batch, size_t out_dim)
{
    #pragma acc parallel loop collapse(2) present(A,B,Grad)
    for (size_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < out_dim; i++) {
            Grad[b * out_dim + i] = A[b * out_dim + i] - B[b * out_dim + i];
        }
    }
}

void update_bias(float* Bias, const float* Output_Grad, size_t batch, float lr, size_t out_dim)
{
    #pragma acc parallel loop present(Bias,Output_Grad)
    for (size_t i = 0; i < out_dim; i++) {
        float grad = 0.0f;
        for (size_t b = 0; b < batch; b++) {
            grad += Output_Grad[b * out_dim + i];
        }
        Bias[i] -= lr * grad / batch;
    }
}

void input_grad(const float* Weight, const float* Output_Grad, float* Input, float* Input_Grad, size_t batch, size_t in_dim, size_t out_dim)
{
    #pragma acc parallel loop collapse(2) present(Weight,Output_Grad,Input_Grad)
    for (size_t b = 0; b < batch; b++) {
        for (size_t i = 0; i < in_dim; i++) {
            float val = 0.0f;
            for (size_t j = 0; j < out_dim; j++) {
                val += Output_Grad[b * out_dim + j] * Weight[i * out_dim + j];
            }
            Input_Grad[b * in_dim + i] = val;
        }
    }
}

void update_weight(float* Weight, const float* Output_Grad, const float* Input, size_t batch, float lr, size_t in_dim, size_t out_dim)
{
    #pragma acc parallel loop collapse(2) present(Output_Grad,Input,Weight)
    for (size_t i = 0; i < in_dim; i++) {
        for (size_t j = 0; j < out_dim; j++) {
            float grad = 0.0f;
            for (size_t b = 0; b < batch; b++) {
                grad += Output_Grad[b * out_dim + j] * Input[b * in_dim + i];
            }
            Weight[i * out_dim + j] -= lr * grad / batch;
        }
    }
}

void relu_grad(const float* A, float* Grad, size_t batch, size_t out_dim)
{
    #pragma acc parallel loop present(A,Grad)
    for (size_t i = 0; i < batch * out_dim; i++) {
        Grad[i] = A[i] > 0 ? Grad[i] : 0.0f;
    }
}

float mean_acc(const unsigned char* result, const unsigned char* labels_array, size_t images_num, size_t num_classes)
{
    size_t correct = 0;
    #pragma acc parallel loop reduction(+:correct) present(result,labels_array)
    for (size_t i = 0; i < images_num; i++) {
        if (result[i] == labels_array[i]) correct++;
    }
    return (float)correct / images_num;
}

void argmax(const float* A, unsigned char* B, size_t num_classes, size_t images_num)
{
    #pragma acc parallel loop present(A,B)
    for (size_t i = 0; i < images_num; i++) {
        float max_val = A[i * num_classes];
        size_t max_idx = 0;
        for (size_t j = 1; j < num_classes; j++) {
            if (A[i * num_classes + j] > max_val) {
                max_val = A[i * num_classes + j];
                max_idx = j;
            }
        }
        B[i] = (unsigned char)max_idx;
    }
}