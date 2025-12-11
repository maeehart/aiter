// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include <torch/extension.h>
#include <optional>

// Declaration of the launcher function
torch::Tensor hipkittens_fp8_gemm_decode(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& A_scale,
    torch::Tensor& B_scale,
    std::optional<torch::Tensor> bias,
    int output_dtype
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hipkittens_fp8_gemm_decode",
          &hipkittens_fp8_gemm_decode,
          "HipKittens FP8 GEMM Decode kernel",
          py::arg("A"),
          py::arg("B"),
          py::arg("A_scale"),
          py::arg("B_scale"),
          py::arg("bias") = std::nullopt,
          py::arg("output_dtype") = 0);
}

