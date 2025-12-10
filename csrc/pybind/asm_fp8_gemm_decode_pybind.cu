// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "asm_fp8_gemm_decode.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fp8_gemm_decode_asm", 
          &custom_fp8_gemm::fp8_gemm_decode_asm, 
          "FP8 GEMM optimized for decode (M~128, memory-bound)",
          py::arg("A"),
          py::arg("B"),
          py::arg("A_scale"),
          py::arg("B_scale"),
          py::arg("out"),
          py::arg("split_k") = 1);
    
    m.def("is_fp8_gemm_decode_available",
          &custom_fp8_gemm::is_fp8_gemm_decode_available,
          "Check if FP8 decode GEMM kernel is available for current GPU");
    
    m.def("get_recommended_split_k",
          &custom_fp8_gemm::get_recommended_split_k,
          "Get recommended split_k for given shape",
          py::arg("M"),
          py::arg("N"),
          py::arg("K"));
}

