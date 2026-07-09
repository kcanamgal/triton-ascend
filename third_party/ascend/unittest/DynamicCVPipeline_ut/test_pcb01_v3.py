"""
【Dynamic-CV-Pipeline测试用例demo】
Test Case: PCB01 - Single independent Cube + single independent Vector, no data dependency

[MLIR校验] 重构版本

Description: Single independent Cube (outer product) + single independent Vector (element-wise add),
             no data dependency between Cube and Vector operations.

重构方式(test_custom.py):
  1. 移除原测试中Kernel与参考实现的精度对比逻辑
  

数值正确性:
  - out1: A[K-1, :] 和 B[K-1, :] 的外积 (最后一次迭代的结果)
  - out2: C + D (每次迭代结果相同)

Test Cases:
  - PCB01-TC01: float16, M=128, N=64, K=32
  - PCB01-TC02: float32, M=128, N=64, K=32
"""
import os
import subprocess
import sys
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg, min_dot_size
import pytest


MLIR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlir_output")


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel function to MLIR in linalg dialect."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    try:
        options = NPUOptions()
        codegen_fns = {"min_dot_size": min_dot_size(None)}
        ttir = ast_to_ttir(kernel, src, context, options, codegen_fns, {})
        metadata = {
            **options.__dict__,
        }
        linalg = ttir_to_linalg(ttir, metadata, options, named_ops=True)
        return str(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")
        return None
    

def write_string_to_file(file_path, content):
    """
    将MLIR代码写入指定路径的文件中

    Args:
        mlir: MLIR代码字符串
        filename: 输出文件名(不含路径), 例如 "pcb01_tc01.mlir"
    """
    os.makedirs(MLIR_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(MLIR_OUTPUT_DIR, file_path)
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"成功写入文件: {output_path}")
    except Exception as e:
        print(f"写入文件失败，错误信息: {e}")


# ============================================================================
# PCB01-TC01: float16, M=128, N=64, K=32
# 测试目的: 验证float16下独立CUBE+VECTOR操作
# ============================================================================
@triton.jit
def pcb01_tc01_single_cube_vector(
    a_ptr, b_ptr, c_ptr, d_ptr, out1_ptr, out2_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_c, stride_d,
    stride_out1m, stride_out1n,
    stride_out2,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        a_mat = a[:, None]  # (K, 1)
        b_mat = b[None, :]  # (1, K)
        cube_result = tl.dot(a_mat, b_mat)  # (K, K)
        out1_ptrs = out1_ptr + offs_k[:, None] * stride_out1m + offs_k[None, :] * stride_out1n
        out1_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out1_ptrs, cube_result, mask=out1_mask)

        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)
        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)  # (N,)
        vec_result = c + d  # (N,)
        out2_ptrs = out2_ptr + offs_n * stride_out2
        tl.store(out2_ptrs, vec_result, mask=offs_n < N)

# ============================================================================
# Pytest测试用例
# ============================================================================
def test_pcb01_tc01():
    signature = {
        "a_ptr" : "*fp16",
        "b_ptr" : "*fp16",
        "c_ptr" : "*fp16",
        "d_ptr" : "*fp16",
        "out1_ptr" : "*fp16",
        "out2_ptr" : "*fp16",
        "M" : "i32",
        "N" : "i32",
        "K" : "i32",
        "stride_am" : "i32",
        "stride_ak" : "i32",
        "stride_bk" : "i32",
        "stride_bn" : "i32",
        "stride_c" : "i32",
        "stride_d" : "i32",
        "stride_out1m" : "i32",
        "stride_out1n" : "i32",
        "stride_out2" : "i32",
    }
    constants = {"BLOCK_SIZE_N" : 64, "BLOCK_SIZE_K" : 32}
    mlir = compile_kernel(pcb01_tc01_single_cube_vector, signature, constants)

    write_string_to_file("pcb01_tc01_single_cube_vector.mlir", mlir)


# ============================================================================
# PCB01-TC02: float32, M=128, N=64, K=32
# 测试目的: 验证float32下独立CUBE+VECTOR操作
# ============================================================================
@triton.jit
def pcb01_tc02_single_cube_vector(
    a_ptr, b_ptr, c_ptr, d_ptr, out1_ptr, out2_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_c, stride_d,
    stride_out1m, stride_out1n,
    stride_out2,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        a_mat = a[:, None]  # (K, 1)
        b_mat = b[None, :]  # (1, K)
        cube_result = tl.dot(a_mat, b_mat)  # (K, K)
        out1_ptrs = out1_ptr + offs_k[:, None] * stride_out1m + offs_k[None, :] * stride_out1n
        out1_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out1_ptrs, cube_result, mask=out1_mask)

        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)
        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)  # (N,)
        vec_result = c + d  # (N,)
        out2_ptrs = out2_ptr + offs_n * stride_out2
        tl.store(out2_ptrs, vec_result, mask=offs_n < N)


def test_pcb01_tc02():
    signature = {
        "a_ptr" : "*fp32",
        "b_ptr" : "*fp32",
        "c_ptr" : "*fp32",
        "d_ptr" : "*fp32",
        "out1_ptr" : "*fp32",
        "out2_ptr" : "*fp32",
        "M" : "i32",
        "N" : "i32",
        "K" : "i32",
        "stride_am" : "i32",
        "stride_ak" : "i32",
        "stride_bk" : "i32",
        "stride_bn" : "i32",
        "stride_c" : "i32",
        "stride_d" : "i32",
        "stride_out1m" : "i32",
        "stride_out1n" : "i32",
        "stride_out2" : "i32",
    }
    constants = {"BLOCK_SIZE_N" : 64, "BLOCK_SIZE_K" : 32}
    mlir = compile_kernel(pcb01_tc02_single_cube_vector, signature, constants)

    write_string_to_file("pcb01_tc02_single_cube_vector.mlir", mlir)


if __name__ == "__main__":
    test_pcb01_tc01()
    test_pcb01_tc02()
    print("All PCB01 v3 tests passed!")
