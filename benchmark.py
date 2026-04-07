#!/usr/bin/env python3
import ctypes
import numpy as np
import time
import sys
import os

def run_benchmark(lib_path, name, sizes):
    if not os.path.exists(lib_path):
        return None

    lib = ctypes.CDLL(lib_path)
    lib.solve_linear_system.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.solve_linear_system.restype = ctypes.c_int

    results = []
    for n in sizes:
        np.random.seed(42)
        A = np.random.rand(n, n).astype(np.float64) * 10
        for i in range(n):
            A[i, i] = np.sum(np.abs(A[i, :])) + 10
        x_true = np.random.rand(n).astype(np.float64)
        b = A @ x_true

        a_flat = A.flatten().copy()
        b_copy = b.copy()
        x = np.zeros(n, dtype=np.float64)

        start = time.time()
        ret = lib.solve_linear_system(
            n,
            a_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        elapsed = time.time() - start

        if ret == 0:
            residual = np.linalg.norm(b - A @ x)
            results.append((n, elapsed, residual))
            print(f"{name}: n={n}, time={elapsed:.4f}s, residual={residual:.2e}")
        else:
            results.append((n, None, None))
            print(f"{name}: n={n} FAILED (code {ret})")

    return results

def main():
    sizes = [100, 200, 300, 400, 500]
    libs = [
        ("./lib/libmatrix_solver_naive.so", "Naive (O0)"),
        ("./lib/libmatrix_solver_block.so", "Block Vectorization"),
        ("./lib/libmatrix_solver_aligned.so", "Aligned + Vectorization")
    ]

    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    all_results = {}
    for lib_path, name in libs:
        print(f"\nRunning {name}...")
        all_results[name] = run_benchmark(lib_path, name, sizes)

    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Size':<8} {'Naive (O0)':<15} {'Block Vectorization':<20} {'Aligned':<15}")
    print("-" * 60)
    for i, n in enumerate(sizes):
        naive = all_results.get("Naive (O0)", [])
        block = all_results.get("Block Vectorization", [])
        aligned = all_results.get("Aligned + Vectorization", [])
        t_naive = naive[i][1] if i < len(naive) and naive[i][1] else 0
        t_block = block[i][1] if i < len(block) and block[i][1] else 0
        t_aligned = aligned[i][1] if i < len(aligned) and aligned[i][1] else 0
        print(f"{n:<8} {t_naive:<15.4f} {t_block:<20.4f} {t_aligned:<15.4f}")

if __name__ == "__main__":
    main()
    