#include "../include/matrix_solver.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>

#ifdef __AVX__
#include <immintrin.h>
#endif

static const int BLOCK_SIZE = 4;
static const size_t ALIGNMENT = 32;

static void* alignedAlloc(size_t size) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, ALIGNMENT, size) != 0) {
    return nullptr;
  }
  return ptr;
}

static void alignedFree(void* ptr) {
  free(ptr);
}

static void swapRows(double* a, double* b, int n, int row1, int row2) {
  for (int j = 0; j < n; ++j) {
    double tmp = a[row1 * n + j];
    a[row1 * n + j] = a[row2 * n + j];
    a[row2 * n + j] = tmp;
  }
  double tmp = b[row1];
  b[row1] = b[row2];
  b[row2] = tmp;
}

#ifdef __AVX__
static void alignedVectorizedSubtract(double* rowK, const double* rowI, int start,
                                      int n, double factor) {
  int j = start;
  // Check if addresses are 32-byte aligned
  if (((uintptr_t)(rowK + j) & 31) == 0 && ((uintptr_t)(rowI + j) & 31) == 0) {
    for (; j + 4 <= n; j += 4) {
      __m256d vFactor = _mm256_set1_pd(factor);
      __m256d vRowI = _mm256_load_pd(rowI + j);
      __m256d vRowK = _mm256_load_pd(rowK + j);
      __m256d vSub = _mm256_mul_pd(vRowI, vFactor);
      __m256d vNew = _mm256_sub_pd(vRowK, vSub);
      _mm256_store_pd(rowK + j, vNew);
    }
  } else {
    // Fallback to unaligned
    for (; j + 4 <= n; j += 4) {
      __m256d vFactor = _mm256_set1_pd(factor);
      __m256d vRowI = _mm256_loadu_pd(rowI + j);
      __m256d vRowK = _mm256_loadu_pd(rowK + j);
      __m256d vSub = _mm256_mul_pd(vRowI, vFactor);
      __m256d vNew = _mm256_sub_pd(vRowK, vSub);
      _mm256_storeu_pd(rowK + j, vNew);
    }
  }
  for (; j < n; ++j) {
    rowK[j] -= factor * rowI[j];
  }
}
#endif

int solve_linear_system(int n, const double* a, const double* b, double* x) {
  if (n <= 0) return -1;

  double* A = (double*)alignedAlloc(n * n * sizeof(double));
  double* B = (double*)alignedAlloc(n * sizeof(double));
  double* X = (double*)alignedAlloc(n * sizeof(double));
  if (!A || !B || !X) {
    alignedFree(A);
    alignedFree(B);
    alignedFree(X);
    return -2;
  }
  memcpy(A, a, n * n * sizeof(double));
  memcpy(B, b, n * sizeof(double));

  for (int i = 0; i < n; ++i) {
    int maxRow = i;
    double maxVal = fabs(A[i * n + i]);
    for (int k = i + 1; k < n; ++k) {
      double val = fabs(A[k * n + i]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = k;
      }
    }
    if (maxVal < 1e-12) {
      alignedFree(A);
      alignedFree(B);
      alignedFree(X);
      return -1;
    }
    if (maxRow != i) {
      swapRows(A, B, n, i, maxRow);
    }
    double diag = A[i * n + i];
    for (int j = i; j < n; ++j) {
      A[i * n + j] /= diag;
    }
    B[i] /= diag;

    for (int k = i + 1; k < n; k += BLOCK_SIZE) {
      int blockEnd = (k + BLOCK_SIZE > n) ? n : k + BLOCK_SIZE;
      for (int r = k; r < blockEnd; ++r) {
        double factor = A[r * n + i];
        if (factor == 0.0) continue;
#ifdef __AVX__
        alignedVectorizedSubtract(A + r * n, A + i * n, i, n, factor);
#else
        for (int j = i; j < n; ++j) {
          A[r * n + j] -= factor * A[i * n + j];
        }
#endif
        B[r] -= factor * B[i];
        A[r * n + i] = 0.0;
      }
    }
  }

  for (int i = n - 1; i >= 0; --i) {
    double sum = 0.0;
    for (int j = i + 1; j < n; ++j) {
      sum += A[i * n + j] * X[j];
    }
    X[i] = B[i] - sum;
  }

  memcpy(x, X, n * sizeof(double));

  alignedFree(A);
  alignedFree(B);
  alignedFree(X);
  return 0;
}
