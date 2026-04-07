#include "matrix_solver.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

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

int solve_linear_system(int n, const double* a, const double* b, double* x) {
  if (n <= 0) return -1;

  double* A = (double*)malloc(n * n * sizeof(double));
  double* B = (double*)malloc(n * sizeof(double));
  if (!A || !B) {
    free(A);
    free(B);
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
      free(A);
      free(B);
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
    for (int k = i + 1; k < n; ++k) {
      double factor = A[k * n + i];
      if (factor == 0.0) continue;
      for (int j = i; j < n; ++j) {
        A[k * n + j] -= factor * A[i * n + j];
      }
      B[k] -= factor * B[i];
    }
  }

  for (int i = n - 1; i >= 0; --i) {
    double sum = 0.0;
    for (int j = i + 1; j < n; ++j) {
      sum += A[i * n + j] * x[j];
    }
    x[i] = B[i] - sum;
  }

  free(A);
  free(B);
  return 0;
}
