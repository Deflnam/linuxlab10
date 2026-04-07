#include "../include/matrix_solver.h"
#include <cstdio>
#include <cmath>

int main() {
  const int n = 3;
  double a[] = {2, 1, -1, -3, -1, 2, -2, 1, 2};
  double b[] = {8, -11, -3};
  double x[3];

  int ret = solve_linear_system(n, a, b, x);
  if (ret != 0) {
    printf("Error: %d\n", ret);
    return 1;
  }

  printf("Solution:\n");
  for (int i = 0; i < n; ++i) {
    printf("x[%d] = %f\n", i, x[i]);
  }
  return 0;
}
