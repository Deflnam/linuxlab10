#ifndef MATRIX_SOLVER_H
#define MATRIX_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

int solve_linear_system(int n, const double* a, const double* b, double* x);

#ifdef __cplusplus
}
#endif

#endif
