#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import ctypes
import numpy as np
import time
import os

class MatrixSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Решение СЛАУ (C++ + Python)")

        self.libs = {}
        lib_names = {
            "Наивная (O0)": "./lib/libmatrix_solver_naive.so",
            "Блочная векторизация": "./lib/libmatrix_solver_block.so",
            "Блочная + выравнивание": "./lib/libmatrix_solver_aligned.so"
        }
        for name, path in lib_names.items():
            if os.path.exists(path):
                try:
                    lib = ctypes.CDLL(path)
                    lib.solve_linear_system.argtypes = [
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double)
                    ]
                    lib.solve_linear_system.restype = ctypes.c_int
                    self.libs[name] = lib
                    print(f"Loaded: {name}")
                except Exception as e:
                    print(f"Error loading {name}: {e}")

        if not self.libs:
            messagebox.showerror("Ошибка", "Библиотеки не найдены. Запустите make")
            root.destroy()
            return

        tk.Label(root, text="Размер матрицы:").grid(row=0, column=0, padx=5, pady=5)
        self.n_var = tk.IntVar(value=100)
        tk.Entry(root, textvariable=self.n_var, width=10).grid(row=0, column=1)

        tk.Label(root, text="Версия:").grid(row=1, column=0)
        self.version_var = tk.StringVar(value=list(self.libs.keys())[0])
        ttk.Combobox(root, textvariable=self.version_var,
                     values=list(self.libs.keys()), state="readonly").grid(row=1, column=1)

        tk.Button(root, text="Сгенерировать матрицу", command=self.generate).grid(row=2, column=0, pady=5)
        tk.Button(root, text="Решить", command=self.solve).grid(row=2, column=1, pady=5)

        self.text = tk.Text(root, height=15, width=60)
        self.text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.A = None
        self.b = None

    def generate(self):
        n = self.n_var.get()
        np.random.seed(42)
        A = np.random.rand(n, n).astype(np.float64) * 10
        for i in range(n):
            A[i, i] = np.sum(np.abs(A[i, :])) + 10
        x_true = np.random.rand(n).astype(np.float64)
        b = A @ x_true
        self.A = A
        self.b = b
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, f"Матрица {n}x{n} сгенерирована\n")
        self.text.insert(tk.END, f"Невязка эталона: {np.linalg.norm(b - A @ x_true):.2e}\n")

    def solve(self):
        if self.A is None:
            messagebox.showwarning("Ошибка", "Сначала сгенерируйте матрицу")
            return

        n = self.n_var.get()
        version = self.version_var.get()
        lib = self.libs[version]

        a_flat = self.A.flatten().copy()
        b_copy = self.b.copy()
        x = np.zeros(n, dtype=np.float64)

        start = time.time()
        ret = lib.solve_linear_system(
            n,
            a_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        elapsed = time.time() - start

        if ret != 0:
            messagebox.showerror("Ошибка", f"Ошибка решения (код {ret})")
            return

        residual = np.linalg.norm(self.b - self.A @ x)

        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, f"Версия: {version}\n")
        self.text.insert(tk.END, f"Размер матрицы: {n}\n")
        self.text.insert(tk.END, f"Время решения: {elapsed:.4f} сек\n")
        self.text.insert(tk.END, f"Невязка: {residual:.2e}\n")
        self.text.insert(tk.END, "\nПервые 5 неизвестных:\n")
        for i in range(min(5, n)):
            self.text.insert(tk.END, f"  x[{i}] = {x[i]:.6f}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = MatrixSolverApp(root)
    root.mainloop()
