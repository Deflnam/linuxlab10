CXX = g++
CXXFLAGS_NAIVE = -O0 -Wall -Wextra -fPIC
CXXFLAGS_BLOCK = -O3 -march=native -ftree-vectorize -mavx -Wall -Wextra -fPIC
CXXFLAGS_ALIGNED = -O3 -march=native -ftree-vectorize -mavx -malign-double -Wall -Wextra -fPIC
LDFLAGS = -shared

INCDIR = include
SRCDIR = src
LIBDIR = lib

TARGETS = $(LIBDIR)/libmatrix_solver_naive.so \
          $(LIBDIR)/libmatrix_solver_block.so \
          $(LIBDIR)/libmatrix_solver_aligned.so

all: $(LIBDIR) $(TARGETS)

$(LIBDIR):
 mkdir -p $(LIBDIR)

$(LIBDIR)/libmatrix_solver_naive.so: $(SRCDIR)/matrix_solver_naive.cpp
 $(CXX) $(CXXFLAGS_NAIVE) $(LDFLAGS) -I$(INCDIR) -o $@ $<

$(LIBDIR)/libmatrix_solver_block.so: $(SRCDIR)/matrix_solver_block.cpp
 $(CXX) $(CXXFLAGS_BLOCK) $(LDFLAGS) -I$(INCDIR) -o $@ $<

$(LIBDIR)/libmatrix_solver_aligned.so: $(SRCDIR)/matrix_solver_aligned.cpp
 $(CXX) $(CXXFLAGS_ALIGNED) $(LDFLAGS) -I$(INCDIR) -o $@ $<

clean:
 rm -rf $(LIBDIR)

.PHONY: all clean
