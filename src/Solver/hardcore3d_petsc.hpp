#ifndef HARDCORE3D_PETSC_SOLVER_INT_HPP
#define HARDCORE3D_PETSC_SOLVER_INT_HPP
#include <petscksp.h>

#include <Eigen/Dense>

#include <forward_list>
#include <atomic>

#include "problem_typedef.hpp"
namespace HArDCore3D {

  extern int master_rank; // Holds the mpi rank of the master process. Used to terminate the program

  class PETscSolver {
    public:
      PETscSolver(int _problem_type);
      ~PETscSolver();
      
      // Return size of cores
      // Pass null to pointer to ignore them
      void Compute_MPISizes(int size_map,int size_bdr, 
                            int *loc_size,int * last_loc_size,
                            int *loc_size_bdr,int * last_loc_size_bdr);

      // Additionnal data for options
      // For Stokes, it must be the size of each block (not their locations)
      void SetOptions(std::vector<int> = std::vector<int>());
      void SetMonitor();

      void Set_guess(Eigen::VectorXd const & i_x);
      // m_A and m_rhs must be already set (call assemble before this)
      void Set_nullspace(const std::atomic<double> *const*at_nullspace, int nullspacesize);

      void solve(Eigen::VectorXd & x);

      int info() const {
        return m_reason;
      }

      void assemble(std::vector<std::forward_list<std::tuple<int,int,double>>> const &mat_triplets,
                    std::vector<std::forward_list<std::tuple<int,int,double>>> const &bdr_triplets,
                    const std::atomic<PetscInt> * at_nnz_mat, 
                    const std::atomic<PetscInt> * at_nnz_mat_diag, 
                    const std::atomic<PetscInt> * at_nnz_bdr, 
                    const std::atomic<PetscInt> * at_nnz_bdr_diag, 
                    const std::atomic<double> * at_rhs,
                    const double * D_vals,
                    int size_map,
                    int size_bdr);

      void Set_rhs(const std::atomic<double> *at_rhs, int size_map);
      void update_rhs(const std::atomic<double> *at_rhs, int size_map);
      void Output_Converged(std::ostream &);
      void Residual(double *abs, double *rhs_n);

      bool is_in_diag(int i, int j) const {
        const int last_start = (m_mpi_size-1)*m_all_loc_size[2];
        if (i < last_start) {
          return i/m_all_loc_size[2] == j/m_all_loc_size[2];
        } else {
          return not (j < last_start);
        }
      }
      bool is_in_diag_bdr(int i,int j) const {
        const int last_start = (m_mpi_size - 1)*m_all_loc_size[2];
        const int last_start_bdr = (m_mpi_size - 1)*m_all_loc_size[3];
        if (i < last_start) {
          return i/m_all_loc_size[2] == j/m_all_loc_size[3];
        } else {
          return not (j < last_start_bdr);
        }
      }
    private:
      Vec m_x,m_rhs,m_mDvals; // Solution, RHS, -Dirichlet values
      Mat m_A,m_bdr; // Matrix
      Mat m_SchurMat;
      KSP m_ksp; // KSP solver context
      PC m_pc; // Preconditioner context
      KSPConvergedReason m_reason;
      PetscViewerAndFormat *m_vf;
      std::vector<IS> m_IS;
      // MPI Data
      int m_rank;
      int m_mpi_size;
      int m_loc_size = -1;
      int m_loc_size_bdr = -1;
      int m_loc_pos = -1;
      int m_all_loc_size[4]; // long_loc_size [0], long_loc_size_bdr [1], short_loc_size [2], short_loc_size_bdr [3]
      bool m_ASSEMBLED = false;
      const int m_problem_type;
      static bool m_IS_PETSC_INITIALIZED; // initialized at false, set to true at the first initialization
      static bool m_IN_USE; // Prevent two instances to exists at the same time (petsc_altmain is not design for this case)

  }; // end of class definition
  
  void PETscSolver_MPI_Finalize(); // Call after deleting the last PETscSolver instance, else the program will hang

} // end of namespace

#endif
