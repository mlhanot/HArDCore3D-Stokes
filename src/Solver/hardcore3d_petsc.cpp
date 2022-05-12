#define __FROM_MASTER

#include "hardcore3d_petsc.hpp"
#include "petsc_common.hpp"

namespace HArDCore3D {

  int master_rank = -1;

  bool PETscSolver::m_IS_PETSC_INITIALIZED = false;
  bool PETscSolver::m_IN_USE = false;

  PETscSolver::PETscSolver(int _problem_type) : m_problem_type(_problem_type)
  {
    PetscErrorCode ierr;
    if (m_IN_USE) { // Prevent desync
      throw std::runtime_error("Only one instance of PETscSolver can be initialized at once");
    } else {
      m_IN_USE = true;
    }
    // Initialize PETsc can only be called one
    if (not m_IS_PETSC_INITIALIZED) { 
      PETSCERRORHANDLE(PetscInitializeNoArguments());
    }
    MPIERRORHANDLE(MPI_Comm_rank(MPI_COMM_WORLD, &m_rank));
    MPIERRORHANDLE(MPI_Comm_size(MPI_COMM_WORLD, &m_mpi_size));
    int buffer_comm_size = m_mpi_size*(MPI_BSEND_OVERHEAD + sizeof(int));
    char *buffer_comm = static_cast<char*>(malloc(buffer_comm_size));
    MPIERRORHANDLE(MPI_Buffer_attach(buffer_comm,buffer_comm_size));

    if (not m_IS_PETSC_INITIALIZED) {// Wake threads
      master_rank = m_rank;
      for (int i = 0; i < m_mpi_size; i++) {
        if (i == m_rank) continue;
        MPIERRORHANDLE(MPI_Bsend(&m_rank,1,MPI_INT,i,0,MPI_COMM_WORLD));
      }
      m_IS_PETSC_INITIALIZED = true;
    }
    MPI_Buffer_detach(&buffer_comm,&buffer_comm_size);
    free(buffer_comm);
  }

  PETscSolver::~PETscSolver()
  {
    m_IN_USE = false;
    if (not m_ASSEMBLED) return;
    int message = MPI_ORDER_FREE;
    MPI_Bcast(&message, 1, MPI_INT, m_rank,MPI_COMM_WORLD);
    KSPDestroy(&m_ksp); // assemble
    VecDestroy(&m_mDvals); // assemble
    VecDestroy(&m_rhs);
    VecDestroy(&m_x);
    MatDestroy(&m_A);
    MatDestroy(&m_bdr);
  }

  // Caller must know local size to compute diag/offdiag
  void PETscSolver::Compute_MPISizes(int size_map, int size_bdr, int * l_size, int * ll_size,
                                     int * lb_size, int * llb_size) {
    int loc_size = size_map/m_mpi_size; // int DIV
    int last_loc_size = size_map - loc_size*(m_mpi_size - 1);
    m_loc_pos = loc_size*m_rank;
    m_loc_size = (m_rank == m_mpi_size-1)? last_loc_size : loc_size;
    m_all_loc_size[0] = last_loc_size;
    m_all_loc_size[2] = loc_size;
    if (l_size) *l_size = loc_size;
    if (ll_size) *ll_size = last_loc_size;
    int loc_size_bdr = size_bdr/m_mpi_size; // int DIV
    int last_loc_size_bdr = size_bdr - loc_size_bdr*(m_mpi_size - 1);
    m_loc_size_bdr = (m_rank == m_mpi_size-1) ? last_loc_size_bdr : loc_size_bdr;
    m_all_loc_size[1] = last_loc_size_bdr;
    m_all_loc_size[3] = loc_size_bdr;
    if (lb_size) *lb_size = loc_size_bdr;
    if (llb_size) *llb_size = last_loc_size_bdr;
  }
  
  void PETscSolver::SetOptions(std::vector<int> data) {
    PetscErrorCode ierr;
    int message = MPI_ORDER_OPTS;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    message = m_problem_type;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    #include "./options.cxx"
  }
  
  void PETscSolver::SetMonitor() {
    PetscErrorCode ierr;
    int message = MPI_ORDER_MONITOR;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    message = m_problem_type;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    #include "./monitor.cxx"
  }

  void PETscSolver::Set_guess(Eigen::VectorXd const & i_x) {
    PetscErrorCode ierr;
    KSPType ksptype;
    PETSCERRORHANDLE(KSPGetType(m_ksp,&ksptype));
    if (strcmp(ksptype,KSPPREONLY)==0) {return;}
    int message = MPI_ORDER_SETGUESS;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    PETSCERRORHANDLE(KSPSetInitialGuessNonzero(m_ksp,PETSC_TRUE));
    for (int i = 0; i < i_x.size(); i++) {
      PETSCERRORHANDLE(VecSetValue(m_x,i,i_x(i),INSERT_VALUES));
    }
    PETSCERRORHANDLE(VecAssemblyBegin(m_x));
    PETSCERRORHANDLE(VecAssemblyEnd(m_x));
  }

  void PETscSolver::Set_nullspace(const std::atomic<double> *const*at_nullspace, int nullspacesize) {
    if (nullspacesize == 0) return;
    PetscErrorCode ierr;
    int message = MPI_ORDER_SETNULLSPACE;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    // send the number of vecs
    message = nullspacesize;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    Vec *nullspace;
    PETSCERRORHANDLE(PetscMalloc1(nullspacesize,&nullspace));
    for (int i = 0; i < nullspacesize;i++) {
      PETSCERRORHANDLE(VecDuplicate(m_rhs,nullspace+i));
      PetscInt vsize;
      PETSCERRORHANDLE(VecGetSize(m_rhs,&vsize));
      for (int j = 0; j < vsize; j++) {
        PETSCERRORHANDLE(VecSetValue(nullspace[i],j,at_nullspace[i][j].load(std::memory_order_relaxed),INSERT_VALUES));
      }
      PETSCERRORHANDLE(VecAssemblyBegin(nullspace[i]));
      PETSCERRORHANDLE(VecAssemblyEnd(nullspace[i]));
      PetscReal vnorm;
      PETSCERRORHANDLE(VecNormalize(nullspace[i],&vnorm));
      //// Check if the vector is indeed in the nullspace
      //Vec res;
      //VecDuplicate(m_rhs,&res);
      //MatMult(m_A,nullspace[i],res);
      //std::cout<<"Norm nullspace: "<<vnorm;
      //VecNorm(res,NORM_2,&vnorm);
      //std::cout<<" norm A*X: "<<vnorm<<std::endl;
    }
    MatNullSpace Mnullspace;
    PETSCERRORHANDLE(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,nullspacesize,nullspace,&Mnullspace));
    // reduce reference count
    for (int i = 0; i < nullspacesize;i++) {
      PETSCERRORHANDLE(VecDestroy(nullspace + i));
    }
    PETSCERRORHANDLE(PetscFree(nullspace));
    // Attach nullspace to matrix
    PETSCERRORHANDLE(MatSetNullSpace(m_A,Mnullspace));
    PETSCERRORHANDLE(MatSetTransposeNullSpace(m_A,Mnullspace));
    PETSCERRORHANDLE(MatNullSpaceDestroy(&Mnullspace));
  }

  void PETscSolver::solve(Eigen::VectorXd & x) {
    PetscErrorCode ierr;
    int message = MPI_ORDER_SOLVE;
    #ifdef DEBUG
    PetscInt N,M;
    PETSCERRORHANDLE(MatGetSize(m_A,&N,&M));
    if (x.size() != N) throw std::runtime_error("Vector x must be resized before calling solve");
    #endif // DEBUG
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    PETSCERRORHANDLE(KSPSolve(m_ksp,m_rhs,m_x));
    PETSCERRORHANDLE(KSPGetConvergedReason(m_ksp,&m_reason));
    ///////////////////////////
    /* Writte eigen values for analysis
    PetscReal r[100],c[100];
    PetscInt neig;
    KSPComputeEigenvalues(m_ksp,100,r,c,&neig);
    for (int i = 0; i < neig; i++) {
      std::cout<<"Eigen "<<i<<" real: "<<r[i]<<" imag: "<<c[i]<<std::endl;
    }
    */
    // export sol
    double *sol_array;
    PETSCERRORHANDLE(VecGetArray(m_x,&sol_array));
    for (int j_mp = 0; j_mp < m_mpi_size - 1;j_mp++) {
      if (j_mp == m_rank) {
        for (int i = 0; i < m_all_loc_size[2]; i++) {
          x(i+j_mp*m_all_loc_size[2]) = sol_array[i];
        }
      } else { // get data from other process
        MPIERRORHANDLE(MPI_Recv(x.data()+j_mp*m_all_loc_size[2], m_all_loc_size[2], MPI_DOUBLE, j_mp, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      }
    }
    // Last chunk
    if (m_rank == m_mpi_size - 1) { // the local chunk is the last
      for (int i = 0; i < m_all_loc_size[0]; i++) { 
        x(i + (m_mpi_size - 1)*m_all_loc_size[2]) = sol_array[i];
      }
    } else {
        MPIERRORHANDLE(MPI_Recv(x.data()+(m_mpi_size-1)*m_all_loc_size[2], m_all_loc_size[0], MPI_DOUBLE, m_mpi_size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    PETSCERRORHANDLE(VecRestoreArray(m_x,&sol_array));
}

// Warning : For matrices that will be factored, you must leave room for (and set) the diagonal entry even if it is zero
  void PETscSolver::assemble(std::vector<std::forward_list<std::tuple<int,int,double>>> const &mat_triplets,
                        std::vector<std::forward_list<std::tuple<int,int,double>>> const &bdr_triplets,
                        const std::atomic<PetscInt> * at_nnz_mat, 
                        const std::atomic<PetscInt> * at_nnz_mat_diag, 
                        const std::atomic<PetscInt> * at_nnz_bdr, 
                        const std::atomic<PetscInt> * at_nnz_bdr_diag, 
                        const std::atomic<double> * at_rhs,
                        const double * D_vals,
                        int size_map,
                        int size_bdr)
  {
    PetscErrorCode ierr;
    int message;

    if (m_ASSEMBLED) std::cerr<<"Warning: System already assembled. Reassembly is not yet fully supported. Memory will leak"<<std::endl; 
    if (m_loc_pos < 0) throw std::runtime_error("Call Compute_MPISizes before assemble");
    // Reset init
    message = MPI_ORDER_RESET;
    MPIERRORHANDLE(MPI_Bcast(&message, 1, MPI_INT, m_rank, MPI_COMM_WORLD));

    // Send global dimensions
    message = size_map;
    MPIERRORHANDLE(MPI_Bcast(&message, 1, MPI_INT, m_rank, MPI_COMM_WORLD));
    message = size_bdr;
    MPIERRORHANDLE(MPI_Bcast(&message, 1, MPI_INT, m_rank, MPI_COMM_WORLD));

    // Convert atomic to PETsc array
    PetscInt *nnz_mat, *nnz_mat_diag;
    PetscInt *nnz_bdr, *nnz_bdr_diag;
    // bdr is size_map x size_bdr
    // map is size_map x size_map
    PETSCERRORHANDLE(PetscMalloc1(size_map,&nnz_mat));
    PETSCERRORHANDLE(PetscMalloc1(size_map,&nnz_bdr)); 
    PETSCERRORHANDLE(PetscMalloc1(size_map,&nnz_mat_diag));
    PETSCERRORHANDLE(PetscMalloc1(size_map,&nnz_bdr_diag)); 
    for (int i = 0; i < size_map;i++) {
      nnz_mat[i] = at_nnz_mat[i].load(std::memory_order_relaxed);
      nnz_mat_diag[i] = at_nnz_mat_diag[i].load(std::memory_order_relaxed);
      nnz_bdr[i] = at_nnz_bdr[i].load(std::memory_order_relaxed);
      nnz_bdr_diag[i] = at_nnz_bdr_diag[i].load(std::memory_order_relaxed);
    }
    
    MPI_Datatype MPI_PetscInt = convertTypeMPI<PetscInt>();
    MPIERRORHANDLE(MPI_Bcast(nnz_mat, size_map, MPI_PetscInt, m_rank, MPI_COMM_WORLD));
    MPIERRORHANDLE(MPI_Bcast(nnz_mat_diag, size_map, MPI_PetscInt, m_rank, MPI_COMM_WORLD));
    MPIERRORHANDLE(MPI_Bcast(nnz_bdr, size_map, MPI_PetscInt, m_rank, MPI_COMM_WORLD));
    MPIERRORHANDLE(MPI_Bcast(nnz_bdr_diag, size_map, MPI_PetscInt, m_rank, MPI_COMM_WORLD));
    // Cast local sizes
    MPIERRORHANDLE(MPI_Bcast(m_all_loc_size, 4, MPI_INT, m_rank, MPI_COMM_WORLD));

    // Create matrix
    PETSCERRORHANDLE(MatCreate(PETSC_COMM_WORLD, &m_A));
    PETSCERRORHANDLE(MatSetType(m_A, MATMPIAIJ));
    PETSCERRORHANDLE(MatSetSizes(m_A, m_loc_size, m_loc_size, size_map, size_map));
    PETSCERRORHANDLE(MatMPIAIJSetPreallocation(m_A,0,nnz_mat_diag + m_loc_pos,0,nnz_mat + m_loc_pos));
    //
    PETSCERRORHANDLE(MatCreate(PETSC_COMM_WORLD, &m_bdr));
    PETSCERRORHANDLE(MatSetType(m_bdr, MATMPIAIJ));
    PETSCERRORHANDLE(MatSetSizes(m_bdr, m_loc_size, m_loc_size_bdr, size_map, size_bdr));
    PETSCERRORHANDLE(MatMPIAIJSetPreallocation(m_bdr,0,nnz_bdr_diag + m_loc_pos,0,nnz_bdr + m_loc_pos));
    //

    // File matrix (master only, distribute ?)
    for (size_t iT = 0; iT < mat_triplets.size();iT++) {
      for (auto it = mat_triplets[iT].begin(); it != mat_triplets[iT].end(); it++) {
        MatSetValue(m_A,std::get<0>(*it),std::get<1>(*it),std::get<2>(*it),ADD_VALUES);
      }
      for (auto it = bdr_triplets[iT].begin(); it != bdr_triplets[iT].end(); it++) {
        MatSetValue(m_bdr,std::get<0>(*it),std::get<1>(*it),std::get<2>(*it),ADD_VALUES);
      }
    }

    // Assemble matrix
    PETSCERRORHANDLE(MatAssemblyBegin(m_A,MAT_FINAL_ASSEMBLY));
    PETSCERRORHANDLE(MatAssemblyEnd(m_A,MAT_FINAL_ASSEMBLY));
    PETSCERRORHANDLE(MatAssemblyBegin(m_bdr,MAT_FINAL_ASSEMBLY));
    PETSCERRORHANDLE(MatAssemblyEnd(m_bdr,MAT_FINAL_ASSEMBLY));

    // Free memory
    PETSCERRORHANDLE(PetscFree(nnz_mat));
    PETSCERRORHANDLE(PetscFree(nnz_bdr)); 
    PETSCERRORHANDLE(PetscFree(nnz_mat_diag));
    PETSCERRORHANDLE(PetscFree(nnz_bdr_diag)); 

    // Assemble vector 
    PETSCERRORHANDLE(VecCreate(PETSC_COMM_WORLD, &m_rhs));
    PETSCERRORHANDLE(VecSetType(m_rhs,VECMPI));
    PETSCERRORHANDLE(VecSetSizes(m_rhs,m_loc_size,size_map));
    for (int i = 0; i < size_map; i++) {
      PETSCERRORHANDLE(VecSetValue(m_rhs,i,at_rhs[i].load(std::memory_order_relaxed),INSERT_VALUES));
    }
    PETSCERRORHANDLE(VecAssemblyBegin(m_rhs));
    PETSCERRORHANDLE(VecAssemblyEnd(m_rhs));
    // -Dval
    PETSCERRORHANDLE(VecCreate(PETSC_COMM_WORLD, &m_mDvals));
    PETSCERRORHANDLE(VecSetType(m_mDvals,VECMPI));
    PETSCERRORHANDLE(VecSetSizes(m_mDvals,m_loc_size_bdr,size_bdr));
    for (int i = 0; i < size_bdr; i++) {
      PETSCERRORHANDLE(VecSetValue(m_mDvals,i,-D_vals[i],INSERT_VALUES));
    }
    PETSCERRORHANDLE(VecAssemblyBegin(m_mDvals));
    PETSCERRORHANDLE(VecAssemblyEnd(m_mDvals));

    // rhs -= bdr*Dvals
    PETSCERRORHANDLE(MatMultAdd(m_bdr,m_mDvals,m_rhs,m_rhs));

    // Duplicate structure
    PETSCERRORHANDLE(VecDuplicate(m_rhs,&m_x));

    // Create solver context
    PETSCERRORHANDLE(KSPCreate(PETSC_COMM_WORLD, &m_ksp));
    PETSCERRORHANDLE(KSPSetOperators(m_ksp,m_A,m_A));
    PETSCERRORHANDLE(KSPGetPC(m_ksp,&m_pc));
    /* 
    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,1000,1000,&viewer);
    PetscViewerDrawSetPause(viewer,-1);
    MatView(m_A,viewer);
    */
    m_ASSEMBLED = true;
  } // end of assemble

  void PETscSolver::Set_rhs(const std::atomic<double> *at_rhs, int size_map) {
    PetscErrorCode ierr;
    int message = MPI_ORDER_SETRHS;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    for (int i = 0; i < size_map;i++) {
      PETSCERRORHANDLE(VecSetValue(m_rhs,i,at_rhs[i].load(std::memory_order_relaxed),INSERT_VALUES));
    }
    PETSCERRORHANDLE(VecAssemblyBegin(m_rhs));
    PETSCERRORHANDLE(VecAssemblyEnd(m_rhs));
    PETSCERRORHANDLE(MatMultAdd(m_bdr,m_mDvals,m_rhs,m_rhs));
  }

  void PETscSolver::update_rhs(const std::atomic<double> *at_rhs, int size_map) {
    PetscErrorCode ierr;
    int message = MPI_ORDER_UPDATERHS;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    for (int i = 0; i < size_map;i++) {
      PETSCERRORHANDLE(VecSetValue(m_rhs,i,at_rhs[i].load(std::memory_order_relaxed),ADD_VALUES));
    }
    PETSCERRORHANDLE(VecAssemblyBegin(m_rhs));
    PETSCERRORHANDLE(VecAssemblyEnd(m_rhs));
  }

  void PETscSolver::Output_Converged(std::ostream & output) {
    PetscErrorCode ierr;
    PetscInt nbit;
    PETSCERRORHANDLE(KSPGetIterationNumber(m_ksp,&nbit));
    output << "Linear solver converged in "<<nbit<<" iterations due to "<< KSPConvergedReasons[info()]<<std::endl;
  }

  void PETscSolver::Residual(double *abs, double *rhs_n) {
    PetscErrorCode ierr;
    Vec res;
    int message = MPI_ORDER_RESIDUAL;
    MPIERRORHANDLE(MPI_Bcast(&message,1,MPI_INT,m_rank,MPI_COMM_WORLD));
    // Compute A*x - b
    PETSCERRORHANDLE(VecDuplicate(m_rhs,&res));
    PETSCERRORHANDLE(MatMult(m_A,m_x,res));
    PETSCERRORHANDLE(VecAXPY(res,-1.,m_rhs));
    // Export norms
    PETSCERRORHANDLE(VecNorm(res,NORM_2, abs));
    PETSCERRORHANDLE(VecDestroy(&res));
    PETSCERRORHANDLE(VecNorm(m_rhs,NORM_2, rhs_n));
  }
    
  void PETscSolver_MPI_Finalize() {
    int message = MPI_ORDER_KILL;
    PetscErrorCode ierr;
    MPIERRORHANDLE(MPI_Bcast(&message, 1, MPI_INT, master_rank, MPI_COMM_WORLD));
    PetscFinalize();
  }
}
