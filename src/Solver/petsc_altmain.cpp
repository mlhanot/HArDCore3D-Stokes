#include "petsc_common.hpp"


int main(int argc, char * argv[])
{
  int rank;
  int mpi_size;
  PetscErrorCode ierr;

  PETSCERRORHANDLE(PetscInitializeNoArguments());
  //PETSCERRORHANDLE(PetscInitialize(&argc,&argv,NULL,NULL));
  MPIERRORHANDLE(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPIERRORHANDLE(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
  MPI_Datatype MPI_PetscInt = convertTypeMPI<PetscInt>();

  int master_rank;
  // Receive master rank
  // Init must begin by the allocation sequence
  MPIERRORHANDLE(MPI_Recv(&master_rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

  int loc_pos=0, loc_size=0, loc_size_bdr=0;
  // Declare data
  KSP m_ksp;
  PC m_pc;
  Mat m_A,bdr;
  Vec m_rhs,m_x,m_mDvals;
  Mat m_SchurMat;
  std::vector<IS> m_IS;
  PetscViewerAndFormat *m_vf;

  // Await order (setup options, compute, solve, free or exit)
  int order;
  int m_problem_type;
state_switch:
  MPIERRORHANDLE(MPI_Bcast(&order,1,MPI_INT,master_rank,MPI_COMM_WORLD));
  switch(order) {
    case (MPI_ORDER_OPTS) : goto lb_opts;
    case (MPI_ORDER_MONITOR) : goto lb_monitor;
    case (MPI_ORDER_FREE) : goto lb_free;
    case (MPI_ORDER_RESET) : goto begining;
    case (MPI_ORDER_SOLVE) : goto lb_solve;
    case (MPI_ORDER_SETGUESS) : goto lb_setguess;
    case (MPI_ORDER_RESIDUAL) : goto lb_residual;
    case (MPI_ORDER_KILL) : goto lb_kill;
    case (MPI_ORDER_SETRHS) : goto lb_setrhs;
    case (MPI_ORDER_UPDATERHS) : goto lb_updaterhs;
    case (MPI_ORDER_SETNULLSPACE) : goto lb_setnullspace;
    default: throw std::runtime_error("Unhandled order");
  }
begining:
  int size_map,size_bdr;
  MPIERRORHANDLE(MPI_Bcast(&size_map,1,MPI_INT,master_rank,MPI_COMM_WORLD));
  MPIERRORHANDLE(MPI_Bcast(&size_bdr,1,MPI_INT,master_rank,MPI_COMM_WORLD));

  // PETsc data
  PetscInt *nnz_mat,*nnz_mat_diag,*nnz_bdr,*nnz_bdr_diag;
  PETSCERRORHANDLE(PetscMalloc1(size_map,&nnz_mat));
  PETSCERRORHANDLE(PetscMalloc1(size_map,&nnz_bdr)); 
  PETSCERRORHANDLE(PetscMalloc1(size_map,&nnz_mat_diag));
  PETSCERRORHANDLE(PetscMalloc1(size_map,&nnz_bdr_diag)); 

  // Receive data
  MPIERRORHANDLE(MPI_Bcast(nnz_mat, size_map, MPI_PetscInt, master_rank, MPI_COMM_WORLD));
  MPIERRORHANDLE(MPI_Bcast(nnz_mat_diag, size_map, MPI_PetscInt, master_rank, MPI_COMM_WORLD));
  MPIERRORHANDLE(MPI_Bcast(nnz_bdr, size_map, MPI_PetscInt, master_rank, MPI_COMM_WORLD));
  MPIERRORHANDLE(MPI_Bcast(nnz_bdr_diag, size_map, MPI_PetscInt, master_rank, MPI_COMM_WORLD));

  // Get size of local data
  int all_loc_size[4];
  MPIERRORHANDLE(MPI_Bcast(all_loc_size, 4, MPI_INT, master_rank, MPI_COMM_WORLD));
  loc_pos = all_loc_size[2]*rank; // Starting pos
  loc_size = (rank == mpi_size-1) ? all_loc_size[0] : all_loc_size[2];
  loc_size_bdr = (rank == mpi_size-1) ? all_loc_size[1] : all_loc_size[3];

  // Create matrix
  PETSCERRORHANDLE(MatCreate(PETSC_COMM_WORLD, &m_A));
  PETSCERRORHANDLE(MatSetType(m_A, MATMPIAIJ));
  PETSCERRORHANDLE(MatSetSizes(m_A, loc_size, loc_size, size_map, size_map));
  PETSCERRORHANDLE(MatMPIAIJSetPreallocation(m_A,0,nnz_mat_diag + loc_pos,0,nnz_mat + loc_pos));
  //
  PETSCERRORHANDLE(MatCreate(PETSC_COMM_WORLD, &bdr));
  PETSCERRORHANDLE(MatSetType(bdr, MATMPIAIJ));
  PETSCERRORHANDLE(MatSetSizes(bdr, loc_size, loc_size_bdr, size_map, size_bdr));
  PETSCERRORHANDLE(MatMPIAIJSetPreallocation(bdr,0,nnz_bdr_diag + loc_pos,0,nnz_bdr + loc_pos));
  //
  // Assemble matrix
  PETSCERRORHANDLE(MatAssemblyBegin(m_A,MAT_FINAL_ASSEMBLY));
  PETSCERRORHANDLE(MatAssemblyEnd(m_A,MAT_FINAL_ASSEMBLY));
  PETSCERRORHANDLE(MatAssemblyBegin(bdr,MAT_FINAL_ASSEMBLY));
  PETSCERRORHANDLE(MatAssemblyEnd(bdr,MAT_FINAL_ASSEMBLY));

  // Free memory
  PETSCERRORHANDLE(PetscFree(nnz_mat));
  PETSCERRORHANDLE(PetscFree(nnz_bdr)); 
  PETSCERRORHANDLE(PetscFree(nnz_mat_diag));
  PETSCERRORHANDLE(PetscFree(nnz_bdr_diag)); 

  // Assemble vector 
  PETSCERRORHANDLE(VecCreate(PETSC_COMM_WORLD, &m_rhs));
  PETSCERRORHANDLE(VecSetType(m_rhs,VECMPI));
  PETSCERRORHANDLE(VecSetSizes(m_rhs,loc_size,size_map));
  
  PETSCERRORHANDLE(VecAssemblyBegin(m_rhs));
  PETSCERRORHANDLE(VecAssemblyEnd(m_rhs));
  // Dvals
  PETSCERRORHANDLE(VecCreate(PETSC_COMM_WORLD, &m_mDvals));
  PETSCERRORHANDLE(VecSetType(m_mDvals,VECMPI));
  PETSCERRORHANDLE(VecSetSizes(m_mDvals,loc_size_bdr,size_bdr));
  
  PETSCERRORHANDLE(VecAssemblyBegin(m_mDvals));
  PETSCERRORHANDLE(VecAssemblyEnd(m_mDvals));

  // m_rhs -= bdr*Dvals
  PETSCERRORHANDLE(MatMultAdd(bdr,m_mDvals,m_rhs,m_rhs));
    
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
  goto state_switch;

lb_opts:
  MPIERRORHANDLE(MPI_Bcast(&m_problem_type,1,MPI_INT,master_rank,MPI_COMM_WORLD));
  #include "./options.cxx"
  goto state_switch;

lb_monitor:
  MPIERRORHANDLE(MPI_Bcast(&m_problem_type,1,MPI_INT,master_rank,MPI_COMM_WORLD));
  #include "./monitor.cxx"
  goto state_switch;

lb_free:
  PETSCERRORHANDLE(KSPDestroy(&m_ksp));
  PETSCERRORHANDLE(VecDestroy(&m_mDvals));
  PETSCERRORHANDLE(VecDestroy(&m_rhs));
  PETSCERRORHANDLE(VecDestroy(&m_x));
  PETSCERRORHANDLE(MatDestroy(&m_A));
  PETSCERRORHANDLE(MatDestroy(&bdr));
  goto state_switch;

lb_setguess:
  PETSCERRORHANDLE(KSPSetInitialGuessNonzero(m_ksp,PETSC_TRUE));
  PETSCERRORHANDLE(VecAssemblyBegin(m_x));
  PETSCERRORHANDLE(VecAssemblyEnd(m_x));
  goto state_switch;

lb_solve:
  PETSCERRORHANDLE(KSPSolve(m_ksp,m_rhs,m_x));
  KSPConvergedReason reason;
  PETSCERRORHANDLE(KSPGetConvergedReason(m_ksp,&reason));
  double *sol_array;
  PETSCERRORHANDLE(VecGetArray(m_x,&sol_array));
  MPIERRORHANDLE(MPI_Ssend(sol_array,loc_size,MPI_DOUBLE,master_rank,0,MPI_COMM_WORLD));
  PETSCERRORHANDLE(VecRestoreArray(m_x,&sol_array));
  goto state_switch;

lb_setrhs:
  PETSCERRORHANDLE(VecAssemblyBegin(m_rhs));
  PETSCERRORHANDLE(VecAssemblyEnd(m_rhs));
  // m_rhs -= bdr*Dvals
  PETSCERRORHANDLE(MatMultAdd(bdr,m_mDvals,m_rhs,m_rhs));
  goto state_switch;

lb_setnullspace:
  int nullspacesize;
  MPIERRORHANDLE(MPI_Bcast(&nullspacesize,1,MPI_INT,master_rank,MPI_COMM_WORLD));
  Vec *nullspace;
  PETSCERRORHANDLE(PetscMalloc1(nullspacesize,&nullspace));
  for (int i = 0; i < nullspacesize;i++) {
    PETSCERRORHANDLE(VecDuplicate(m_rhs,nullspace+i));
    PETSCERRORHANDLE(VecAssemblyBegin(nullspace[i]));
    PETSCERRORHANDLE(VecAssemblyEnd(nullspace[i]));
    PetscReal vnorm;
    PETSCERRORHANDLE(VecNormalize(nullspace[i],&vnorm));
      //// Check if the vector is indeed in the nullspace
      //Vec res2;
      //VecDuplicate(m_rhs,&res2);
      //MatMult(m_A,nullspace[i],res2);
      //VecNorm(res2,NORM_2,&vnorm);
  }
  MatNullSpace Mnullspace;
  PETSCERRORHANDLE(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,nullspacesize,nullspace,&Mnullspace));
  // reduce reference count
  for (int i = 0; i < nullspacesize;i++) {
    PETSCERRORHANDLE(VecDestroy(nullspace +i));
  }
  PETSCERRORHANDLE(PetscFree(nullspace));
  // Attach nullspace to matrix
  PETSCERRORHANDLE(MatSetNullSpace(m_A,Mnullspace));
  PETSCERRORHANDLE(MatSetTransposeNullSpace(m_A,Mnullspace));
  PETSCERRORHANDLE(MatNullSpaceDestroy(&Mnullspace));
  goto state_switch;

lb_updaterhs:
  PETSCERRORHANDLE(VecAssemblyBegin(m_rhs));
  PETSCERRORHANDLE(VecAssemblyEnd(m_rhs));
  goto state_switch;

lb_residual:
  double resval;
  Vec res;
  // Compute A*x - b
  PETSCERRORHANDLE(VecDuplicate(m_rhs,&res));
  PETSCERRORHANDLE(MatMult(m_A,m_x,res));
  PETSCERRORHANDLE(VecAXPY(res,-1.,m_rhs));
  // Export norms
  PETSCERRORHANDLE(VecNorm(res,NORM_2, &resval));
  PETSCERRORHANDLE(VecDestroy(&res));
  PETSCERRORHANDLE(VecNorm(m_rhs,NORM_2, &resval));
  goto state_switch;

lb_kill:
  PETSCERRORHANDLE(PetscFinalize());
  return 0;
}

