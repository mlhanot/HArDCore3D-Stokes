#ifndef PETSC_COMMON_HPP
#define PETSC_COMMON_HPP

#include <mpi.h>
#include <petscksp.h>
#include <petscpc.h>
#include <vector>
#include <iostream>
// #include "petscdraw.h" 

#define PETSCERRORHANDLE(func) ierr = func; if (ierr) throw std::runtime_error(std::string(#func) + std::to_string(ierr))
#define MPIERRORHANDLE(func) ierr = func; if (ierr != MPI_SUCCESS) throw std::runtime_error(std::string(#func) + std::to_string(ierr))
// Old style with return, does not fit 
//#define PETSCERRORHANDLE(func) ierr = func; CHKERRQ(ierr)

constexpr int MPI_ORDER_OPTS = 1;
constexpr int MPI_ORDER_MONITOR = 2;
constexpr int MPI_ORDER_FREE = 3;
constexpr int MPI_ORDER_RESET = 4;
constexpr int MPI_ORDER_SOLVE = 5;
constexpr int MPI_ORDER_SETGUESS = 6;
constexpr int MPI_ORDER_RESIDUAL = 7;
constexpr int MPI_ORDER_KILL = 9;
constexpr int MPI_ORDER_SETRHS = 10;
constexpr int MPI_ORDER_UPDATERHS = 11;
constexpr int MPI_ORDER_SETNULLSPACE = 12;

#include "problem_typedef.hpp"

// Match Petsc datatype with MPI data types
template<class> inline constexpr bool dependent_false_v = false;
template<typename T> MPI_Datatype convertTypeMPI() {
  if constexpr (sizeof(int)==sizeof(T)) {return MPI_INT;}
  else if constexpr (sizeof(long int)==sizeof(T)) {return MPI_LONG;}
  else if constexpr (sizeof(long long int)==sizeof(T)) {return MPI_LONG_LONG;}
  else {static_assert(dependent_false_v<T>,"Cannot match PetscInt with MPI datatype");}
}

// Wrapper by overload
static PetscErrorCode MyKSPMonitorResidual_Laplacian(KSP ksp, PetscInt it, PetscReal rnorm, void *mctx)
{
    PetscErrorCode ierr = 0;
    if ((it-1)%1000 == 0) {
      //ierr = KSPMonitorTrueResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
      ierr = KSPMonitorResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
      PetscLogDouble mem;
      ierr += PetscMemoryGetCurrentUsage(&mem);
      std::cout<<"Local memory usage: "<<mem<<std::endl;
    } 
    //PetscPrintf(PETSC_COMM_WORLD,"Approximated residual:\n");
    //return KSPMonitorResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
    return ierr;
}
static PetscErrorCode MyKSPMonitorResidual_Stokes(KSP ksp, PetscInt it, PetscReal rnorm, void *mctx)
{
    PetscErrorCode ierr = 0;
    //if (it-1%1000 == 0) {
      ierr = KSPMonitorTrueResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
      //ierr = KSPMonitorSingularValue(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
      //ierr = KSPMonitorResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
    //}
    //PetscPrintf(PETSC_COMM_WORLD,"Approximated residual:\n");
    //return KSPMonitorResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
    return ierr;
}
[[maybe_unused]] static PetscErrorCode MyKSPMonitorResidual_basic(KSP ksp, PetscInt it, PetscReal rnorm, void *mctx)
{
    PetscErrorCode ierr = 0;
    PetscPrintf(PETSC_COMM_WORLD,"%i ",it);
    //if (it-1%1000 == 0) {
      //ierr = KSPMonitorTrueResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
      //ierr = KSPMonitorSingularValue(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
      //ierr = KSPMonitorResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
    //}
    //PetscPrintf(PETSC_COMM_WORLD,"Approximated residual:\n");
    //return KSPMonitorResidual(ksp,it,rnorm,static_cast<PetscViewerAndFormat*>(mctx));
    return ierr;
}
[[maybe_unused]] static void * KSPstderrSC() {
  std::cerr.precision(13);
  std::cerr.setf(std::ios_base::scientific);
  PetscErrorCode ierr;
  PetscReal *mctx;
  PETSCERRORHANDLE(PetscMalloc1(1,&mctx));
  return mctx;
}
[[maybe_unused]] PetscErrorCode KSPstderrDC(void**p_mctx) {
  PetscErrorCode ierr = PetscFree(*p_mctx);
  *p_mctx = NULL;
  return ierr;
}
template <int ksp_n>
PetscErrorCode KSPstderrMonitor(KSP ksp, PetscInt it, PetscReal rnorm, void *mctx) {
  #ifdef __FROM_MASTER
    if (it == 0) *static_cast<PetscReal*>(mctx) = rnorm;
    if constexpr (ksp_n == 0) {
      std::cerr<<"\t\t";
    } else if constexpr (ksp_n == 1) {
      std::cerr<<"\t";
    }
    std::cerr<<it<<" subKSP["<<ksp_n<<"] resid norm "<<rnorm<<" ||r(i)||/||b|| "<<rnorm/(*static_cast<PetscReal*>(mctx))<<"\n";
  #endif
  return 0;
}

#endif
