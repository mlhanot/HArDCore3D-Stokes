// Most config will go to petsc_common.hpp
switch(m_problem_type) {
  case(PBTYPE_LAPLACIAN):
PETSCERRORHANDLE(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT, &m_vf));
PETSCERRORHANDLE(KSPMonitorSet(m_ksp,
                               MyKSPMonitorResidual_Laplacian,
                               m_vf,
                               reinterpret_cast<PetscErrorCode (*)(void**)>(PetscViewerAndFormatDestroy)));
    break;
  case(PBTYPE_STOKES):
PETSCERRORHANDLE(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT, &m_vf));
PETSCERRORHANDLE(KSPMonitorSet(m_ksp,
                               MyKSPMonitorResidual_Stokes,
                               m_vf,
                               reinterpret_cast<PetscErrorCode (*)(void**)>(PetscViewerAndFormatDestroy)));
    break;
  default:
    throw std::runtime_error("monitor.cxx: unknow problem");

}

