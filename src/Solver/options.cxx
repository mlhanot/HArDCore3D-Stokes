switch(m_problem_type) {
  case(PBTYPE_LAPLACIAN):
// Relative tol, Absolute tol, Divergence tol, Max itterations
// Default: rtol=1e-5, atol=1e-50, dtol=1e5, and maxits=1e4
PETSCERRORHANDLE(KSPSetTolerances(m_ksp,1e-8,1e-20,1e5,1e5));
    break;
  case(PBTYPE_STOKES):
  PetscInt size_map;
  PETSCERRORHANDLE(MatGetSize(m_A,&size_map,NULL));
  if (size_map < 5e4) { // small system, use direct solver
    PETSCERRORHANDLE(KSPSetType(m_ksp,KSPPREONLY));
    PETSCERRORHANDLE(PCSetType(m_pc,PCLU));
    PETSCERRORHANDLE(PCFactorSetMatSolverType(m_pc,MATSOLVERSUPERLU_DIST));
  } else {
    constexpr size_t nb_blocks = 6;
    #define NRESET_VALI(i) i != 0 && i != 2
    int sizes[nb_blocks]; // Must be int for MPI
    #ifdef __FROM_MASTER
    // size of nabla & xsl
      assert(data.size() == nb_blocks);
      for (size_t i = 0; i < nb_blocks; i++) {
        sizes[i] = data[i];
      }
    #endif 
    MPIERRORHANDLE(MPI_Bcast(sizes, nb_blocks, MPI_INT, master_rank, MPI_COMM_WORLD));
    // Setup splitting
    PetscInt local_row_b,local_row_end;
    PETSCERRORHANDLE(MatGetOwnershipRange(m_A,&local_row_b,&local_row_end));
    PetscInt nb_elements[nb_blocks], local_starts[nb_blocks];
    // Compute start and size for each block
    for (size_t i = 0; i < nb_blocks; i++) {
      local_starts[i] = std::max(local_row_b, (NRESET_VALI(i))? static_cast<PetscInt>(sizes[i-1]) : 0);
      nb_elements[i] = std::min(local_row_end,static_cast<PetscInt>(sizes[i])) - local_starts[i];
      if (nb_elements[i] < 0) nb_elements[i] = 0; // Empty set
      // Create as many IS
      PetscInt *idx;
      PETSCERRORHANDLE(PetscMalloc(sizeof(PetscInt)*nb_elements[i],&idx));
      // fill the array
      for (int j = 0; j < nb_elements[i]; j++) {
        idx[j] = local_starts[i] + j;
      }
      m_IS.emplace_back();
      PETSCERRORHANDLE(ISCreateGeneral(PETSC_COMM_WORLD,nb_elements[i],idx,PETSC_OWN_POINTER,&m_IS.back()));
      //ISView(m_IS.back(),PETSC_VIEWER_STDOUT_WORLD);
    }
    ////
    // Do not set tolerance for m_ksp much higher than tolerance for subksp[0] & subksp[1]
    // The preconditionned residual will decrease to any bound but the real will stop decreasing
    // at that point.
    ////

    // Setup KSP
    PETSCERRORHANDLE(KSPSetTolerances(m_ksp,1e-12,1e-8,1e5,1e6));
    PETSCERRORHANDLE(KSPSetType(m_ksp,KSPGCR)); 
    //PETSCERRORHANDLE(KSPSetComputeEigenvalues(m_ksp,PETSC_TRUE));

    // Set PC type to fieldsplit
    PETSCERRORHANDLE(PCSetType(m_pc,PCFIELDSPLIT));
    // Create splits
    PETSCERRORHANDLE(PCFieldSplitSetIS(m_pc,"velocity",m_IS[0]));
    PETSCERRORHANDLE(PCFieldSplitSetIS(m_pc,"pressure",m_IS[1]));
    // Set subtype to schur
    PETSCERRORHANDLE(PCFieldSplitSetType(m_pc,PC_COMPOSITE_SCHUR));
    PETSCERRORHANDLE(PCFieldSplitSetSchurFactType(m_pc,PC_FIELDSPLIT_SCHUR_FACT_FULL));  

    Mat A,B,C;
    PETSCERRORHANDLE(MatCreateSubMatrix(m_A,m_IS[0],m_IS[0],MAT_INITIAL_MATRIX,&A));
    PETSCERRORHANDLE(MatCreateSubMatrix(m_A,m_IS[0],m_IS[1],MAT_INITIAL_MATRIX,&B));
    PETSCERRORHANDLE(MatCreateSubMatrix(m_A,m_IS[1],m_IS[0],MAT_INITIAL_MATRIX,&C));

    PETSCERRORHANDLE(MatCreateSchurComplement(A,A,B,C,NULL,&m_SchurMat));

    PETSCERRORHANDLE(MatDestroy(&A));
    PETSCERRORHANDLE(MatDestroy(&B));
    PETSCERRORHANDLE(MatDestroy(&C));
    //KSP S_ksp;
    //PETSCERRORHANDLE(MatSchurComplementGetKSP(m_SchurMat,&S_ksp));
    // S_ksp is never called

    PETSCERRORHANDLE(PCFieldSplitSetSchurPre(m_pc,PC_FIELDSPLIT_SCHUR_PRE_USER,m_SchurMat)); 

    // Extract sub-pcs
    PETSCERRORHANDLE(PCSetUp(m_pc));
    // Store subksp
    PetscInt nbSubKSP;
    KSP *subksp;
    PETSCERRORHANDLE(PCFieldSplitGetSubKSP(m_pc,&nbSubKSP,&subksp)); // inside Schur and over Schur, use PCFieldSplitSchurGetSubKSP for the others
    // subksp[0] is for A00 and subksp[1] is for S

    PETSCERRORHANDLE(KSPSetType(subksp[0],KSPFGMRES));
    //PETSCERRORHANDLE(KSPSetType(subksp[0],KSPGCR));
    PETSCERRORHANDLE(KSPSetTolerances(subksp[0],1e-6,1e-10,1e5,5e3));
    PC pc_sub0;
    PETSCERRORHANDLE(KSPGetPC(subksp[0],&pc_sub0));
    //KSPMonitorSet(subksp[0],MyKSPMonitorResidual_basic,m_vf,NULL);
    //KSPMonitorSet(subksp[0],MyKSPMonitorResidual_Stokes,m_vf,NULL);
    //PETSCERRORHANDLE(KSPMonitorSet(subksp[0],KSPstderrMonitor<0>,KSPstderrSC(),KSPstderrDC));

    // Set PC type to fieldsplit
    PETSCERRORHANDLE(PCSetType(pc_sub0,PCFIELDSPLIT));
    // Create splits
    PETSCERRORHANDLE(PCFieldSplitSetIS(pc_sub0,"vertices",m_IS[2]));
    PETSCERRORHANDLE(PCFieldSplitSetIS(pc_sub0,"edges",m_IS[3]));
    PETSCERRORHANDLE(PCFieldSplitSetIS(pc_sub0,"faces",m_IS[4]));
    PETSCERRORHANDLE(PCFieldSplitSetIS(pc_sub0,"cells",m_IS[5]));
    // Set subtype
    PETSCERRORHANDLE(PCFieldSplitSetType(pc_sub0,PC_COMPOSITE_MULTIPLICATIVE));


    //PETSCERRORHANDLE(KSPSetType(subksp[1],KSPGMRES));
    PETSCERRORHANDLE(KSPSetType(subksp[1],KSPGCR));
    PETSCERRORHANDLE(KSPSetTolerances(subksp[1],1e-6,1e-8,1e5,1e4));
    //KSPMonitorSet(subksp[1],MyKSPMonitorResidual_Stokes,m_vf,NULL);
    PETSCERRORHANDLE(KSPMonitorSet(subksp[1],KSPstderrMonitor<1>,KSPstderrSC(),KSPstderrDC));

    // free ksp array
    PETSCERRORHANDLE(PetscFree(subksp));
    PETSCERRORHANDLE(MatDestroy(&m_SchurMat));
    for (size_t i = 0; i < m_IS.size(); i++) {
      PETSCERRORHANDLE(ISDestroy(&m_IS[i]));
    }
    m_IS.resize(0);

  }
    break;
  default:
    throw std::runtime_error("options.cxx: unknown problem");
}

