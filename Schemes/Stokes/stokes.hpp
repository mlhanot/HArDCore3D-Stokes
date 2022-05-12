
#ifndef STOKES_HPP
#define STOKES_HPP

#ifdef WITH_PASTIX
  #include <Eigen/PaStiXSupport>
#elif defined WITH_UMFPACK
  #include <Eigen/UmfPackSupport>
#elif defined WITH_MKL
  #define EIGEN_USE_MKL_ALL
  #include <Eigen/PardisoSupport>
#endif

#include <Eigen/Sparse>

#include <stokescore.hpp>
#include <xnablastokes.hpp>
#include <xvlstokes.hpp>

#include <parallel_for.hpp>

#if __cplusplus > 201703L // std>c++17
#define __LIKELY [[likely]]
#define __UNLIKELY [[unlikely]]
#else
#define __LIKELY
#define __UNLIKELY
#endif

/*!
 * @defgroup Stokes
 * @brief Implementation of the solver for the Stokes problem
 */

namespace HArDCore3D
{
  
  constexpr double prune_threshold = 1e-20;
  
  // Preset for harmonics forms, to add other inplement their cases in assemble_system and setup_Harmonics
  enum Harmonics_premade {
    None,
    Velocity,
    Pressure,
    Custom
  };

  class StokesProblem {
    public:
      typedef Eigen::SparseMatrix<double> SystemMatrixType;
      // Select solver
      #ifdef ITTERATIVE
      typedef Eigen::BiCGSTAB<SystemMatrixType> SolverType;
      const std::string SolverName = "BiCGSTAB with DiagonalPreconditioner";
      #elif defined ITTERATIVE_LU
      typedef Eigen::BiCGSTAB<SystemMatrixType,Eigen::IncompleteLUT<double>> SolverType;
      const std::string SolverName = "BiCGSTAB with IncompleteLUT";
      #elif defined WITH_PASTIX
      typedef Eigen::PastixLU<SystemMatrixType> SolverType;
      const std::string SolverName = "PastixLU";
      #elif defined WITH_UMFPACK
      typedef Eigen::UmfPackLU<SystemMatrixType> SolverType;
      const std::string SolverName = "UmfPackLU";
      #elif defined WITH_MKL
      typedef Eigen::PardisoLU<SystemMatrixType> SolverType;
      const std::string SolverName = "PardisoLU";
      #else
      typedef Eigen::SparseLU<SystemMatrixType,Eigen::COLAMDOrdering<int> > SolverType;
      const std::string SolverName = "SparseLU";
      #endif // ITTERATIVE
      typedef std::function<VectorRd(const VectorRd &)> SourceFunctionType;

      // Constructor
      StokesProblem(const Mesh &mesh, const size_t degree, bool _use_threads = true, std::ostream & output = std::cout) 
        : use_threads(_use_threads), m_output(output),
          m_stokescore(mesh,degree),
          m_xnabla(m_stokescore),
          m_xsl(m_stokescore),
          m_xvl(m_stokescore),
          m_DDOFs(Eigen::VectorXi::Zero(m_xnabla.dimension()+m_xsl.dimension())),
          m_Dval(Eigen::VectorXd::Zero(m_xnabla.dimension()+m_xsl.dimension())) {
          setup_dofsmap();
      }
      
      /// Return the dimension of solutions 
      size_t systemEffectiveDim() const 
      {
        return systemTotalDim() + m_dimH - m_dimDBC;
      }

      /// Return the dimension of the dofs in the tensor product spaces, excluding harmonics spaces
      size_t systemTotalDim() const 
      {
        return m_xnabla.dimension() + m_xsl.dimension();
      }
      
      /// Set the space of harmonics forms, must be called before assemble_system
      void setup_Harmonics(Harmonics_premade htype);
      /// Assemble the system, compute the RHS at the same time if a source is provided
      void assemble_system(const SourceFunctionType &f = nullptr,size_t degree = 0);
      /// Set rhs vector from function
      void set_rhs (const SourceFunctionType &f, size_t degree = 0);
      /// Set rhs vector from vector 
      void set_rhs (const Eigen::VectorXd &rhs);
      /// Setup the solver
      void compute();
      /// Solve the system and store the solution in the given vector
      Eigen::VectorXd solve();
      /// Solve the system for the given rhs and store the solution in the given vector
      Eigen::VectorXd solve(const Eigen::VectorXd &rhs);
      #if defined ITTERATIVE_LU or defined ITTERATIVE
      Eigen::VectorXd solve_with_guess(const Eigen::VectorXd &Guess);
      #endif
      // Take a vector without dirichlets and return a vector with dirichlet value reinserted (systemEffectiveDim() -> systemTotalDim();
      Eigen::VectorXd reinsertDirichlet(const Eigen::VectorXd &u) const;
      // Set Dirichlet from function, f must return true on Dirichlet boundary.
      // Automacally calls setup_dofsmap()
      void set_Dirichlet_boundary(std::function<bool(const VectorRd &)> const & f);

      /// Return the core
      const StokesCore & stokescore() const 
      {
        return m_stokescore;
      }

      /// Return XNabla space
      const XNablaStokes & xnabla() const 
      {
        return m_xnabla;
      }
      
      /// Return XSL space
      const XSLStokes & xsl() const 
      {
        return m_xsl;
      }
      
      /// Return XVL space
      const XVLStokes & xvl() const 
      {
        return m_xvl;
      }

      const Eigen::VectorXi & DDOFs() const
      {
        return m_DDOFs;
      }
      
      Eigen::VectorXi & DDOFs()
      {
        return m_DDOFs;
      }
      
      void setup_Dirichlet_everywhere(); // Set DDOFs to enforce a Dirichlet condition on the whole boundary, automatically call setup_dofsmap()
      void interpolate_boundary_value(XNablaStokes::FunctionType const & f, size_t degree = 0); // setup boundary value by interpolating a function; rhs must be recomputed after any change made to the boundary values
      void setup_Dirichlet_values(Eigen::VectorXd const &vals); // Interpolation on xnabla + xsl of the target values; rhs must be recomputed after any change made to the boundary values

      void setup_dofsmap(); // Call after editing DDOFs to register the changes
      bool use_threads;
    private:
      std::ostream & m_output;
      SystemMatrixType m_system;
      SystemMatrixType m_bdrMatrix;
      Eigen::VectorXd m_rhs;
      SolverType m_solver;
      StokesCore m_stokescore;
      XNablaStokes m_xnabla;
      XSLStokes m_xsl;
      XVLStokes m_xvl;
      Eigen::VectorXi m_DDOFs; //DDOFs : 1 if Dirichlet dof, 0 otherwise
      Eigen::VectorXd m_Dval;
      size_t m_dimH = 0;
      Harmonics_premade m_htype = Harmonics_premade::None;
      size_t m_dimDBC = 0;
      Eigen::VectorXi m_DDOFs_map; //mapping generated from DDOFs converting dofs including Dirichlets to dofs excluding Dirichlet dofs (and -1 to Dirichlet dofs)

      /// Compute the gram_matrix of the integral int_T v, v in Polyk3po(T) on the left
      Eigen::MatrixXd compute_IntXNabla(size_t iT) const;
      /// Compute the gram_matrix of the integral int_F q, q in Polyk on the left
      Eigen::MatrixXd compute_IntXSL(size_t iT) const;
      /// Compute source for the rhs
      Eigen::MatrixXd compute_IntPf(size_t iT, const SourceFunctionType & f, size_t degree) const;
  }; // End of StokesProblem

  void StokesProblem::setup_dofsmap() {
    m_DDOFs_map.resize(systemTotalDim());
    size_t acc = 0; // number of unknown corresponding to Dirichlet condition
    for (size_t itt = 0; itt < systemTotalDim(); itt++) {
      if (m_DDOFs(itt) == 0) {
        m_DDOFs_map(itt) = itt - acc;
      } else {
        m_DDOFs_map(itt) = -1;
        acc++;
      }
    }
    m_dimDBC = acc;
  }

  void StokesProblem::assemble_system(const SourceFunctionType &f, size_t degree) {
    const bool f_exists = f != nullptr;
    //m_output << ((f_exists)? " Assembling rhs ..." : " Skipping rhs assembly as no source is provided.")<<std::flush;
    
    // Callback function to parallel_assembly
    std::function<void(size_t start,size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs,std::list<Eigen::Triplet<double>> *triplets_bdr, Eigen::VectorXd *vec2)> batch_local_assembly = [this,f,f_exists,degree](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs,std::list<Eigen::Triplet<double>> *triplets_bdr, Eigen::VectorXd *)->void {
      for (size_t iT = start; iT < end; iT++) {
        const Cell & T = *m_xnabla.mesh().cell(iT);
        Eigen::MatrixXd loca = m_xvl.computeL2Product_GG(iT,m_xnabla); // (Gv,Gu)
        Eigen::MatrixXd locb = m_xnabla.cellOperators(iT).divergence.transpose()*m_xsl.compute_Gram_Cell(iT); // (Dv,p)
        //Harmonic constrain
        Eigen::MatrixXd locvh;
        switch(m_htype) {
          case(Harmonics_premade::Velocity):
            locvh = m_xnabla.cellOperators(iT).potential.transpose()*compute_IntXNabla(iT); // int_T P v
            break;
          case(Harmonics_premade::Pressure):
            locvh = compute_IntXSL(iT);
            break;
          default:
            locvh = Eigen::MatrixXd();
        }
        Eigen::VectorXd locR;
        if (f_exists) { // source term provided
          size_t dqr = (degree > 0)? degree : 2*m_stokescore.degree() + 3;
          locR = m_xnabla.cellOperators(iT).potential.transpose()*compute_IntPf(iT,f,dqr);
        }
        // dofs_map
        std::vector<size_t> dofmap_xnabla = m_xnabla.globalDOFIndices(T);
        std::vector<size_t> dofmap_xsl = m_xsl.globalDOFIndices(T);
        size_t dim_xnabla = m_xnabla.dimension();
        /// Global system :
        //  (Gv,Gu) & -(Dv,p) & [(Pv,h)]
        //  (q,Du)  &    0    & [(q,h)]
        //  [(h,Pu)]& [(h,p)] &    0
        for (size_t i = 0; i < m_xnabla.dimensionCell(iT);i++) { // A[1,:]
          int gi = m_DDOFs_map(dofmap_xnabla[i]); // global location of i after removal of BC dofs
          if (gi < 0) continue; // BC dofs
          for (size_t j = 0; j < m_xnabla.dimensionCell(iT);j++) { // A[1,1]
            if (std::abs(loca(i,j)) < prune_threshold) continue; // does not contribute 
            int gj = m_DDOFs_map(dofmap_xnabla[j]);
            if (gj < 0) { // BC contribution to RHS
              triplets_bdr->emplace_back(gi,dofmap_xnabla[j],loca(i,j)); 
            } else { 
              triplets->emplace_back(gi,gj,loca(i,j));
            }
          } // for j in XNabla
          for (size_t j = 0; j < m_xsl.dimensionCell(iT);j++) { // A[1,2]
            if (std::abs(locb(i,j)) < prune_threshold) continue;
            int gj = m_DDOFs_map(dim_xnabla + dofmap_xsl[j]); // global loc of j in XSL
            if (gj < 0) { // BC contribution to RHS
              triplets_bdr->emplace_back(gi,dim_xnabla + dofmap_xsl[j],-locb(i,j));
            } else {
              triplets->emplace_back(gi,gj,-locb(i,j));
            }
          } // for j in XSL
          // Harmonic part
          if (m_htype == Harmonics_premade::Velocity) {
            for (size_t j = 0; j < m_dimH; j++) { // A[1,3]
              if (std::abs(locvh(i,j)) < prune_threshold) continue;
              int gj = systemEffectiveDim() - m_dimH + j;
              triplets->emplace_back(gi,gj,locvh(i,j));
            }
          }
          // RHS
          if (f_exists) {
            (*rhs)(gi) += locR(i); // add
          }
        } // i in XNabla
        for (size_t i = 0; i < m_xsl.dimensionCell(iT); i++) { // i in XSL (discontinuous Pk(T))
          int gi = m_DDOFs_map(dim_xnabla+ dofmap_xsl[i]);
          if (gi < 0) continue; __UNLIKELY // BC inside cell
          for (size_t j = 0; j < m_xnabla.dimensionCell(iT);j++) { // A[2,1]
            if (std::abs(locb(j,i)) < prune_threshold) continue;
            int gj = m_DDOFs_map(dofmap_xnabla[j]);
            if (gj < 0) { 
              triplets_bdr->emplace_back(gi,dofmap_xnabla[j],locb(j,i)); // transpose
            } else {
              triplets->emplace_back(gi,gj,locb(j,i));
            }
          }
          // Nothing in XSLxXSL
          if (m_htype == Harmonics_premade::Pressure) {
            for (size_t j = 0; j < m_dimH; j++) {
              if (std::abs(locvh(i,j)) < prune_threshold) continue;
              int gj = systemEffectiveDim() - m_dimH + j;
              triplets->emplace_back(gi,gj,locvh(i,j));
            }
          } // harmonics
        } // i in XSL
        for (size_t i = 0; i < m_dimH; i++) { // i in H
          int gi = systemEffectiveDim() - m_dimH + i;
          if (m_htype == Harmonics_premade::Velocity) {
            for (size_t j = 0; j < m_xnabla.dimensionCell(iT); j++) { // A[3,1]
              if (std::abs(locvh(j,i)) < prune_threshold) continue;
              int gj = m_DDOFs_map(dofmap_xnabla[j]);
              if (gj < 0) {
                triplets_bdr->emplace_back(gi,dofmap_xnabla[j],locvh(j,i));
              } else {
                triplets->emplace_back(gi,gj,locvh(j,i));
              }
            }
          } else if (m_htype == Harmonics_premade::Pressure) {
            for (size_t j = 0; j < m_xsl.dimensionCell(iT); j++) { // A[3,2]
              if (std::abs(locvh(j,i)) < prune_threshold) continue;
              int gj = m_DDOFs_map(dim_xnabla + dofmap_xsl[j]);
              if (gj < 0) { __UNLIKELY
                triplets_bdr->emplace_back(gi,dim_xnabla + dofmap_xsl[j],locvh(j,i));
              } else { __LIKELY
                triplets->emplace_back(gi,gj,locvh(j,i));
              }
            }
          }
        } // i in H
      } // iT
    }; // std::function batch_local_assembly
    m_output << "[StokesProblem] Assembling global system from local contributions..."<<std::flush;
    Eigen::VectorXd dummy;
    std::tie(m_system,m_rhs,m_bdrMatrix,dummy) = parallel_assembly_system(m_stokescore.mesh().n_cells(),systemEffectiveDim(),std::make_pair(systemEffectiveDim(),systemTotalDim()),0,batch_local_assembly,use_threads);
    // Incorporate contribution from BC into RHS
    m_rhs -= m_bdrMatrix*m_Dval;
    m_output << "\r[StokesProblem] Assembled global system                              "<<std::endl;
  }

  void StokesProblem::set_rhs (const SourceFunctionType &f, size_t degree) {
    size_t dqr = (degree > 0) ? degree : 2*m_stokescore.degree() + 3;

    std::function<void(size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)> batch_local_assembly = [this,f,&dqr](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)->void {
      for (size_t iT = start; iT < end; iT++) {
        const Cell & T = *m_xnabla.mesh().cell(iT);
        Eigen::VectorXd locR = m_xnabla.cellOperators(iT).potential.transpose()*compute_IntPf(iT,f,dqr);
        std::vector<size_t> dofmap_xnabla = m_xnabla.globalDOFIndices(T);
        for (size_t i = 0; i < m_xnabla.dimensionCell(iT); i++) { // i in XNabla
          int gi = m_DDOFs_map(dofmap_xnabla[i]);
          if (gi >= 0) 
            (*rhs)(gi) += locR(i);
        }
      }
    };

    m_rhs = parallel_assembly_system(m_stokescore.mesh().n_cells(),systemEffectiveDim(),batch_local_assembly,use_threads).second;
    // Incorporate the value Dirichlet dofs
    m_rhs -= m_bdrMatrix*m_Dval;
  }

  void StokesProblem::set_rhs (const Eigen::VectorXd &rhs) {
    if (rhs.size() != m_rhs.size()) {
      std::cerr << "[StokesProblem] Setting rhs from vector failed, size dismatched. Expected :"<<m_rhs.size()<<" got :"<<rhs.size()<<std::endl;
      return;
    }
    m_rhs = rhs;
  }
  void StokesProblem::compute() {
    m_output << "[StokesProblem] Setting solver "<<SolverName<<" with "<<systemEffectiveDim()<<" degrees of freedom"<<std::endl;
    #if defined ITTERATIVE_LU
    m_solver.preconditioner().setDroptol(1.e-9);
    #endif
    m_solver.compute(m_system);
    if (m_solver.info() != Eigen::Success) {
      std::cerr << "[StokesProblem] Failed to factorize the system" << std::endl;
      throw std::runtime_error("Factorization failed");
    }
  }
      
  Eigen::VectorXd StokesProblem::solve() {
    m_output << "[StokesProblem] Solving" <<std::endl;
    Eigen::VectorXd u = m_solver.solve(m_rhs);
    if (m_solver.info() != Eigen::Success) {
      std::cerr << "[StokesProblem] Failed to solve the system" << std::endl;
      throw std::runtime_error("Solve failed");
    }
    #ifdef CHECK_SOLVE
    double error = (m_system*u - m_rhs).norm();
    m_output<< "Absolute error :" << error << " Relative error :" << error/m_rhs.norm() << std::endl;
    #endif
    return u;
  }

  Eigen::VectorXd StokesProblem::solve(const Eigen::VectorXd &rhs) {
    m_output << "[StokesProblem] Solving" <<std::endl;
    Eigen::VectorXd u = m_solver.solve(rhs);
    if (m_solver.info() != Eigen::Success) {
      std::cerr << "[StokesProblem] Failed to solve the system" << std::endl;
      throw std::runtime_error("Solve failed");
    }
    #ifdef CHECK_SOLVE
    double error = (m_system*u - m_rhs).norm();
    m_output<< "Absolute error :" << error << " Relative error :" << error/m_rhs.norm() << std::endl;
    #endif
    return u;
  }

  #if defined ITTERATIVE_LU or defined ITTERATIVE
  Eigen::VectorXd StokesProblem::solve_with_guess(const Eigen::VectorXd &guess) {
    if (guess.size() != (long int)systemTotalDim()) {
      std::cerr << "[StokesProblem] solve_with_guess assumes vector of size systemTotalDim" << std::endl;
      throw std::runtime_error("Wrong guess size");
    }
    Eigen::VectorXd bcguess = Eigen::VectorXd::Zero(systemEffectiveDim());
    for (size_t i = 0; i < systemTotalDim(); i++) {
      if (m_DDOFs_map(i) < 0) continue;
      bcguess(m_DDOFs_map(i)) = guess(i);
    }
    m_output << "[StokesProblem] Solving with Guess" <<std::endl;
    Eigen::VectorXd u = m_solver.solveWithGuess(m_rhs,bcguess);
    if (m_solver.info() != Eigen::Success) {
      std::cerr << "[StokesProblem] Failed to solve the system" << std::endl;
      throw std::runtime_error("Solve failed");
    }
    #ifdef CHECK_SOLVE
    double error = (m_system*u - m_rhs).norm();
    m_output<< "Absolute error :" << error << " Relative error :" << error/m_rhs.norm() << std::endl;
    #endif
    return u;
  }
  #endif

  Eigen::VectorXd StokesProblem::reinsertDirichlet(const Eigen::VectorXd &u) const {
    assert(u.size() == (int)systemEffectiveDim());
    Eigen::VectorXd rv = Eigen::VectorXd::Zero(systemTotalDim());
    size_t acc = 0;
    for (size_t itt = 0; itt < systemTotalDim(); itt++) {
      if (m_DDOFs(itt) > 0) { // Dirichlet dof, skip it
        rv(itt) = m_Dval(itt);
        acc++;
      } else {
        rv(itt) = u(itt - acc);
      }
    }
    return rv;
  }

template<typename Type> bool inside_domain(std::function<bool(const VectorRd &)> const & f,const Type &T) {
  bool rv = true;
  for (size_t iV = 0; iV < T.n_vertices();iV++) {
    rv &= f(T.vertex(iV)->coords()); // stay true iff all vertex are inside the domain of f
  }
  return rv;
}


  void StokesProblem::set_Dirichlet_boundary(std::function<bool(const VectorRd &)> const & f) {
    m_DDOFs.setZero();
    const Mesh &mesh = m_stokescore.mesh();
    // Itterate over vertices
    for (size_t iV = 0; iV < mesh.n_vertices();iV++) {
      const Vertex &V = *mesh.vertex(iV);
      if (not V.is_boundary() || not f(V.coords())) continue;
      size_t offset = V.global_index()*m_xnabla.numLocalDofsVertex();
      for (size_t i = 0; i < m_xnabla.numLocalDofsVertex();i++) {
        m_DDOFs(offset + i) = 1;
      }
      // No dofs on vertices of xsl
    }
    // Itterate over edges
    for (size_t iE = 0; iE < mesh.n_edges();iE++) {
      const Edge &E = *mesh.edge(iE);
      if (not E.is_boundary() || not (f(E.vertex(0)->coords()) && f(E.vertex(1)->coords()))) continue;
      size_t offset = mesh.n_vertices()*m_xnabla.numLocalDofsVertex() + E.global_index()*m_xnabla.numLocalDofsEdge();
      for (size_t i = 0; i < m_xnabla.numLocalDofsEdge();i++) {
        m_DDOFs(offset + i) = 1;
      }
      // No dofs on edges of xsl
    }
    // Itterate over faces
    for (size_t iF = 0; iF < mesh.n_faces();iF++) {
      const Face &F = *mesh.face(iF);
      if (not F.is_boundary() || not inside_domain(f,F)) continue; 
      size_t offset = mesh.n_vertices()*m_xnabla.numLocalDofsVertex() + mesh.n_edges()*m_xnabla.numLocalDofsEdge() + F.global_index()*m_xnabla.numLocalDofsFace();
      for (size_t i = 0; i < m_xnabla.numLocalDofsFace(); i++) {
        m_DDOFs(offset + i) = 1;
      }
      // No dofs on faces of xsl
    }
    setup_dofsmap();
  }

  // Simple warper to apply BC everywhere 
  void StokesProblem::setup_Dirichlet_everywhere() {
    std::function<bool(const VectorRd &)> f = [](const VectorRd &)->bool {return true;};
    set_Dirichlet_boundary(f);
  }

  // Compute the matrix giving the integral of P agaisnt [x,y,z], P in Pk3po 
  Eigen::MatrixXd StokesProblem::compute_IntXNabla(size_t iT) const {
    Cell &T = *m_stokescore.mesh().cell(iT);
    QuadratureRule quad_kpo_T = generate_quadrature_rule(T,m_stokescore.degree() + 1);
    auto basis_Pk3po_T_quad = evaluate_quad<Function>::compute(*m_stokescore.cellBases(iT).Polyk3po,quad_kpo_T);
    Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(m_stokescore.cellBases(iT).Polyk3po->dimension(),dimspace);
    for (size_t i = 0; i < basis_Pk3po_T_quad.shape()[0]; i++) {
      for (size_t iqn = 0; iqn < quad_kpo_T.size(); iqn++) {
        rv.row(i) += quad_kpo_T[iqn].w*basis_Pk3po_T_quad[i][iqn];
      }
    }
    return rv;
  }

  // Compute the vector giving the integral of q, q in xsl
  Eigen::MatrixXd StokesProblem::compute_IntXSL(size_t iT) const {
    Cell &T = *m_stokescore.mesh().cell(iT);
    QuadratureRule quad_k_T = generate_quadrature_rule(T,m_stokescore.degree());
    auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*m_stokescore.cellBases(iT).Polyk,quad_k_T);
    Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(m_stokescore.cellBases(iT).Polyk->dimension(),1);
    for (size_t i = 0; i < basis_Pk_T_quad.shape()[0]; i++) {
      for (size_t iqn = 0; iqn < quad_k_T.size(); iqn++) {
        rv(i,0) += quad_k_T[iqn].w*basis_Pk_T_quad[i][iqn];
      }
    }
    return rv;
  }

  // Return the evaluation of the integral of f against each elements of the basis Polyk3po
  Eigen::MatrixXd StokesProblem::compute_IntPf(size_t iT, const SourceFunctionType & f,size_t degree) const {
    Cell &T = *m_stokescore.mesh().cell(iT);
    QuadratureRule quad_dqr_T = generate_quadrature_rule(T,m_stokescore.degree() + 1 + degree);
    auto basis_Pk3po_T_quad = evaluate_quad<Function>::compute(*m_stokescore.cellBases(iT).Polyk3po,quad_dqr_T);
    std::vector<VectorRd> intf;
    intf.resize(quad_dqr_T.size()); // store the value of f at each node
    for (size_t iqn = 0; iqn < quad_dqr_T.size(); iqn++) {
      intf[iqn] = f(quad_dqr_T[iqn].vector());
    }
    Eigen::VectorXd rv = Eigen::VectorXd::Zero(m_stokescore.cellBases(iT).Polyk3po->dimension());
    for (size_t i = 0; i < basis_Pk3po_T_quad.shape()[0]; i++) {
      for (size_t iqn = 0; iqn < quad_dqr_T.size(); iqn++) {
        rv(i) += quad_dqr_T[iqn].w*(basis_Pk3po_T_quad[i][iqn]).dot(intf[iqn]);
      }
    }
    return rv;
  }

  void StokesProblem::setup_Harmonics(Harmonics_premade htype) {
    m_htype = htype;
    switch(htype) {
      case (Harmonics_premade::None) :
        m_dimH = 0;
        return;
      case (Harmonics_premade::Velocity) :
        m_dimH = dimspace;
        return;
      case (Harmonics_premade::Pressure) :
        m_dimH = 1;
        return;
      default :
        m_dimH = 0;
        m_output << "[StokesProblem] Warning : harmonics type not yet implemented" << std::endl;
        return;
    }
  }

  void StokesProblem::interpolate_boundary_value(XNablaStokes::FunctionType const & f,size_t degree) {
    size_t dqr = (degree > 0) ? degree : 2*m_xnabla.degree() + 3;
    m_Dval.head(m_xnabla.dimension()) = m_xnabla.interpolate(f,dqr);
  }

  void StokesProblem::setup_Dirichlet_values(Eigen::VectorXd const &vals) {
    assert(m_Dval.size() == vals.size());
    m_Dval = vals;
  }

///-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Helper to analyse
///-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  double Norm_H1p(const XNablaStokes &xnabla, const XVLStokes &xvl,const Eigen::VectorXd &v,bool use_threads=true) {
    Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
    std::function<void(size_t,size_t)> compute_local_squarednorms = [&xnabla,&xvl,&v,&local_sqnorms](size_t start,size_t end)->void {
      for (size_t iT = start;iT < end; iT++) {
        local_sqnorms[iT] = (xnabla.restrictCell(iT,v)).dot((xnabla.computeL2Product(iT) + xvl.computeL2Product_GG(iT,xnabla))*xnabla.restrictCell(iT,v));
      }
    };
    parallel_for(xnabla.mesh().n_cells(),compute_local_squarednorms,use_threads);

    return std::sqrt(std::abs(local_sqnorms.sum()));
  }

  double Norm_L2s(const XSLStokes &xsl,const Eigen::VectorXd &v,bool use_threads=true) {
    Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xsl.mesh().n_cells());
    std::function<void(size_t,size_t)> compute_local_squarednorms = [&xsl,&v,&local_sqnorms](size_t start,size_t end)->void {
      for (size_t iT = start;iT < end; iT++) {
        local_sqnorms[iT] = xsl.restrictCell(iT,v).dot(xsl.compute_Gram_Cell(iT)*xsl.restrictCell(iT,v));
      }
    };
    parallel_for(xsl.mesh().n_cells(),compute_local_squarednorms,use_threads);

    return std::sqrt(std::abs(local_sqnorms.sum()));
  }

///-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Helper to export function to vtu
///-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  // Uniformize the size computation between scalar and VectorRd
  template<typename type> size_t get_sizeof() {
    if constexpr (std::is_same<type,double>::value) { // if constexpr in template with a condition not value-dependent after instantiation then the discarded block is not instantiated. Else we would fail to compile double::SizeAtCompileTime;
      return 1;
    } else {
      return type::SizeAtCompileTime;
    }
  }

  // The first argument of a member function is this* 
  template<typename Core>
  Eigen::VectorXd get_vertices_values(const Core &core, const Eigen::VectorXd &vh) {
    size_t size_rv = get_sizeof<typename std::invoke_result<decltype(&Core::evaluatePotential),Core*,size_t,const Eigen::VectorXd &,const VectorRd &>::type>();
    Eigen::VectorXd vval = Eigen::VectorXd::Zero(size_rv*core.mesh().n_vertices());
    for (size_t i = 0; i < core.mesh().n_vertices();i++) {
      size_t adjcell_id = core.mesh().vertex(i)->cell(0)->global_index();
      vval.segment(i*size_rv,size_rv) << core.evaluatePotential(adjcell_id,core.restrictCell(adjcell_id,vh),core.mesh().vertex(i)->coords());
    }
    return vval;
  }
  template<typename F,typename Core>
  Eigen::VectorXd evaluate_vertices_values(const F &f,const Core &core) {
    size_t size_rv = get_sizeof<typename std::invoke_result<F,const Eigen::VectorXd &>::type>();
    Eigen::VectorXd vval = Eigen::VectorXd::Zero(size_rv*core.mesh().n_vertices());
    for (size_t i = 0; i < core.mesh().n_vertices();i++) {
      vval.segment(i*size_rv,size_rv) << f(core.mesh().vertex(i)->coords());
    }
    return vval;
  }


} // namespace
#endif // STOKES_HPP
