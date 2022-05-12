#ifndef STOKES_HPP
#define STOKES_HPP

#include <hardcore3d_petsc.hpp>

#include <stokescore.hpp>
#include <xnablastokes.hpp>
#include <xvlstokes.hpp>

#include <parallel_for.hpp>
#include <display_timer.hpp>

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
  
  constexpr double prune_threshold = 1e-30;
  
  // Preset for harmonics forms, to add other inplement their cases in assemble_system and setup_Harmonics
  enum Harmonics_premade {
    None,
    Velocity,
    Pressure,
    Custom
  };

template<typename Type> bool inside_domain(std::function<bool(const VectorRd &)> const & f,const Type &T);

  class StokesProblem {
    public:
      typedef std::function<VectorRd(const VectorRd &)> SourceFunctionType;

      // Constructor
      StokesProblem(const Mesh &mesh, const size_t degree, bool _use_threads = true, std::ostream & output = std::cout, bool _with_timer = true) 
        : use_threads(_use_threads),
          m_timer_hist(_with_timer), m_with_timer(_with_timer),
          m_output(output), m_solver(PBTYPE_STOKES),
          m_stokescore(mesh,degree),
          m_xnabla(m_stokescore,use_threads,m_output),
          m_xsl(m_stokescore,use_threads,m_output),
          m_xvl(m_stokescore,use_threads,m_output),
          m_DDOFs(Eigen::VectorXi::Zero(m_xnabla.dimension()+m_xsl.dimension())),
          m_Dval(Eigen::VectorXd::Zero(m_xnabla.dimension()+m_xsl.dimension())) {
          if (m_with_timer) m_timer_hist.stop("Construct spaces");
          setup_dofsmap();
      }      

      // Constructor
      StokesProblem(const Mesh &mesh, const size_t degree, const std::string &from_file, bool _use_threads = true, std::ostream & output = std::cout, bool _with_timer = true) 
        : use_threads(_use_threads),
          m_timer_hist(_with_timer), m_with_timer(_with_timer),
          m_output(output), m_solver(PBTYPE_STOKES),
          m_stokescore(mesh,degree),
          m_xnabla(m_stokescore,from_file,use_threads,m_output),
          m_xsl(m_stokescore,use_threads,m_output),
          m_xvl(m_stokescore,use_threads,m_output),
          m_DDOFs(Eigen::VectorXi::Zero(m_xnabla.dimension()+m_xsl.dimension())),
          m_Dval(Eigen::VectorXd::Zero(m_xnabla.dimension()+m_xsl.dimension())) {
          if (m_with_timer) m_timer_hist.stop("Construct spaces");
          setup_dofsmap();
      }
      
      /// Return the dimension of solutions 
      size_t systemEffectiveDim() const 
      {
        return systemTotalDim() - m_dimDBC; // We use KSPNullSpace instead of increasing the dimensino with dimH
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
      /// Add Neumann contribution to rhs
      void set_neumann(std::function<bool(const VectorRd &)> const &fb,const SourceFunctionType &fdun, size_t degree = 0);
      /// Set rhs vector from function, reset the previous rhs
      void set_rhs (const SourceFunctionType &f, size_t degree = 0);
      /// Setup the solver
      void compute();
      /// Solve the system and return the solution in the given vector
      Eigen::VectorXd solve();
      Eigen::VectorXd solve_with_guess(const Eigen::VectorXd &Guess);

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

      PETscSolver & solver()
      {
        return m_solver;
      }
      
      void setup_Dirichlet_everywhere(); // Set DDOFs to enforce a Dirichlet condition on the whole boundary, automatically call setup_dofsmap()
      void interpolate_boundary_value(XNablaStokes::FunctionType const & f, size_t degree = 0); // setup boundary value by interpolating a function; rhs must be recomputed after any change made to the boundary values
      void setup_Dirichlet_values(Eigen::VectorXd const &vals); // Interpolation on xnabla + xsl of the target values; rhs must be recomputed after any change made to the boundary values

      void setup_dofsmap(); // Call after editing DDOFs to register the changes
      bool use_threads;

      timer_hist m_timer_hist;
    private:
      bool m_with_timer;
      std::ostream & m_output;
      PETscSolver m_solver;
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
    // Structure to store the data
    std::vector<std::forward_list<std::tuple<int,int,double>>> mat_triplets(m_stokescore.mesh().n_cells());
    std::vector<std::forward_list<std::tuple<int,int,double>>> bdr_triplets(m_stokescore.mesh().n_cells());
    std::atomic<PetscInt>* nnz_mat = new std::atomic<PetscInt>[systemEffectiveDim()];
    std::atomic<PetscInt>* nnz_mat_diag = new std::atomic<PetscInt>[systemEffectiveDim()];
    std::atomic<PetscInt>* nnz_bdr = new std::atomic<PetscInt>[systemEffectiveDim()];
    std::atomic<PetscInt>* nnz_bdr_diag = new std::atomic<PetscInt>[systemEffectiveDim()];
    std::atomic<double>* rhs = new std::atomic<double>[systemEffectiveDim()];
    // Null space
    std::atomic<double>** nullspace = static_cast<std::atomic<double>**>(malloc(sizeof(std::atomic<double>*)*m_dimH));
    for (size_t i = 0; i < m_dimH; i++) {
      nullspace[i] = new std::atomic<double>[systemEffectiveDim()];
    }

    // Initialize
    for (size_t i = 0; i < systemEffectiveDim(); i++) {
      nnz_mat[i] = 0; 
      nnz_mat_diag[i] = 0; 
      nnz_bdr[i] = 0;
      nnz_bdr_diag[i] = 0;
      rhs[i] = 0.;
      for (size_t j = 0; j < m_dimH;j++) {
        nullspace[j][i] = 0.;
      }
    }
    // Init sizes
    int last_loc_size, loc_size, last_loc_size_bdr, loc_size_bdr;
    m_solver.Compute_MPISizes(systemEffectiveDim(),systemTotalDim(),&loc_size,&last_loc_size,&loc_size_bdr,&last_loc_size_bdr);
    
    // Callback function to parallel_assembly
    std::function<void(size_t start,size_t end)> batch_local_assembly = [this,f,f_exists,degree,&mat_triplets,&bdr_triplets,
                                                                         nnz_mat,nnz_mat_diag,nnz_bdr,nnz_bdr_diag,rhs,
                                                                         nullspace](size_t start, size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        const Cell & T = *m_xnabla.mesh().cell(iT);
        Eigen::MatrixXd loca = m_xvl.computeL2Product_GG(iT,m_xnabla); // (Gv,Gu)
        Eigen::MatrixXd locb = m_xnabla.cellOperators(iT).divergence.transpose()*m_xsl.compute_Gram_Cell(iT); // (Dv,p)
        // Symmetrize (should be symmetric but is not, linked to operation order?)
        loca = 0.5*loca;
        loca += loca.transpose().eval();
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
              if (m_solver.is_in_diag_bdr(gi,dofmap_xnabla[j])) {
                nnz_bdr_diag[gi]++;
              } else {
                nnz_bdr[gi]++; // atomic increment
              }
              bdr_triplets[iT].emplace_front(std::make_tuple(gi,dofmap_xnabla[j],loca(i,j)));
            } else { 
              if (m_solver.is_in_diag(gi,gj)){
                nnz_mat_diag[gi]++;
              } else {
                nnz_mat[gi]++;
              }
              mat_triplets[iT].emplace_front(std::make_tuple(gi,gj,loca(i,j)));
            }
          } // for j in XNabla
          for (size_t j = 0; j < m_xsl.dimensionCell(iT);j++) { // A[1,2]
            if (std::abs(locb(i,j)) < prune_threshold) continue;
            int gj = m_DDOFs_map(dim_xnabla + dofmap_xsl[j]); // global loc of j in XSL
            if (gj < 0) { // BC contribution to RHS
              if (m_solver.is_in_diag_bdr(gi,dim_xnabla + dofmap_xsl[j])) {
                nnz_bdr_diag[gi]++;
              } else {
                nnz_bdr[gi]++;
              }
              bdr_triplets[iT].emplace_front(std::make_tuple(gi,dim_xnabla + dofmap_xsl[j],-locb(i,j)));
            } else {
              if (m_solver.is_in_diag(gi,gj)){
                nnz_mat_diag[gi]++;
              } else {
                nnz_mat[gi]++;
              }
              mat_triplets[iT].emplace_front(std::make_tuple(gi,gj,-locb(i,j)));
            }
          } // for j in XSL
          // RHS
          if (f_exists) {
            rhs[gi].fetch_add(locR(i)); // atomic add
          }
        } // i in XNabla
        for (size_t i = 0; i < m_xsl.dimensionCell(iT); i++) { // i in XSL (discontinuous Pk(T))
          int gi = m_DDOFs_map(dim_xnabla+ dofmap_xsl[i]);
          if (gi < 0) __UNLIKELY continue; // BC inside cell
          for (size_t j = 0; j < m_xnabla.dimensionCell(iT);j++) { // A[2,1]
            if (std::abs(locb(j,i)) < prune_threshold) continue;
            int gj = m_DDOFs_map(dofmap_xnabla[j]);
            if (gj < 0) { 
              if (m_solver.is_in_diag_bdr(gi,dofmap_xnabla[j])) {
                nnz_bdr_diag[gi]++;
              } else {
                nnz_bdr[gi]++;
              }
              bdr_triplets[iT].emplace_front(std::make_tuple(gi,dofmap_xnabla[j],-locb(j,i))); // transpose
            } else {
              if (m_solver.is_in_diag(gi,gj)) {
                nnz_mat_diag[gi]++;
              } else {
                nnz_mat[gi]++;
              }
              mat_triplets[iT].emplace_front(std::make_tuple(gi,gj,-locb(j,i)));
            }
          }
        } // i in XSL

        //Harmonic constrain
        switch(m_htype) {
          case(Harmonics_premade::Velocity):
            throw std::runtime_error("Dev: TODO Implement harmonics with KSPNullSpace");
            //locvh = m_xnabla.cellOperators(iT).potential.transpose()*compute_IntXNabla(iT); // int_T P v
            break;
          case(Harmonics_premade::Pressure):
            {
              Eigen::MatrixXd locvh = compute_IntXSL(iT);
              for (size_t i = 0; i < m_xsl.dimensionCell(iT);i++) {
                int gi = m_DDOFs_map (dim_xnabla + dofmap_xsl[i]);
                nullspace[0][gi].fetch_add(locvh(i));
              }
            }
            break;
          default:
            ;
        }
      } // iT
    }; // std::function batch_local_assembly
    m_output << "[StokesProblem] Assembling global system from local contributions..."<<std::flush;

// Intended for nnz analysis
/*
    int dim = systemEffectiveDim();
    int dofV = m_stokescore.mesh().n_vertices()*m_xnabla.numLocalDofsVertex();
    for (size_t i = dofV; i < systemTotalDim();i++) {
      dofV = m_DDOFs_map(i);
      if (dofV > 0) break;
    }
    int dofE = m_stokescore.mesh().n_vertices()*m_xnabla.numLocalDofsVertex() 
              + m_stokescore.mesh().n_edges()*m_xnabla.numLocalDofsEdge();
    for (size_t i = dofE; i < systemTotalDim();i++) {
      dofE = m_DDOFs_map(i);
      if (dofE > 0) break;
    }
    int dofF = m_stokescore.mesh().n_vertices()*m_xnabla.numLocalDofsVertex() 
              + m_stokescore.mesh().n_edges()*m_xnabla.numLocalDofsEdge()
              + m_stokescore.mesh().n_faces()*m_xnabla.numLocalDofsFace();
    for (size_t i = dofF; i < systemTotalDim();i++) {
      dofF = m_DDOFs_map(i);
      if (dofF > 0) break;
    }
    int dim_xn = systemEffectiveDim() - m_xsl.dimension();
    std::cout<<std::endl;
    std::cout<<"Number last dof V: "<< dofV<<" Relative %: "<< double(dofV)/double(dim)*100<<std::endl;
    std::cout<<"Number last dof E: "<< dofE<<" Relative %: "<< double(dofE)/double(dim)*100<<std::endl;
    std::cout<<"Number last dof F: "<< dofF<<" Relative %: "<< double(dofF)/double(dim)*100<<std::endl;
    std::cout<<"Number last dof T: "<< dim_xn<<" Relative %: "<< double(dim_xn)/double(dim)*100<<std::endl;
*/
    if (m_with_timer) m_timer_hist.start();
    // Fill the two triplets list and number of nnz per row
    parallel_for(m_stokescore.mesh().n_cells(),batch_local_assembly,use_threads);
    if (m_with_timer) m_timer_hist.stop("Assemble triplets");
    // nnz are largely over estimated, bound them (else petsc complains)
    PetscInt offd_size = systemEffectiveDim() - loc_size;
    PetscInt offd_size_bdr = systemTotalDim() - loc_size_bdr;
    for (size_t i = 0; i < systemEffectiveDim() - last_loc_size; i++) {
      PetscInt tmp;
      tmp = nnz_mat[i].load(std::memory_order_relaxed); // run on a single thread
      tmp = (tmp > offd_size) ? offd_size : tmp;
      nnz_mat[i].store(tmp,std::memory_order_relaxed);
      tmp = nnz_mat_diag[i].load(std::memory_order_relaxed); // run on a single thread
      tmp = (tmp > loc_size) ? loc_size : tmp;
      nnz_mat_diag[i].store(tmp,std::memory_order_relaxed);
      tmp = nnz_bdr[i].load(std::memory_order_relaxed); // run on a single thread
      tmp = (tmp > offd_size_bdr) ? offd_size_bdr : tmp;
      nnz_bdr[i].store(tmp,std::memory_order_relaxed);
      tmp = nnz_bdr_diag[i].load(std::memory_order_relaxed); // run on a single thread
      tmp = (tmp > loc_size_bdr) ? loc_size_bdr : tmp;
      nnz_bdr_diag[i].store(tmp,std::memory_order_relaxed);
    }
    offd_size = systemEffectiveDim() - last_loc_size;
    offd_size_bdr = systemTotalDim() - last_loc_size_bdr;
    for (size_t i = systemEffectiveDim() - last_loc_size; i < systemEffectiveDim(); i++) {
      PetscInt tmp;
      tmp = nnz_mat[i].load(std::memory_order_relaxed); // run on a single thread
      tmp = (tmp > offd_size) ? offd_size : tmp;
      nnz_mat[i].store(tmp,std::memory_order_relaxed);
      tmp = nnz_mat_diag[i].load(std::memory_order_relaxed); // run on a single thread
      tmp = (tmp > last_loc_size) ? last_loc_size : tmp;
      nnz_mat_diag[i].store(tmp,std::memory_order_relaxed);
      tmp = nnz_bdr[i].load(std::memory_order_relaxed); // run on a single thread
      tmp = (tmp > offd_size_bdr) ? offd_size_bdr : tmp;
      nnz_bdr[i].store(tmp,std::memory_order_relaxed);
      tmp = nnz_bdr_diag[i].load(std::memory_order_relaxed); // run on a single thread
      tmp = (tmp > last_loc_size_bdr) ? last_loc_size_bdr : tmp;
      nnz_bdr_diag[i].store(tmp,std::memory_order_relaxed);
    }
      
    m_output << "\r[StokesProblem] Assembling PETsc matrix from triplets...           "<<std::flush;
    if (m_with_timer) m_timer_hist.start();
    m_solver.assemble(mat_triplets,bdr_triplets,
                      nnz_mat,nnz_mat_diag,
                      nnz_bdr,nnz_bdr_diag,
                      rhs,m_Dval.data(),
                      systemEffectiveDim(),systemTotalDim());
    if (m_with_timer) m_timer_hist.stop("Assemble matrix");
    m_output << "\r[StokesProblem] Assembled global system                              "<<std::endl;
    
    m_solver.Set_nullspace(nullspace,m_dimH);

    for (size_t i = 0; i < m_dimH; i++) {
      delete[] nullspace[i];
    }
    free(nullspace);
    delete[] nnz_mat;
    delete[] nnz_mat_diag;
    delete[] nnz_bdr;
    delete[] nnz_bdr_diag;
    delete[] rhs;
  }

 void StokesProblem::set_neumann(std::function<bool(const VectorRd &)> const &fb,const SourceFunctionType &fdun, size_t degree) {
    // Structure to store the data
    std::atomic<double>* rhs = new std::atomic<double>[systemEffectiveDim()];
    // Initialize
    for (size_t i = 0; i < systemEffectiveDim(); i++) {
      rhs[i] = 0.;
    }
    size_t dqr = (degree > 0)? degree : 2*m_stokescore.degree() + 7;
    
    // Callback function to parallel_assembly
    std::function<void(size_t start,size_t end)> batch_local_assembly = [this,fb,fdun,dqr,rhs](size_t start, size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        const Cell & T = *m_xnabla.mesh().cell(iT);
        for (size_t iF = 0; iF < T.n_faces(); iF++) {
          const Face & F = *T.face(iF);
          if (not F.is_boundary() || not inside_domain(fb,F)) continue;
          // F is on a Neumann boundary
          QuadratureRule quad_dqr_F = generate_quadrature_rule(T,dqr);
          auto basis_Pk3p2_F_quad = evaluate_quad<Function>::compute(*m_xnabla.faceBases(F.global_index()).Polyk3p2,quad_dqr_F);
          Eigen::VectorXd p_fdun = l2_projection(fdun,*m_xnabla.faceBases(F.global_index()).Polyk3p2,quad_dqr_F, basis_Pk3p2_F_quad);
          Eigen::VectorXd locm = -T.face_orientation(iF)*m_xnabla.faceOperators(F.global_index()).potential.transpose()*compute_gram_matrix(basis_Pk3p2_F_quad,quad_dqr_F)*p_fdun;

          std::vector<size_t> dofmap_xnabla = m_xnabla.globalDOFIndices(F);
          for (size_t i = 0; i < m_xnabla.dimensionFace(F.global_index());i++) {
            int gi = m_DDOFs_map(dofmap_xnabla[i]);
            if (gi < 0) continue;
            rhs[gi].fetch_add(locm(i));
          }
        }
      }
    };
    // Fill rhs
    parallel_for(m_stokescore.mesh().n_cells(),batch_local_assembly,use_threads);
    // Assemble 
    m_solver.update_rhs(rhs,systemEffectiveDim());
    // Free
    delete[] rhs;
  }

// TODO avoid code duplication with assemble()
  void StokesProblem::set_rhs (const SourceFunctionType &f, size_t degree) {
    size_t dqr = (degree > 0) ? degree : 2*m_stokescore.degree() + 3;

    std::atomic<double>* rhs = new std::atomic<double>[systemEffectiveDim()];
    // Callback function to parallel_assembly
    std::function<void(size_t start,size_t end)> batch_local_assembly = [this,f,dqr,rhs](size_t start, size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        const Cell & T = *m_xnabla.mesh().cell(iT);
        Eigen::VectorXd locR = m_xnabla.cellOperators(iT).potential.transpose()*compute_IntPf(iT,f,dqr);
        // dofs_map
        std::vector<size_t> dofmap_xnabla = m_xnabla.globalDOFIndices(T);
        for (size_t i = 0; i < m_xnabla.dimensionCell(iT);i++) { // A[1,:]
          int gi = m_DDOFs_map(dofmap_xnabla[i]); // global location of i after removal of BC dofs
          if (gi < 0) continue; // BC dofs
          rhs[gi].fetch_add(locR(i)); // atomic add
        } // i in XNabla
      } // iT
    }; // std::function batch_local_assembly
    m_solver.Set_rhs(rhs,systemEffectiveDim());
    
    delete[] rhs;
  }

  void StokesProblem::compute() {
    m_output << "[StokesProblem] Setting solver with "<<systemEffectiveDim()<<" degrees of freedom"<<std::endl;
    std::vector<int> sizes(6);
    sizes[0] = m_DDOFs_map(m_xnabla.dimension());
    sizes[1] = systemEffectiveDim();
    int dofV = m_stokescore.mesh().n_vertices()*m_xnabla.numLocalDofsVertex();
    for (size_t i = dofV; i < systemTotalDim();i++) {
      dofV = m_DDOFs_map(i);
      if (dofV > 0) break;
    }
    int dofE = m_stokescore.mesh().n_vertices()*m_xnabla.numLocalDofsVertex() 
              + m_stokescore.mesh().n_edges()*m_xnabla.numLocalDofsEdge();
    for (size_t i = dofE; i < systemTotalDim();i++) {
      dofE = m_DDOFs_map(i);
      if (dofE > 0) break;
    }
    int dofF = m_stokescore.mesh().n_vertices()*m_xnabla.numLocalDofsVertex() 
              + m_stokescore.mesh().n_edges()*m_xnabla.numLocalDofsEdge()
              + m_stokescore.mesh().n_faces()*m_xnabla.numLocalDofsFace();
    for (size_t i = dofF; i < systemTotalDim();i++) {
      dofF = m_DDOFs_map(i);
      if (dofF > 0) break;
    }
    sizes[2] = dofV;
    sizes[3] = dofE;
    sizes[4] = dofF;
    sizes[5] = sizes[0];

    m_solver.SetMonitor();
    m_solver.SetOptions(sizes);
  }

  Eigen::VectorXd StokesProblem::solve() {
    m_output << "[StokesProblem] Solving" <<std::endl;
    Eigen::VectorXd u;
    u.resize(systemEffectiveDim()); 
    if (m_with_timer) m_timer_hist.start();
    m_solver.solve(u);
    if (m_with_timer) m_timer_hist.stop("Solve");
    m_solver.Output_Converged(m_output);
    if (m_solver.info() < 0) {
      std::cerr << "[StokesProblem] Failed to solve the system: "<<m_solver.info() << std::endl;
      //m_output<< "Iterations :" << m_solver.iterations()<< " maxIterations :"<< m_solver.maxIterations()<< "tolerance :"<<m_solver.tolerance()<<std::endl;
      throw std::runtime_error("Solve failed");
    }
    #ifdef CHECK_SOLVE
    double abserr,rhs_n;
    m_solver.Residual(&abserr,&rhs_n);
    m_output<< "Absolute error :" << abserr << " Relative error :" << abserr/rhs_n << std::endl;
    #endif
    return u;
  }

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
    // Setting guess
    m_solver.Set_guess(bcguess);

    m_output << "[StokesProblem] Solving with Guess" <<std::endl;
    Eigen::VectorXd u;
    u.resize(systemEffectiveDim()); 
    if (m_with_timer) m_timer_hist.start();
    m_solver.solve(u);
    if (m_with_timer) m_timer_hist.stop("Solve");
    m_solver.Output_Converged(m_output);
    if (m_solver.info() < 0) {
      std::cerr << "[StokesProblem] Failed to solve the system: "<<m_solver.info() << std::endl;
      //m_output<< "Iterations :" << m_solver.iterations()<< " maxIterations :"<< m_solver.maxIterations()<< "tolerance :"<<m_solver.tolerance()<<std::endl;
      throw std::runtime_error("Solve failed");
    }
    #ifdef CHECK_SOLVE
    double abserr,rhs_n;
    m_solver.Residual(&abserr,&rhs_n);
    m_output<< "Absolute error :" << abserr << " Relative error :" << abserr/rhs_n << std::endl;
    #endif
    return u;
  }

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

    return std::sqrt(local_sqnorms.sum());
  }

  double Norm_L2s(const XSLStokes &xsl,const Eigen::VectorXd &v,bool use_threads=true) {
    Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xsl.mesh().n_cells());
    std::function<void(size_t,size_t)> compute_local_squarednorms = [&xsl,&v,&local_sqnorms](size_t start,size_t end)->void {
      for (size_t iT = start;iT < end; iT++) {
        local_sqnorms[iT] = xsl.restrictCell(iT,v).dot(xsl.compute_Gram_Cell(iT)*xsl.restrictCell(iT,v));
      }
    };
    parallel_for(xsl.mesh().n_cells(),compute_local_squarednorms,use_threads);

    return std::sqrt(local_sqnorms.sum());
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
