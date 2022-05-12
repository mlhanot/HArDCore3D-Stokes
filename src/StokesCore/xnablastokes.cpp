#include <xnablastokes.hpp>
#include <basis.hpp>
#include <parallel_for.hpp>
#include <GMpoly_cell.hpp>
#include <GMpoly_face.hpp>
#include <GMpoly_edge.hpp>

#include <savestates.hpp>

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XNablaStokes::XNablaStokes(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : GlobalDOFSpace(
	     stokes_core.mesh(), 
       3, 
       3*PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree()+1),
	     PolynomialSpaceDimension<Face>::Poly(stokes_core.degree())
        + PolynomialSpaceDimension<Face>::Goly(stokes_core.degree())
        + PolynomialSpaceDimension<Face>::GolyCompl(stokes_core.degree()),
	     PolynomialSpaceDimension<Cell>::Goly(stokes_core.degree() - 1) + PolynomialSpaceDimension<Cell>::GolyCompl(stokes_core.degree())
	      ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output),
    m_cell_operators(stokes_core.mesh().n_cells()),
    m_face_operators(stokes_core.mesh().n_faces()),
    m_edge_operators(stokes_core.mesh().n_edges())
{
  m_output << "[XNablaStokes] Initializing" << std::endl;
  if (use_threads) {
    m_output << "[XNablaStokes] Parallel execution" << std::endl;
  } else {
    m_output << "[XNablaStokes] Sequential execution" << std::endl;
  }

  // Construct edge nablas and potentials
  std::function<void(size_t, size_t)> construct_all_edge_nablas_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iE = start; iE < end; iE++) {
          m_edge_operators[iE].reset( new LocalOperators(_compute_edge_nabla_potential(iE)) );
        } // for iE
      };

  m_output << "[XNablaStokes] Constructing edge nablas and potentials" << std::endl;
  parallel_for(mesh().n_edges(), construct_all_edge_nablas_potentials, use_threads);
  // Construct face nablas and potentials
  std::function<void(size_t, size_t)> construct_all_face_nablas_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          m_face_operators[iF].reset( new LocalOperators(_compute_face_nabla_potential(iF)) );
        } // for iF
      };

  m_output << "[XNablaStokes] Constructing face nablas and potentials" << std::endl;
  parallel_for(mesh().n_faces(), construct_all_face_nablas_potentials, use_threads);

  // Construct cell nablas and potentials
  std::function<void(size_t, size_t)> construct_all_cell_nablas_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          m_cell_operators[iT].reset( new LocalOperators(_compute_cell_nabla_potential(iT)) );
        } // for iT
      };

  m_output << "[XNablaStokes] Constructing cell nablas and potentials" << std::endl;
  parallel_for(mesh().n_cells(), construct_all_cell_nablas_potentials, use_threads);
}

XNablaStokes::XNablaStokes(const StokesCore & stokes_core, const std::string &from_file, bool use_threads, std::ostream & output)
  : GlobalDOFSpace(
	     stokes_core.mesh(), 
       3, 
       3*PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree()+1),
	     PolynomialSpaceDimension<Face>::Poly(stokes_core.degree())
        + PolynomialSpaceDimension<Face>::Goly(stokes_core.degree())
        + PolynomialSpaceDimension<Face>::GolyCompl(stokes_core.degree()),
	     PolynomialSpaceDimension<Cell>::Goly(stokes_core.degree() - 1) + PolynomialSpaceDimension<Cell>::GolyCompl(stokes_core.degree())
	      ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output),
    m_cell_operators(stokes_core.mesh().n_cells()),
    m_face_operators(stokes_core.mesh().n_faces()),
    m_edge_operators(stokes_core.mesh().n_edges())
{
  m_output << "[XNablaStokes] Initializing from datafile" << std::endl;

  MatReader reader(from_file);
  for (size_t iE = 0; iE < mesh().n_edges(); iE++) {
    m_edge_operators[iE].reset( new LocalOperators(reader.read_MatXd(),reader.read_MatXd(),reader.read_MatXd()));
  }
  for (size_t iF = 0; iF < mesh().n_faces(); iF++){
    m_face_operators[iF].reset( new LocalOperators(reader.read_MatXd(),reader.read_MatXd(),reader.read_MatXd()));
  }
  for (size_t iT = 0; iT < mesh().n_cells(); iT++){
    m_cell_operators[iT].reset( new LocalOperators(reader.read_MatXd(),reader.read_MatXd(),reader.read_MatXd()));
  }
}

void XNablaStokes::Write_internal(const std::string &to_file) const {
  MatWriter writer(to_file);
  for (size_t iE = 0; iE < mesh().n_edges(); iE++) {
    writer.append_mat(m_edge_operators[iE]->divergence);
    writer.append_mat(m_edge_operators[iE]->nabla);
    writer.append_mat(m_edge_operators[iE]->potential);
  }
  for (size_t iF = 0; iF < mesh().n_faces(); iF++){
    writer.append_mat(m_face_operators[iF]->divergence);
    writer.append_mat(m_face_operators[iF]->nabla);
    writer.append_mat(m_face_operators[iF]->potential);
  }
  for (size_t iT = 0; iT < mesh().n_cells(); iT++){
    writer.append_mat(m_cell_operators[iT]->divergence);
    writer.append_mat(m_cell_operators[iT]->nabla);
    writer.append_mat(m_cell_operators[iT]->potential);
  }
}

//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XNablaStokes::interpolate(const FunctionType & w,  const int doe_cell, const int doe_face, const int doe_edge) const
{
  Eigen::VectorXd wh = Eigen::VectorXd::Zero(dimension());

  // Degrees of quadrature rules
  size_t dqr_cell = (doe_cell >= 0 ? doe_cell : 2 * degree() + 3);
  size_t dqr_face = (doe_face >= 0 ? doe_face : 2 * degree() + 4);
  size_t dqr_edge = (doe_edge >= 0 ? doe_edge : 2 * degree() + 5);

  // Interpolate at vertices
  std::function<void(size_t, size_t)> interpolate_vertices
    = [this, &wh, w](size_t start, size_t end)->void
      {
        for (size_t iV = start; iV < end; iV++) {
          wh.segment(3*iV,3) = w(mesh().vertex(iV)->coords());
        } // for iV
      };
  parallel_for(mesh().n_vertices(), interpolate_vertices, m_use_threads); 

  // Interpolate at edges
  std::function<void(size_t, size_t)> interpolate_edges
    = [this, &wh, w, &dqr_edge](size_t start, size_t end)->void
      {
	      for (size_t iE = start; iE < end; iE++) {
	        const Edge & E = *mesh().edge(iE);

	        QuadratureRule quad_dqr_E = generate_quadrature_rule(E, dqr_edge);
	        auto basis_Pk3po_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk3po, quad_dqr_E);
	        wh.segment(globalOffset(E), edgeBases(iE).Polyk3po->dimension())
	          = l2_projection(w, *edgeBases(iE).Polyk3po, quad_dqr_E, basis_Pk3po_E_quad);
          /*
	        auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykpo, quad_dqr_E);
	        size_t dim_Pk_E = PolynomialSpaceDimension<Edge>::Poly(degree()+1);
          for (size_t i = 0; i < 3; i++) {
          wh.segment(globalOffset(E) + i*dim_Pk_E, dim_Pk_E)
	          = l2_projection(
              [w,i](const Eigen::Vector3d & x)->double {
                return w(x)(i);
              },
              *edgeBases(iE).Polykpo, quad_dqr_E, basis_Pkpo_E_quad);
          } // for i*/
	      } // for iE
      };
  parallel_for(mesh().n_edges(), interpolate_edges, m_use_threads); 

  // Interpolate at faces
  std::function<void(size_t, size_t)> interpolate_faces
    = [this, &wh, w, &dqr_face](size_t start, size_t end)->void
      {
	for (size_t iF = start; iF < end; iF++) {
	  const Face & F = *mesh().face(iF);

	  Eigen::Vector3d nF = F.normal();
	  auto w_dot_nF = [&nF, w](const Eigen::Vector3d & x)->double {
			    return w(x).dot(nF);
			  };
	  
	  QuadratureRule quad_dqr_F = generate_quadrature_rule(F, dqr_face);

    size_t offset_F = globalOffset(F);
	  auto basis_Pk_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Polyk, quad_dqr_F);
	  wh.segment(offset_F, PolynomialSpaceDimension<Face>::Poly(degree()))
	    = l2_projection(w_dot_nF, *faceBases(iF).Polyk, quad_dqr_F, basis_Pk_F_quad);

    auto w_t = [&nF, w](const Eigen::Vector3d & x)->Eigen::Vector3d {
          return nF.cross(w(x).cross(nF));
        };

    offset_F += PolynomialSpaceDimension<Face>::Poly(degree());
    auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Golyk, quad_dqr_F);
    wh.segment(offset_F, PolynomialSpaceDimension<Face>::Goly(degree()))
      = l2_projection(w_t,*faceBases(iF).Golyk, quad_dqr_F, basis_Gk_F_quad);

    offset_F += PolynomialSpaceDimension<Face>::Goly(degree());
    if (degree() > 0) {
      auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).GolyComplk, quad_dqr_F);
      wh.segment(offset_F, PolynomialSpaceDimension<Face>::GolyCompl(degree()))
        = l2_projection(w_t,*faceBases(iF).GolyComplk, quad_dqr_F, basis_Gck_F_quad);
    }
	} // for iF
      };
  parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);
  
  // Interpolate at cells
  if (degree() > 0) {  
    std::function<void(size_t, size_t)> interpolate_cells
      = [this, &wh, w, &dqr_cell](size_t start, size_t end)->void
	{
	  for (size_t iT = start; iT < end; iT++) {		 
	    const Cell & T = *mesh().cell(iT);

	    size_t offset_T = globalOffset(T);

	    QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr_cell);
	    MonomialCellIntegralsType int_mono_2k = IntegrateCellMonomials(T, 2*degree());
	    auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Golykmo, quad_dqr_T);
	    wh.segment(offset_T, PolynomialSpaceDimension<Cell>::Goly(degree() - 1))
	      = l2_projection(w, *cellBases(iT).Golykmo, quad_dqr_T, basis_Gkmo_T_quad, GramMatrix(T, *cellBases(iT).Golykmo, int_mono_2k));

	    offset_T += PolynomialSpaceDimension<Cell>::Goly(degree() - 1);	    
	    auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).GolyComplk, quad_dqr_T);
	    wh.segment(offset_T, PolynomialSpaceDimension<Cell>::GolyCompl(degree()))
	      = l2_projection(w, *cellBases(iT).GolyComplk, quad_dqr_T, basis_Gck_T_quad, GramMatrix(T, *cellBases(iT).GolyComplk, int_mono_2k));
	  } // for iT
	};
    parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
  } // if degree() > 0
	
  return wh;
}

//------------------------------------------------------------------------------
// Nabla and potential reconstruction
//------------------------------------------------------------------------------

// Edge
//------------------------------------------------------------------------------
XNablaStokes::LocalOperators XNablaStokes::_compute_edge_nabla_potential(size_t iE)
{
  const Edge & E = *mesh().edge(iE);

  //------------------------------------------------------------------------------
  // Nabla
  //------------------------------------------------------------------------------

  // w_E'
  //------------------------------------------------------------------------------
  // Left-hand side matrix

  MonomialEdgeIntegralsType int_mono_2kp5_E = IntegrateEdgeMonomials(E, 2*degree() + 5);
  Eigen::MatrixXd Gram_Pkp2_E = GramMatrix(E, *edgeBases(iE).Polykp2, int_mono_2kp5_E);
  size_t dim_Pkp2_E = PolynomialSpaceDimension<Edge>::Poly(degree()+2);
  size_t dim_Pkpo_E = PolynomialSpaceDimension<Edge>::Poly(degree()+1);
  Eigen::MatrixXd MNE_wE = Eigen::MatrixXd::Zero(edgeBases(iE).Polyk3p2->dimension(),edgeBases(iE).Polyk3p2->dimension());
  for (size_t i = 0; i < 3; i++) {
    MNE_wE.block(i*dim_Pkp2_E,i*dim_Pkp2_E,dim_Pkp2_E,dim_Pkp2_E) = Gram_Pkp2_E;
  }

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BNE_wE = 
    Eigen::MatrixXd::Zero(edgeBases(iE).Polyk3p2->dimension(), dimensionEdge(iE));
  for (size_t i = 0; i < edgeBases(iE).Polyk3p2->dimension(); i++) {
    BNE_wE(i, 0) = -edgeBases(iE).Polyk3p2->function(i,mesh().edge(iE)->vertex(0)->coords())(0); // x component
    BNE_wE(i, 1) = -edgeBases(iE).Polyk3p2->function(i,mesh().edge(iE)->vertex(0)->coords())(1); // y component
    BNE_wE(i, 2) = -edgeBases(iE).Polyk3p2->function(i,mesh().edge(iE)->vertex(0)->coords())(2); // z component
    BNE_wE(i, 3) = edgeBases(iE).Polyk3p2->function(i,mesh().edge(iE)->vertex(1)->coords())(0); // x component
    BNE_wE(i, 4) = edgeBases(iE).Polyk3p2->function(i,mesh().edge(iE)->vertex(1)->coords())(1); // y component
    BNE_wE(i, 5) = edgeBases(iE).Polyk3p2->function(i,mesh().edge(iE)->vertex(1)->coords())(2); // z component
  } // for i
  GradientBasis<StokesCore::PolyBasisEdgeType> grad_Pkp2_E(*edgeBases(iE).Polykp2);
  //TensorizedVectorFamily<GradientBasis<StokesCore::PolyBasisEdgeType>,3> grad_Pk3p3_E(GradientBasis<StokesCore::PolyBasisEdgeType>(*edgeBases(iE).Polykp3));
  Eigen::MatrixXd mGram_GPkp2_Pkpo_E = -GramMatrix(E,grad_Pkp2_E, *edgeBases(iE).Polykpo, int_mono_2kp5_E);
  for (size_t i = 0; i < 3; i++) {
    BNE_wE.block(i*dim_Pkp2_E,6+i*dim_Pkpo_E,dim_Pkp2_E,dim_Pkpo_E) = mGram_GPkp2_Pkpo_E;
  }

  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------
  
  // Pk+3 [x y z]
  //------------------------------------------------------------------------------
  // Right-hand side matrix
  size_t dim_Pk3p3_E = 3*PolynomialSpaceDimension<Edge>::Poly(degree()+3);
  size_t dim_Pk3p2_E = 3*PolynomialSpaceDimension<Edge>::Poly(degree()+2);
  Eigen::MatrixXd BPE
    = Eigen::MatrixXd::Zero(dim_Pk3p3_E,dimensionEdge(iE));

  // Enforce the gradient of the potential reconstruction
  BPE.topRows(dim_Pk3p2_E) = BNE_wE;

  // Enforce the average value of the potential reconstruction
  QuadratureRule quad_kpo_E = generate_quadrature_rule(E, degree()+1);
  auto basis_Pk3po_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk3po, quad_kpo_E);
  
  // We set the average value of the potential equal to the average of the edge unknown
  for (size_t i = 0; i < 3*PolynomialSpaceDimension<Edge>::Poly(degree()+1); i++) {
    for (size_t iqn = 0; iqn < quad_kpo_E.size(); iqn++) {
      BPE.bottomRows(3).block(0,6 + i,3,1) += quad_kpo_E[iqn].w * basis_Pk3po_E_quad[i][iqn];
    } // for iqn
  } // for i
  
  //------------------------------------------------------------------------------
  // Left-hand side matrix

  Eigen::MatrixXd MPE
    = Eigen::MatrixXd::Zero(dim_Pk3p3_E,dim_Pk3p3_E);

  GradientBasis<StokesCore::PolyBasisEdgeType> grad_Pkp3_E(*edgeBases(iE).Polykp3);
  Eigen::MatrixXd Gram_Pkp2_GPkp3 = GramMatrix(E,*edgeBases(iE).Polykp2, grad_Pkp3_E, int_mono_2kp5_E);
  size_t dim_Pkp3_E = PolynomialSpaceDimension<Edge>::Poly(degree()+3);
  for (size_t i = 0; i < 3; i++) {
    MPE.block(i*dim_Pkp2_E,i*dim_Pkp3_E,dim_Pkp2_E,dim_Pkp3_E) = Gram_Pkp2_GPkp3;
  }

  // Average
  //TensorizedVectorFamily<MonomialScalarBasisEdge,3> basis_3P0_E(MonomialScalarBasisEdge(E,0));
  MonomialScalarBasisEdge basis_P0_E(E,0);
  Eigen::MatrixXd Gram_P0_Pkp3_E = GramMatrix(E, basis_P0_E, *edgeBases(iE).Polykp3, int_mono_2kp5_E);
  for (size_t i = 0; i < 3; i++) {
    MPE.bottomRows(3).block(i,i*dim_Pkp3_E,1,dim_Pkp3_E) = Gram_P0_Pkp3_E;
  }

  return LocalOperators(Eigen::MatrixXd(),MNE_wE.ldlt().solve(BNE_wE),MPE.partialPivLu().solve(BPE));
}


// Face
//------------------------------------------------------------------------------
XNablaStokes::LocalOperators XNablaStokes::_compute_face_nabla_potential(size_t iF)
{
  const Face & F = *mesh().face(iF);
  //------------------------------------------------------------------------------
  // Nabla
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*(degree() + 2));
  auto basis_tildePkpo_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).tildePolykpo,quad_2kp2_F);
  Eigen::MatrixXd MGF = compute_gram_matrix(basis_tildePkpo_F_quad,quad_2kp2_F);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BGF 
    = Eigen::MatrixXd::Zero(faceBases(iF).tildePolykpo->dimension(), dimensionFace(iF));

  // Boundary contribution
  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    
    QuadratureRule quad_2kp3_E = generate_quadrature_rule(E, 2 * (degree() + 3));
    auto basis_tildePkpo_nFE_E_quad
      = matrix_vector_product(evaluate_quad<Function>::compute(*faceBases(iF).tildePolykpo, quad_2kp3_E), F.edge_normal(iE));
    auto basis_Pk3p3_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk3p3, quad_2kp3_E);
    Eigen::MatrixXd BGF_E
      = F.edge_orientation(iE) * compute_gram_matrix(basis_tildePkpo_nFE_E_quad, basis_Pk3p3_E_quad, quad_2kp3_E) * m_edge_operators[E.global_index()]->potential; // the potential reconstruct the continuous polynomial and insert the missing dofs 

    // Assemble local contribution
    BGF.middleCols(localOffset(F, *E.vertex(0)),3) += BGF_E.middleCols(0,3); // vx(x0),vy(x0),vz(x0)
    BGF.middleCols(localOffset(F, *E.vertex(1)),3) += BGF_E.middleCols(3,3); // vx(x1),vy(x1),vz(x1)
    BGF.middleCols(localOffset(F, E),dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1)))
        = BGF_E.rightCols(dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1)));
  } // for iE

  // Face contribution
  auto divergence_Rbk_F_quad = evaluate_quad<Divergence>::compute(*faceBases(iF).Rolybkpo, quad_2kp2_F);
  auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Golyk, quad_2kp2_F);

  auto divergence_bPk_F_quad = scalar_product(evaluate_quad<Divergence>::compute(*faceBases(iF).bPolykpo, quad_2kp2_F),F.normal());
  auto basis_Pk_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Polyk, quad_2kp2_F);

  Eigen::MatrixXd BGFpart = Eigen::MatrixXd::Zero(faceBases(iF).tildePolykpo->dimension(),faceBases(iF).Polyk->dimension() + faceBases(iF).Golyk->dimension() + PolynomialSpaceDimension<Face>::GolyCompl(degree()));
  // tildePkpo = bPk^ + ((Rbc^k + Rb^k) + R2^k
  // dofsF : P + G + Gc
  size_t dim_Pk = PolynomialSpaceDimension<Face>::Poly(degree());
  size_t dim_Gk = PolynomialSpaceDimension<Face>::Goly(degree());
  size_t dim_Gck = PolynomialSpaceDimension<Face>::GolyCompl(degree());
  size_t dim_bPkpo = 2*PolynomialSpaceDimension<Face>::Poly(degree()+1);
  size_t dim_Rbckpo = PolynomialSpaceDimension<Face>::RolybCompl(degree()+1);
  size_t dim_Rbkpo = PolynomialSpaceDimension<Face>::Rolyb(degree()+1);
  BGFpart.block(0,0,dim_bPkpo,dim_Pk) 
      = -compute_gram_matrix(divergence_bPk_F_quad, basis_Pk_F_quad, quad_2kp2_F);

  if (degree() > 0) {
    auto divergence_Rbck_F_quad = evaluate_quad<Divergence>::compute(*faceBases(iF).RolybComplkpo, quad_2kp2_F);
    auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).GolyComplk, quad_2kp2_F);
    BGFpart.block(dim_bPkpo,dim_Pk+dim_Gk,dim_Rbckpo,dim_Gck) 
        = -compute_gram_matrix(divergence_Rbck_F_quad, basis_Gck_F_quad, quad_2kp2_F);
  } // degree > 0
  
  BGFpart.block(dim_bPkpo+dim_Rbckpo,dim_Pk,dim_Rbkpo,dim_Gk) 
      = -compute_gram_matrix(divergence_Rbk_F_quad, basis_Gk_F_quad, quad_2kp2_F);

  BGF.rightCols(dim_Pk + dim_Gk + dim_Gck) = BGFpart;

  Eigen::MatrixXd GF = MGF.ldlt().solve(BGF);

  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  QuadratureRule quad_2kp3_F = generate_quadrature_rule(F,2*(degree()+3));
  auto basis_Pk3p2_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Polyk3p2, quad_2kp3_F);
  auto divergence_Rck3p3_F_quad = evaluate_quad<Divergence>::compute(*faceBases(iF).RolyComplk3p3, quad_2kp3_F);
  Eigen::MatrixXd MPF = compute_gram_matrix(divergence_Rck3p3_F_quad, basis_Pk3p2_F_quad, quad_2kp3_F);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  // Face contribution
  Eigen::MatrixXd BPF
    = -compute_gram_matrix(
      evaluate_quad<Function>::compute(*faceBases(iF).RolyComplk3p3, quad_2kp3_F),
      evaluate_quad<Function>::compute(*faceBases(iF).tildePolykpo, quad_2kp3_F),
      quad_2kp3_F
    ) * GF;

  // Boundary contribution
  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    
    QuadratureRule quad_2kp3_E = generate_quadrature_rule(E, 2 * (degree() + 3));
    auto basis_Rck3p3_nFE_E_quad
      = matrix_vector_product(evaluate_quad<Function>::compute(*faceBases(iF).RolyComplk3p3, quad_2kp3_E), F.edge_normal(iE));
    auto basis_Pk3p3_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk3p3, quad_2kp3_E);
    Eigen::MatrixXd BPF_E
      = F.edge_orientation(iE)*compute_gram_matrix(basis_Rck3p3_nFE_E_quad, basis_Pk3p3_E_quad, quad_2kp3_E) * m_edge_operators[E.global_index()]->potential; // the potential reconstruct the continuous polynomial and insert the missing dofs 

    // Assemble local contribution
    BPF.middleCols(localOffset(F, *E.vertex(0)),3) += BPF_E.middleCols(0,3); // vx(x0),vy(x0),vz(x0)
    BPF.middleCols(localOffset(F, *E.vertex(1)),3) += BPF_E.middleCols(3,3); // vx(x1),vy(x1),vz(x1)
    BPF.middleCols(localOffset(F, E), dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1))) 
        += BPF_E.rightCols(dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1)));
  } // for iE

  return LocalOperators(Eigen::MatrixXd(),GF,MPF.partialPivLu().solve(BPF));
}

// Cell
//------------------------------------------------------------------------------
XNablaStokes::LocalOperators XNablaStokes::_compute_cell_nabla_potential(size_t iT)
{
  const Cell & T = *mesh().cell(iT);
  //------------------------------------------------------------------------------
  // Nabla
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  QuadratureRule quad_2kp2_T = generate_quadrature_rule(T,2*degree() + 2);
  auto basis_RTbkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo,quad_2kp2_T);
  Eigen::MatrixXd MGT = compute_gram_matrix(basis_RTbkpo_T_quad,quad_2kp2_T);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BGT 
    = Eigen::MatrixXd::Zero(cellBases(iT).RTbkpo->dimension(), dimensionCell(iT));

  // Boundary contribution
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    
    QuadratureRule quad_2kp3_F = generate_quadrature_rule(F, 2 * (degree() + 3));
    auto basis_RTbkpo_nF_F_quad
      = matrix_vector_product(evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kp3_F), F.normal());
    auto basis_Pk3p2_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Polyk3p2, quad_2kp3_F);
    Eigen::MatrixXd PF = extendOperator(T,F, m_face_operators[F.global_index()]->potential);
    BGT += T.face_orientation(iF) * compute_gram_matrix(basis_RTbkpo_nF_F_quad, basis_Pk3p2_F_quad, quad_2kp3_F) * PF; 
  } // for iF

  // Cell contribution
  if (degree() > 0) {
    auto divergence_Rbck_T_quad = evaluate_quad<Divergence>::compute(*cellBases(iT).RolybComplkpo, quad_2kp2_T);
    auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).GolyComplk, quad_2kp2_T);

    auto divergence_Rbkmo_T_quad = evaluate_quad<Divergence>::compute(*cellBases(iT).Rolybk, quad_2kp2_T);
    auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Golykmo, quad_2kp2_T);

    Eigen::MatrixXd BGTpart = Eigen::MatrixXd::Zero(cellBases(iT).RTbkpo->dimension(),cellBases(iT).Golykmo->dimension() + cellBases(iT).GolyComplk->dimension());
    // RTbkpo = (Rbc^k + Rb^k) + R3^k
    // dofsT : G + Gc
    size_t dim_Gkmo = PolynomialSpaceDimension<Cell>::Goly(degree()-1);
    size_t dim_Gck = PolynomialSpaceDimension<Cell>::GolyCompl(degree());
    size_t dim_Rbckpo = PolynomialSpaceDimension<Cell>::RolybCompl(degree()+1);
    size_t dim_Rbk = PolynomialSpaceDimension<Cell>::Rolyb(degree());
    BGTpart.block(0,dim_Gkmo,dim_Rbckpo,dim_Gck) 
        = -compute_gram_matrix(divergence_Rbck_T_quad, basis_Gck_T_quad, quad_2kp2_T);
    BGTpart.block(dim_Rbckpo,0,dim_Rbk,dim_Gkmo) 
        = -compute_gram_matrix(divergence_Rbkmo_T_quad, basis_Gkmo_T_quad, quad_2kp2_T);

    BGT.rightCols(dim_Gkmo + dim_Gck) = BGTpart;
  } // degree > 0

  Eigen::MatrixXd GT = MGT.ldlt().solve(BGT);

  //------------------------------------------------------------------------------
  // Divergence
  //------------------------------------------------------------------------------
  // Trace of gradient

  auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk,quad_2kp2_T);
  Eigen::MatrixXd MDT = compute_gram_matrix(basis_Pk_T_quad,quad_2kp2_T);
  auto basis_RTbk_Tr_T_quad = eval_trace_quad(basis_RTbkpo_T_quad);
  Eigen::MatrixXd BDT 
    = compute_gram_matrix(basis_Pk_T_quad,basis_RTbk_Tr_T_quad,quad_2kp2_T) * GT;

  Eigen::MatrixXd DT = MDT.ldlt().solve(BDT);
  

  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  QuadratureRule quad_2kp3_T = generate_quadrature_rule(T,2*(degree()+3));
  auto basis_Pk3po_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk3po, quad_2kp3_T);
  auto divergence_Rck3p2_T_quad = evaluate_quad<Divergence>::compute(*cellBases(iT).RolyComplk3p2, quad_2kp3_T);
  Eigen::MatrixXd MPT = compute_gram_matrix(divergence_Rck3p2_T_quad, basis_Pk3po_T_quad, quad_2kp3_T);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  // Cell contribution
  Eigen::MatrixXd BPT
    = -compute_gram_matrix(
      evaluate_quad<Function>::compute(*cellBases(iT).RolyComplk3p2, quad_2kp3_T),
      evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kp3_T),
      quad_2kp3_T
    ) * GT;

  // Boundary contribution
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    
    QuadratureRule quad_2kp3_F = generate_quadrature_rule(F, 2 * (degree() + 3));
    auto basis_Rck3p2_nF_F_quad
      = matrix_vector_product(evaluate_quad<Function>::compute(*cellBases(iT).RolyComplk3p2, quad_2kp3_F), F.normal());
    auto basis_Pk3p2_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Polyk3p2, quad_2kp3_F);
    Eigen::MatrixXd PF = extendOperator(T,F, m_face_operators[F.global_index()]->potential);
    BPT += T.face_orientation(iF)*compute_gram_matrix(basis_Rck3p2_nF_F_quad, basis_Pk3p2_F_quad, quad_2kp3_F) * PF;

  } // for iF

  return LocalOperators(DT,GT,MPT.partialPivLu().solve(BPT));
}

//------------------------------------------------------------------------------
//        Functions to compute matrices for local L2 products on XNabla
//------------------------------------------------------------------------------

Eigen::MatrixXd XNablaStokes::computeL2Product(
                                        const size_t iT,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & mass_Pk3po_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 

  // create the weighted mass matrix, with simple product if weight is constant
  Eigen::MatrixXd w_mass_Pk3po_T;
  if (weight.deg(T)==0){
    // constant weight
    if (mass_Pk3po_T.rows()==1){
      // We have to compute the mass matrix
      MonomialCellIntegralsType int_mono_2kpo = IntegrateCellMonomials(T, 2*(degree()+1));
      w_mass_Pk3po_T = weight.value(T, T.center_mass()) * GramMatrix(T, *cellBases(iT).Polyk3po, int_mono_2kpo);
    }else{
      w_mass_Pk3po_T = weight.value(T, T.center_mass()) * mass_Pk3po_T;
    }
  }else{
    // weight is not constant, we create a weighted mass matrix
    QuadratureRule quad_2kpw_T = generate_quadrature_rule(T, 2 * (degree()+1) + weight.deg(T));
    auto basis_Pk3po_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk3po, quad_2kpw_T);
    std::function<double(const Eigen::Vector3d &)> weight_T 
              = [&T, &weight](const Eigen::Vector3d &x)->double {
                  return weight.value(T, x);
                };
    w_mass_Pk3po_T = compute_weighted_gram_matrix(weight_T, basis_Pk3po_T_quad, basis_Pk3po_T_quad, quad_2kpw_T, "sym");
  }

  // leftOp and rightOp come from the potentials
  std::vector<Eigen::MatrixXd> potentialOp(T.n_edges() + T.n_faces() + 1);
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    potentialOp[iE] = extendOperator(T, E, m_edge_operators[E.global_index()]->potential);
  }
  for (size_t iF = 0; iF < T.n_faces(); iF++){
    const Face & F = *T.face(iF);
    potentialOp[T.n_edges() + iF] = extendOperator(T, F, m_face_operators[F.global_index()]->potential);
  }
  potentialOp[T.n_edges() + T.n_faces()] = m_cell_operators[iT]->potential;

  return computeL2Product_with_Ops(iT, potentialOp, potentialOp, penalty_factor, w_mass_Pk3po_T, weight);

}


Eigen::MatrixXd XNablaStokes::computeL2Product_with_Ops(
                                        const size_t iT,
                                        const std::vector<Eigen::MatrixXd> & leftOp,
                                        const std::vector<Eigen::MatrixXd> & rightOp,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & w_mass_Pk3po_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 

  // leftOp and rightOp must list the operators acting on the DOFs, and which we want to
  // use for the L2 product. Specifically, each one lists operators (matrices) returning
  // values in edges space P^k+3(E)^3, faces space P^k+2(F)^3 and element space P^k+1(T)^3.
  // For the standard Xnabla L2 product, these will respectively be potential for each edge, each face, and PT. 
  // All these operators must have the same domain, so possibly being extended appropriately
  // using extendOperator from globaldofspace.

  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(leftOp[0].cols(), rightOp[0].cols());
  
  size_t offset_F = T.n_edges();
  size_t offset_T = T.n_edges() + T.n_faces();
  
  // Edge penalty terms
  // Counted twices outside the faces loop
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    // Compute gram matrices
    QuadratureRule quad_2kp3_E = generate_quadrature_rule(E, 2 * (degree()+3));
    
    auto basis_Pk3p3_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk3p3, quad_2kp3_E);
    auto basis_Pk3po_TE_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk3po, quad_2kp3_E);
    Eigen::MatrixXd gram_EE = compute_gram_matrix(basis_Pk3p3_E_quad,quad_2kp3_E);
    Eigen::MatrixXd gram_TT = compute_gram_matrix(basis_Pk3po_TE_quad,quad_2kp3_E);
    Eigen::MatrixXd gram_TE = compute_gram_matrix(basis_Pk3po_TE_quad,basis_Pk3p3_E_quad,quad_2kp3_E,"nonsym");

    // Weight including scaling hE (we compute the max over quadrature nodes to get an estimate of the max of the weight over the edge)
    double max_weight_quad_E = weight.value(T, quad_2kp3_E[0].vector());
    // If the weight is not constant, we want to take the largest along the edge
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2kp3_E.size(); iqn++) {
        max_weight_quad_E = std::max(max_weight_quad_E, weight.value(T, quad_2kp3_E[iqn].vector()));
      } // for
    }
    double w_hE2 = 2.*max_weight_quad_E * E.diam()*E.diam();

    // The penalty term int_E (leftOp - (leftOp)_E) * (rightOp - (rightOp)_E) is computed by developping
    // Contribution of face F
    L2P += w_hE2 * ( leftOp[offset_T].transpose() * gram_TT * rightOp[offset_T]
                - leftOp[offset_T].transpose() * gram_TE * rightOp[iE] 
                - leftOp[iE].transpose() * gram_TE.transpose() * rightOp[offset_T]
                + leftOp[iE].transpose() * gram_EE * rightOp[iE]                  
                  );
  } // for iE

  // Face penalty terms
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);

    // Compute gram matrices
    QuadratureRule quad_2kp2_F = generate_quadrature_rule(F, 2 * (degree()+2));
    
    auto basis_Pk3p2_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Polyk3p2, quad_2kp2_F);
    auto basis_Pk3po_TF_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk3po, quad_2kp2_F);
    Eigen::MatrixXd gram_FF = compute_gram_matrix(basis_Pk3p2_F_quad,quad_2kp2_F);
    Eigen::MatrixXd gram_TT = compute_gram_matrix(basis_Pk3po_TF_quad,quad_2kp2_F);
    Eigen::MatrixXd gram_TF = compute_gram_matrix(basis_Pk3po_TF_quad,basis_Pk3p2_F_quad,quad_2kp2_F,"nonsym");

    // Weight including scaling hF (we compute the max over quadrature nodes to get an estimate of the max of the weight over the face)
    double max_weight_quad_F = weight.value(T, quad_2kp2_F[0].vector());
    // If the weight is not constant, we want to take the largest along the edge
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2kp2_F.size(); iqn++) {
        max_weight_quad_F = std::max(max_weight_quad_F, weight.value(T, quad_2kp2_F[iqn].vector()));
      } // for
    }
    double w_hF = max_weight_quad_F * F.diam();

    // The penalty term int_F (leftOp - (leftOp)_F) * (rightOp - (rightOp)_F) is computed by developping
    // Contribution of face F
    L2P += w_hF * ( leftOp[offset_T].transpose() * gram_TT * rightOp[offset_T]
                - leftOp[offset_T].transpose() * gram_TF * rightOp[offset_F + iF] 
                - leftOp[offset_F + iF].transpose() * gram_TF.transpose() * rightOp[offset_T]
                + leftOp[offset_F + iF].transpose() * gram_FF * rightOp[offset_F + iF]                  
                  );

  } // for iF

  L2P *= penalty_factor;
  
  // Consistent (cell) term
  L2P += leftOp[offset_T].transpose() * w_mass_Pk3po_T * rightOp[offset_T];
 
  return L2P;
}


Eigen::MatrixXd XNablaStokes::computeL2opnEdge(size_t iE) const {
  const Edge & E = *mesh().edge(iE);
  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(dimensionEdge(iE),dimensionEdge(iE));
  // Vertices
  L2P.block(0,0,6,6) = E.diam()*Eigen::MatrixXd::Identity(6,6);
  // L2 product of each components
  QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2 * degree()+2);
  auto basis_Pk3po_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk3po, quad_2kp2_E);
  size_t dimPk3po = 3*PolynomialSpaceDimension<Edge>::Poly(degree() +1 );
  L2P.block(6,6,dimPk3po,dimPk3po) = compute_gram_matrix(basis_Pk3po_E_quad,quad_2kp2_E);

  return L2P;
}

Eigen::MatrixXd XNablaStokes::computeL2opnFace(size_t iF) const {
  const Face & F = *mesh().face(iF);
  QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * degree());
  auto basis_Pk_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).Polyk, quad_2k_F);
  auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).Golyk, quad_2k_F);
  auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).GolyComplk, quad_2k_F);
  Eigen::MatrixXd gram_Pk_F = compute_gram_matrix(basis_Pk_F_quad, quad_2k_F);
  Eigen::MatrixXd gram_Gk_F = compute_gram_matrix(basis_Gk_F_quad, quad_2k_F);
  Eigen::MatrixXd gram_Gck_F = compute_gram_matrix(basis_Gck_F_quad, quad_2k_F);
  size_t dimPk = PolynomialSpaceDimension<Face>::Poly(degree());
  size_t dimGk = PolynomialSpaceDimension<Face>::Goly(degree());
  size_t dimGck = PolynomialSpaceDimension<Face>::GolyCompl(degree());

  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(dimensionFace(iF),dimensionEdge(iF));
  size_t offset = localOffset(F);
  L2P.block(offset,offset,dimPk,dimPk) = gram_Pk_F;
  offset += dimPk;
  L2P.block(offset,offset,dimGk,dimGk) = gram_Gk_F;
  offset += dimGk;
  L2P.block(offset,offset,dimGck,dimGck) = gram_Gck_F;
  
  return L2P;
}

Eigen::MatrixXd XNablaStokes::computeL2opnCell(size_t iT) const {
  const Cell & T = *mesh().cell(iT);
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * degree());
  auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(T.global_index()).Golykmo, quad_2k_T);
  auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*cellBases(T.global_index()).GolyComplk, quad_2k_T);
  Eigen::MatrixXd gram_Gkmo_T = compute_gram_matrix(basis_Gkmo_T_quad, quad_2k_T);
  Eigen::MatrixXd gram_Gck_T = compute_gram_matrix(basis_Gck_T_quad, quad_2k_T);
  size_t dimGkmo = PolynomialSpaceDimension<Cell>::Goly(degree()-1);
  size_t dimGck = PolynomialSpaceDimension<Cell>::GolyCompl(degree());

  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(dimensionCell(iT),dimensionCell(iT));
  size_t offset = localOffset(T);
  L2P.block(offset,offset,dimGkmo,dimGkmo) = gram_Gkmo_T;
  offset += dimGkmo;
  L2P.block(offset,offset,dimGck,dimGck) = gram_Gck_T;

  return L2P;
}

VectorRd XNablaStokes::evaluatePotential(size_t iT, const Eigen::VectorXd & uh, const VectorRd & x) const
{
  VectorRd rv = VectorRd::Zero();
  Eigen::VectorXd Puh = cellOperators(iT).potential * uh;
  for (size_t i = 0; i < 3*PolynomialSpaceDimension<Cell>::Poly(degree()+1);i++) {
    rv += cellBases(iT).Polyk3po->function(i,x)*Puh(i);
  }
  return rv;
}

