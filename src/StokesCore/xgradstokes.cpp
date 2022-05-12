#include "xgradstokes.hpp"

#include <basis.hpp>
#include <parallel_for.hpp>
#include <GMpoly_cell.hpp>
#include <GMpoly_face.hpp>
#include <GMpoly_edge.hpp>

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XGradStokes::XGradStokes(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : GlobalDOFSpace(stokes_core.mesh(),
	     4,
	     PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree() - 1) 
          + 2 * PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree()),
	     2 * PolynomialSpaceDimension<Face>::Poly(stokes_core.degree() - 1),
	     PolynomialSpaceDimension<Cell>::Poly(stokes_core.degree() - 1)
	     ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output),
    m_edge_operators(stokes_core.mesh().n_edges()),
    m_face_operators(stokes_core.mesh().n_faces()),
    m_cell_operators(stokes_core.mesh().n_cells())
{
  m_output << "[XGradStokes] Initializing" << std::endl;
  if (use_threads) {
    m_output << "[XGradStokes] Parallel execution" << std::endl;
  } else {
    m_output << "[XGradStokes] Sequential execution" << std::endl;
  }
  
  // Construct edge gradients and potentials
  std::function<void(size_t, size_t)> construct_all_edge_gradients_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iE = start; iE < end; iE++) {
          m_edge_operators[iE].reset( new LocalOperators(_compute_edge_gradient_potential(iE)) );
        } // for iE
      };

  m_output << "[XGradStokes] Constructing edge gradients and potentials" << std::endl;
  parallel_for(mesh().n_edges(), construct_all_edge_gradients_potentials, use_threads);

  // Construct face gradients and potentials
  std::function<void(size_t, size_t)> construct_all_face_gradients_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          m_face_operators[iF].reset( new LocalOperators(_compute_face_gradient_potential(iF)) );
        } // for iF
      };

  m_output << "[XGradStokes] Constructing face gradients and potentials" << std::endl;
  parallel_for(mesh().n_faces(), construct_all_face_gradients_potentials, use_threads);

  // Construct cell gradients and potentials
  std::function<void(size_t, size_t)> construct_all_cell_gradients_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          m_cell_operators[iT].reset( new LocalOperators(_compute_cell_gradient_potential(iT)) );
        } // for iT
      };

  m_output << "[XGradStokes] Constructing cell gradients and potentials" << std::endl;
  parallel_for(mesh().n_cells(), construct_all_cell_gradients_potentials, use_threads);
}

//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XGradStokes::interpolate(const FunctionType & q, const FunctionGradType & Gq, const int doe_cell, const int doe_face, const int doe_edge) const
{
  Eigen::VectorXd qh = Eigen::VectorXd::Zero(dimension());

  // Degrees of quadrature rules
  size_t dqr_cell = (doe_cell >= 0 ? doe_cell : 2 * degree() + 3);
  size_t dqr_face = (doe_face >= 0 ? doe_face : 2 * degree() + 3);
  size_t dqr_edge = (doe_edge >= 0 ? doe_edge : 2 * degree() + 3);
  
  // Interpolate at vertices
  std::function<void(size_t, size_t)> interpolate_vertices
    = [this, &qh, q, Gq](size_t start, size_t end)->void
      {
        for (size_t iV = start; iV < end; iV++) {
          qh.segment(4*iV,3) = Gq(mesh().vertex(iV)->coords());
          qh(4*iV + 3) = q(mesh().vertex(iV)->coords());
        } // for iV
      };
  parallel_for(mesh().n_vertices(), interpolate_vertices, m_use_threads);

  if (degree() == 0) {
    // interpolate at edges
    std::function<void(size_t, size_t)> interpolate_edges
      = [this, &qh, q, Gq, &dqr_edge](size_t start, size_t end)->void
        {
          for (size_t iE = start; iE < end; iE++) {
            const Edge & E = *mesh().edge(iE);

            //Eigen::Vector3d tE = E.tangent();
            std::vector<Eigen::Vector3d> basisE = E.edge_normalbasis();
            auto nE1_tE_c_Gq_c_tE = [&basisE, Gq](const Eigen::Vector3d & x)->double {
                //return basisE[0].dot(tE.cross(Gq(x).cross(tE)));
                return basisE[0].dot(Gq(x));
            };
            auto nE2_tE_c_Gq_c_tE = [&basisE, Gq](const Eigen::Vector3d & x)->double {
                //return basisE[1].dot(tE.cross(Gq(x).cross(tE)));
                return basisE[1].dot(Gq(x));
            };
            QuadratureRule quad_dqr_E = generate_quadrature_rule(E, dqr_edge);
            auto basis_pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_dqr_E);
            qh.segment(globalOffset(E), PolynomialSpaceDimension<Edge>::Poly(degree())) 
              = l2_projection(nE1_tE_c_Gq_c_tE, *edgeBases(iE).Polyk, quad_dqr_E, basis_pk_E_quad);
            qh.segment(globalOffset(E) + PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree())) 
              = l2_projection(nE2_tE_c_Gq_c_tE, *edgeBases(iE).Polyk, quad_dqr_E, basis_pk_E_quad);
          } // for iE
        };
    parallel_for(mesh().n_edges(), interpolate_edges, m_use_threads);

  } else { // degree() > 0 
    // interpolate at edges
    std::function<void(size_t, size_t)> interpolate_edges
      = [this, &qh, q, Gq, &dqr_edge](size_t start, size_t end)->void
        {
          for (size_t iE = start; iE < end; iE++) {
            const Edge & E = *mesh().edge(iE);

            Eigen::Vector3d tE = E.tangent();
            std::vector<Eigen::Vector3d> basisE = E.edge_normalbasis();
            auto nE1_tE_c_Gq_c_tE = [&tE, &basisE, Gq](const Eigen::Vector3d & x)->double {
                return basisE[0].dot(tE.cross(Gq(x).cross(tE)));
            };
            auto nE2_tE_c_Gq_c_tE = [&tE, &basisE, Gq](const Eigen::Vector3d & x)->double {
                return basisE[1].dot(tE.cross(Gq(x).cross(tE)));
            };
            QuadratureRule quad_dqr_E = generate_quadrature_rule(E, dqr_edge);
            auto basis_pkmo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykmo, quad_dqr_E);
            qh.segment(globalOffset(E),PolynomialSpaceDimension<Edge>::Poly(degree()-1)) 
              = l2_projection(q, *edgeBases(iE).Polykmo, quad_dqr_E, basis_pkmo_E_quad);
            auto basis_pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_dqr_E);
            qh.segment(globalOffset(E) + PolynomialSpaceDimension<Edge>::Poly(degree()-1), PolynomialSpaceDimension<Edge>::Poly(degree())) 
              = l2_projection(nE1_tE_c_Gq_c_tE, *edgeBases(iE).Polyk, quad_dqr_E, basis_pk_E_quad);
            qh.segment(globalOffset(E) + PolynomialSpaceDimension<Edge>::Poly(degree()-1) + PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree())) 
              = l2_projection(nE2_tE_c_Gq_c_tE, *edgeBases(iE).Polyk, quad_dqr_E, basis_pk_E_quad);
          } // for iE
        };
    parallel_for(mesh().n_edges(), interpolate_edges, m_use_threads);
    
    // Interpolate at faces
    std::function<void(size_t, size_t)> interpolate_faces
      = [this, &qh, q, Gq, &dqr_face](size_t start, size_t end)->void
        {
          for (size_t iF = start; iF < end; iF++) {
            const Face & F = *mesh().face(iF);
            QuadratureRule quad_dqr_F = generate_quadrature_rule(F, dqr_face);
            auto basis_Pkmo_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Polykmo, quad_dqr_F);
            qh.segment(globalOffset(F), PolynomialSpaceDimension<Face>::Poly(degree() - 1)) 
              = l2_projection(q, *faceBases(iF).Polykmo, quad_dqr_F, basis_Pkmo_F_quad);

            Eigen::Vector3d nF = mesh().face(iF)->normal();
            auto Gq_nF = [&nF, Gq](const Eigen::Vector3d & x)->double {
                return Gq(x).dot(nF);
            };
            qh.segment(globalOffset(F) + PolynomialSpaceDimension<Face>::Poly(degree() - 1),PolynomialSpaceDimension<Face>::Poly(degree() - 1)) 
              = l2_projection(Gq_nF, *faceBases(iF).Polykmo, quad_dqr_F, basis_Pkmo_F_quad);

          } // for iF
        };
    parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);

    // Interpolate at cells
    std::function<void(size_t, size_t)> interpolate_cells
      = [this, &qh, q, &dqr_cell](size_t start, size_t end)->void
        {
          for (size_t iT = start; iT < end; iT++) {
            const Cell & T = *mesh().cell(iT);
            QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr_cell);
            auto basis_Pkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykmo, quad_dqr_T);
            MonomialCellIntegralsType int_mono_2km2 = IntegrateCellMonomials(T, 2*degree()-2);
            qh.segment(globalOffset(T), PolynomialSpaceDimension<Cell>::Poly(degree() - 1)) 
              = l2_projection(q, *cellBases(iT).Polykmo, quad_dqr_T, basis_Pkmo_T_quad, GramMatrix(T, *cellBases(iT).Polykmo, int_mono_2km2));
          } // for iT
        };
    parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
  } // if degree() > 0 

  return qh;
}

//------------------------------------------------------------------------------
// Gradient and potential reconstructions
//------------------------------------------------------------------------------

XGradStokes::LocalOperators XGradStokes::_compute_edge_gradient_potential(size_t iE)
{
  const Edge & E = *mesh().edge(iE);
  
  std::vector<Eigen::Vector3d> basisE = E.edge_normalbasis();
  /*
  Eigen::Matrix3d Pen2xyz;
  Pen2xyz.col(0) = basisE[0];
  Pen2xyz.col(1) = basisE[1];
  Pen2xyz.col(2) = basisE[2];
  Eigen::Matrix3d Pxyz2en = Pen2xyz.inverse();
  */
  //------------------------------------------------------------------------------
  // Gradient
  //------------------------------------------------------------------------------

  // q_E'
  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  MonomialEdgeIntegralsType int_mono_2kp1_E = IntegrateEdgeMonomials(E, 2*degree()+1);
  Eigen::MatrixXd MGE_qE = GramMatrix(E, *edgeBases(iE).Polyk, int_mono_2kp1_E);

  //------------------------------------------------------------------------------
  // Right-hand side matrix
  
  Eigen::MatrixXd BGE_qE
    = Eigen::MatrixXd::Zero(edgeBases(iE).Polyk->dimension(), dimensionEdge(iE));
  for (size_t i = 0; i < edgeBases(iE).Polyk->dimension(); i++) {
    BGE_qE(i, 3) = -edgeBases(iE).Polyk->function(i, mesh().edge(iE)->vertex(0)->coords()); // location of q(V)
    BGE_qE(i, 7) = edgeBases(iE).Polyk->function(i, mesh().edge(iE)->vertex(1)->coords());
  } // for i
  
  if (degree() > 0) {
    GradientBasis<StokesCore::PolyBasisEdgeType> grad_Pk_E(*edgeBases(iE).Polyk);
    BGE_qE.middleCols(8,PolynomialSpaceDimension<Edge>::Poly(degree() - 1)) 
          = -GramMatrix(E, grad_Pk_E, *edgeBases(iE).Polykmo, int_mono_2kp1_E);
  }
  // q_E' = MGE_qE^-1 BGE_qE

  // v_E'
  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  MonomialEdgeIntegralsType int_mono_2kp3_E = IntegrateEdgeMonomials(E, 2*degree()+3);
  Eigen::MatrixXd MGE_vE = GramMatrix(E, *edgeBases(iE).Polykpo, int_mono_2kp3_E);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BGE_vEn1p
    = Eigen::MatrixXd::Zero(edgeBases(iE).Polykpo->dimension(), dimensionEdge(iE));
  Eigen::MatrixXd BGE_vEn2p
    = Eigen::MatrixXd::Zero(edgeBases(iE).Polykpo->dimension(), dimensionEdge(iE));
  for (size_t i = 0; i < edgeBases(iE).Polykpo->dimension(); i++) {
    BGE_vEn1p.block(i,0,1,3) = -edgeBases(iE).Polykpo->function(i, mesh().edge(iE)->vertex(0)->coords())*basisE[0].transpose(); // dot product with nE1
    BGE_vEn2p.block(i,0,1,3) = -edgeBases(iE).Polykpo->function(i, mesh().edge(iE)->vertex(0)->coords())*basisE[1].transpose(); // dot product with nE2
    BGE_vEn1p.block(i,4,1,3) = edgeBases(iE).Polykpo->function(i, mesh().edge(iE)->vertex(1)->coords())*basisE[0].transpose(); // dot product with nE1
    BGE_vEn2p.block(i,4,1,3) = edgeBases(iE).Polykpo->function(i, mesh().edge(iE)->vertex(1)->coords())*basisE[1].transpose(); // dot product with nE2
  } // for i
  
  GradientBasis<StokesCore::PolyBasisEdgeType> grad_Pkpo_E(*edgeBases(iE).Polykpo);
  BGE_vEn1p.middleCols(8+PolynomialSpaceDimension<Edge>::Poly(degree()-1),PolynomialSpaceDimension<Edge>::Poly(degree())) 
        = -GramMatrix(E, grad_Pkpo_E, *edgeBases(iE).Polyk, int_mono_2kp3_E);
  BGE_vEn2p.middleCols(8+PolynomialSpaceDimension<Edge>::Poly(degree()-1) + PolynomialSpaceDimension<Edge>::Poly(degree()), 
  PolynomialSpaceDimension<Edge>::Poly(degree())) 
        = -GramMatrix(E, grad_Pkpo_E, *edgeBases(iE).Polyk, int_mono_2kp3_E);
  // v_En1' = MGE_vE^-1 BGE_vEn1p
  //------------------------------------------------------------------------------
  // Zip dofs
  Eigen::MatrixXd uGE = Eigen::MatrixXd::Zero(12+3*PolynomialSpaceDimension<Edge>::Poly(degree()) + 3*PolynomialSpaceDimension<Edge>::Poly(degree()+1),dimensionEdge(iE));
  // R_{v,V} = 0, skip first 3 lines
  // v_E(x_V) = G_{q,V} 
  uGE(3,0)=1.;uGE(4,1)=1.;uGE(5,2)=1.;
  // Redo for second vertice
  uGE(9,4)=1.;uGE(10,5)=1.;uGE(11,6)=1.;
  // pi{k}(vE.tE) = qE'
  uGE.middleRows(12,PolynomialSpaceDimension<Edge>::Poly(degree())) = MGE_qE.ldlt().solve(BGE_qE);
  // pi{k}(vE.nE1) = G_{q,E}.nE1
  uGE.block(12+PolynomialSpaceDimension<Edge>::Poly(degree()),8+PolynomialSpaceDimension<Edge>::Poly(degree()-1),
    PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree())) 
    = Eigen::MatrixXd::Identity(PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree()));
  // pi{k}(vE.nE2) = G_{q,E}.nE2
  uGE.block(12+2*PolynomialSpaceDimension<Edge>::Poly(degree()),8+PolynomialSpaceDimension<Edge>::Poly(degree()-1) + PolynomialSpaceDimension<Edge>::Poly(degree()),
    PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree())) 
    = Eigen::MatrixXd::Identity(PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree()));

  // pi{k+1}(R_{v,E}) = 0 tE + v_En2' nE1 - v_En1' nE2
  // compute v_Eni' as temporary
  Eigen::MatrixXd dofs2v_En1p = MGE_vE.ldlt().solve(BGE_vEn1p);
  Eigen::MatrixXd dofs2v_En2p = MGE_vE.ldlt().solve(BGE_vEn2p);
  for (size_t i = 0; i < 3; i++) { // itterate over x y z
    uGE.middleRows(12+3*PolynomialSpaceDimension<Edge>::Poly(degree())+i*PolynomialSpaceDimension<Edge>::Poly(degree()+1),PolynomialSpaceDimension<Edge>::Poly(degree())+1) 
      = basisE[0](i)*dofs2v_En2p - basisE[1](i)*dofs2v_En1p;
  }

  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BPE
    = Eigen::MatrixXd::Zero(PolynomialSpaceDimension<Edge>::Poly(degree()) + 1, dimensionEdge(iE));

  // Enforce the gradient of the potential reconstruction
  BPE.topRows(PolynomialSpaceDimension<Edge>::Poly(degree())) = BGE_qE;

  // Enforce the average value of the potential reconstruction
  if (degree() == 0) {
    // We set the average equal to the mean of vertex values
    BPE.bottomRows(1)(0, 3) = 0.5 * E.measure();
    BPE.bottomRows(1)(0, 7) = 0.5 * E.measure();
  } else {
    QuadratureRule quad_kmo_E = generate_quadrature_rule(E, degree() - 1);
    auto basis_Pkmo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykmo, quad_kmo_E);
    
    // We set the average value of the potential equal to the average of the edge unknown
    for (size_t i = 0; i < PolynomialSpaceDimension<Edge>::Poly(degree() - 1); i++) {
      for (size_t iqn = 0; iqn < quad_kmo_E.size(); iqn++) {
        BPE.bottomRows(1)(0, 8 + i) += quad_kmo_E[iqn].w * basis_Pkmo_E_quad[i][iqn];
      } // for iqn
    } // for i
  }
  
  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  Eigen::MatrixXd MPE
    = Eigen::MatrixXd::Zero(PolynomialSpaceDimension<Edge>::Poly(degree()) + 1, PolynomialSpaceDimension<Edge>::Poly(degree() + 1));

  // GradientBasis<StokesCore::PolyBasisEdgeType> grad_Pkpo_E(*edgeBases(iE).Polykpo);
  MPE.topRows(PolynomialSpaceDimension<Edge>::Poly(degree()))
        = GramMatrix(E, *edgeBases(iE).Polyk, grad_Pkpo_E, int_mono_2kp1_E);

  MonomialScalarBasisEdge basis_P0_E(E, 0);
  MPE.bottomRows(1) = GramMatrix(E, basis_P0_E, *edgeBases(iE).Polykpo, int_mono_2kp1_E);

  return LocalOperators(uGE, Eigen::MatrixXd(), MPE.partialPivLu().solve(BPE));
}

//------------------------------------------------------------------------------

XGradStokes::LocalOperators XGradStokes::_compute_face_gradient_potential(size_t iF)
{
  const Face & F = *mesh().face(iF);

  //------------------------------------------------------------------------------
  // Gradient
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix
  MonomialFaceIntegralsType int_mono_2kp3_F = IntegrateFaceMonomials(F, 2*degree()+3);
  Eigen::MatrixXd MGF = GramMatrix(F, *faceBases(iF).Polyk2, int_mono_2kp3_F);

  //------------------------------------------------------------------------------
  // Right-hand side matrix
  
  Eigen::MatrixXd BGF
    = Eigen::MatrixXd::Zero(faceBases(iF).Polyk2->dimension(), dimensionFace(iF));

  // Boundary contribution
  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    
    QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * (degree() + 1));
    auto basis_Pk2_nFE_E_quad
      = scalar_product(evaluate_quad<Function>::compute(*faceBases(iF).Polyk2, quad_2kpo_E), F.edge_normal(iE));
    auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polykpo, quad_2kpo_E);
    Eigen::MatrixXd PE = extendOperator(F, E, edgeOperators(E).potential);
    BGF += F.edge_orientation(iE) * compute_gram_matrix(basis_Pk2_nFE_E_quad, basis_Pkpo_E_quad, quad_2kpo_E) * PE;
  } // for iE

  // Face contribution
  if (degree() > 0) {
    DivergenceBasis<StokesCore::Poly2BasisFaceType> div_Pk2_F(*faceBases(iF).Polyk2);
    // Correspond to q_F
    BGF.middleCols(dimensionFace(iF) - 2*PolynomialSpaceDimension<Face>::Poly(degree() - 1),PolynomialSpaceDimension<Face>::Poly(degree() - 1))
      -= GramMatrix(F, div_Pk2_F, *faceBases(iF).Polykmo, int_mono_2kp3_F);
  } // if degree() > 0

  Eigen::MatrixXd GF = MGF.ldlt().solve(BGF);
  
  // Gradient_perp
  //------------------------------------------------------------------------------
  // Right-hand side matrix
  
  Eigen::MatrixXd BGFperp
    = Eigen::MatrixXd::Zero(faceBases(iF).Polyk2->dimension(), dimensionFace(iF));

  // Boundary contribution
  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    
    QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * (degree() + 1));
    // dot tE instead of nFE
    auto basis_Pk2_tE_E_quad
      = scalar_product(evaluate_quad<Function>::compute(*faceBases(iF).Polyk2, quad_2kpo_E), E.tangent());
    // matrix of int_E (w_F . tE ) P^{k}
    auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk, quad_2kpo_E);
    Eigen::MatrixXd gramG_q = compute_gram_matrix(basis_Pk2_tE_E_quad, basis_Pk_E_quad, quad_2kpo_E);
    // get the normal space of E
    std::vector<Eigen::Vector3d> basisE = E.edge_normalbasis();
    // Map the dofs and introduce G.nF as an operator from E to P^k
    Eigen::MatrixXd PE_loc = Eigen::MatrixXd::Zero(edgeBases(E).Polyk->dimension(),dimensionEdge(E.global_index()));
    PE_loc.middleCols(8+PolynomialSpaceDimension<Edge>::Poly(degree()-1),PolynomialSpaceDimension<Edge>::Poly(degree())) 
      = basisE[0].dot(F.normal()) * Eigen::MatrixXd::Identity(PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree()));
    PE_loc.middleCols(8+PolynomialSpaceDimension<Edge>::Poly(degree()-1)+PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree())) 
      = basisE[1].dot(F.normal()) * Eigen::MatrixXd::Identity(PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree()));


    Eigen::MatrixXd PE = extendOperator(F, E, PE_loc);
    BGFperp -= F.edge_orientation(iE) * compute_gram_matrix(basis_Pk2_tE_E_quad, basis_Pk_E_quad, quad_2kpo_E) * PE;
  } // for iE

  // Face contribution
  if (degree() > 0) {
    RotBasis<StokesCore::Poly2BasisFaceType> rot_Pk2_F(*faceBases(iF).Polyk2);
    QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2*(degree() + 1));
    auto basis_rot_F_quad = evaluate_quad<Function>::compute(rot_Pk2_F, quad_2kpo_F);
    auto basis_Pkmo_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Polykmo, quad_2kpo_F);
    // Correspond to G_{q,F}
    BGFperp.rightCols(PolynomialSpaceDimension<Face>::Poly(degree() - 1))
      // -= GramMatrix(F, rot_Pk2_F, *faceBases(iF).Polykmo, int_mono_2kp3_F); // Need to implement mono for these basis
      -= compute_gram_matrix(basis_rot_F_quad, basis_Pkmo_F_quad, quad_2kpo_F);
  } // if degree() > 0

  Eigen::MatrixXd GFperp = MGF.ldlt().solve(BGFperp);
  
  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------
  // Tangential trace, left unchanged (the shift in dofs is already taken care of by the new definitions of GF and edgeOperators(E).potential
  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  DivergenceBasis<StokesCore::RolyComplBasisFaceType> div_Rckp2_F(*faceBases(iF).RolyComplkp2);
  Eigen::MatrixXd MPF = GramMatrix(F, div_Rckp2_F, *faceBases(iF).Polykpo, int_mono_2kp3_F);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  // Face contribution: we need integrals up to 2k+3 here because Polyk2 is a restriction of a basis of degree k+1
  Eigen::MatrixXd BPF = -GramMatrix(F, *faceBases(iF).RolyComplkp2, *faceBases(iF).Polyk2, int_mono_2kp3_F) * GF;
  
  // Boundary contribution
  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    
    QuadratureRule quad_2kp4_E = generate_quadrature_rule(E, 2 * (degree() + 2));
    auto basis_Rckp2_nFE_E_quad
      = scalar_product(evaluate_quad<Function>::compute(*faceBases(iF).RolyComplkp2, quad_2kp4_E), F.edge_normal(iE));
    auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polykpo, quad_2kp4_E);
    Eigen::MatrixXd PE = extendOperator(F, E, edgeOperators(E).potential);
    BPF += F.edge_orientation(iE) * compute_gram_matrix(basis_Rckp2_nFE_E_quad, basis_Pkpo_E_quad, quad_2kp4_E) * PE;
  } // for iE
  
  return LocalOperators(GF, GFperp, MPF.partialPivLu().solve(BPF));
}

//------------------------------------------------------------------------------
// Left unchanged, the potentials on faces and edges have already been remapped and the definition is the same whithout additionnal dofs.
XGradStokes::LocalOperators XGradStokes::_compute_cell_gradient_potential(size_t iT)
{
  const Cell & T = *mesh().cell(iT);

  //------------------------------------------------------------------------------
  // Gradient
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  // Compute all integrals of monomial powers to degree 2k+2 and the mass matrix
  MonomialCellIntegralsType int_mono_2kp3 = IntegrateCellMonomials(T, 2*degree()+3);
  Eigen::MatrixXd MGT = GramMatrix(T, *cellBases(iT).Polyk3, int_mono_2kp3);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BGT
    = Eigen::MatrixXd::Zero(cellBases(iT).Polyk3->dimension(), dimensionCell(iT));

  // Boundary contribution
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    
    DecomposePoly dec(F, MonomialScalarBasisFace(F, degree()));
    auto Pk3T_dot_nTF_nodes = scalar_product(evaluate_quad<Function>::compute(*cellBases(iT).Polyk3, dec.get_nodes()), T.face_normal(iF));
    auto Pk3T_dot_nTF_family_PkF = dec.family(Pk3T_dot_nTF_nodes);
    Eigen::MatrixXd PF = extendOperator(T, F, faceOperators(F).potential);
    MonomialFaceIntegralsType int_mono_2kp1_F = IntegrateFaceMonomials(F, 2*degree()+1);
    BGT += GramMatrix(F, Pk3T_dot_nTF_family_PkF, *faceBases(F).Polykpo, int_mono_2kp1_F) * PF;

    // Following commented block could replace the block above, without using DecomposePoly (more expensive, but sometimes better rounding errors)
    /*
      QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2 * degree() + 1);
      auto basis_Pk3_nTF_F_quad = scalar_product(
					         evaluate_quad<Function>::compute(*cellBases(iT).Polyk3, quad_2kpo_F),
					         T.face_normal(iF)
					         );
      auto basis_Pkpo_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Polykpo, quad_2kpo_F);
      Eigen::MatrixXd PF = extendOperator(T, F, faceOperators(F).potential);
      BGT += compute_gram_matrix(basis_Pk3_nTF_F_quad, basis_Pkpo_F_quad, quad_2kpo_F) * PF;
    */
    
  } // for iF

  // Cell contribution
  if (degree() > 0) {
    DivergenceBasis<StokesCore::Poly3BasisCellType> div_Pk3_basis(*cellBases(iT).Polyk3);
    BGT.rightCols(PolynomialSpaceDimension<Cell>::Poly(degree() - 1)) -= GramMatrix(T, div_Pk3_basis, *cellBases(iT).Polykmo, int_mono_2kp3);
  } // if degree() > 0

  Eigen::MatrixXd GT = MGT.ldlt().solve(BGT);
  
  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  DivergenceBasis<StokesCore::RolyComplBasisCellType> div_Rckp2_basis(*cellBases(iT).RolyComplkp2);
  Eigen::MatrixXd MPT = GramMatrix(T, div_Rckp2_basis, *cellBases(iT).Polykpo, int_mono_2kp3);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  // Cell contribution: we need integrals up to degree 2k+3 because Polyk3 comes from the restriction of a basis of degree k+1
  Eigen::MatrixXd BPT
   = -GramMatrix(T, *cellBases(iT).RolyComplkp2, *cellBases(iT).Polyk3, int_mono_2kp3) * GT;

  // Boundary contribution
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
   const Face & F = *T.face(iF);

   MonomialScalarBasisFace basis_Pkp2_F(F, degree()+2);
   DecomposePoly dec(F, basis_Pkp2_F);
   auto Rckp2T_dot_nF_nodes = scalar_product(evaluate_quad<Function>::compute(*cellBases(iT).RolyComplkp2, dec.get_nodes()), F.normal());
   Family<MonomialScalarBasisFace> Rckp2T_dot_nF_family_Pkp2F = dec.family(Rckp2T_dot_nF_nodes);
   auto PF = extendOperator(T, F, faceOperators(F).potential);
   MonomialFaceIntegralsType int_mono_2kp3_F = IntegrateFaceMonomials(F, 2*degree()+3);
   BPT += T.face_orientation(iF) * GramMatrix(F, Rckp2T_dot_nF_family_Pkp2F, *faceBases(F).Polykpo, int_mono_2kp3_F) * PF;

   // Following commented block does the same as above, but with DecomposePoly and seems to lead to increased errors
   /*
     QuadratureRule quad_2kp2_F = generate_quadrature_rule(F, 2 * (degree() + 2));
     auto basis_Rckp2_nF_F_quad
       = scalar_product(evaluate_quad<Function>::compute(*cellBases(iT).RolyComplkp2, quad_2kp2_F), F.normal());
     auto basis_Pkpo_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Polykpo, quad_2kp2_F);
     auto PF = extendOperator(T, F, faceOperators(F).potential);
     BPT += T.face_orientation(iF) * compute_gram_matrix(basis_Rckp2_nF_F_quad, basis_Pkpo_F_quad, quad_2kp2_F) * PF;
   */
   
  } // for iF                                              

  return LocalOperators(GT, Eigen::MatrixXd(), MPT.partialPivLu().solve(BPT));
}

//-----------------------------------------------------------------------------
// local L2 inner product
//-----------------------------------------------------------------------------
Eigen::MatrixXd XGradStokes::computeL2Product(
                                        const size_t iT,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & mass_Pkpo_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 
  
  // create the weighted mass matrix, with simple product if weight is constant
  Eigen::MatrixXd w_mass_Pkpo_T;
  if (weight.deg(T)==0){
    // constant weight
    if (mass_Pkpo_T.rows()==1){
      // We have to compute the mass matrix
      MonomialCellIntegralsType int_mono_2kp2 = IntegrateCellMonomials(T, 2*degree()+2);
      w_mass_Pkpo_T = weight.value(T, T.center_mass()) * GramMatrix(T, *cellBases(iT).Polykpo, int_mono_2kp2);
    }else{
      w_mass_Pkpo_T = weight.value(T, T.center_mass()) * mass_Pkpo_T;
    }
  }else{
    // weight is not constant, we create a weighted mass matrix
    QuadratureRule quad_2kpo_pw_T = generate_quadrature_rule(T, 2 * (degree() + 1) + weight.deg(T));
    auto basis_Pkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykpo, quad_2kpo_pw_T);
    std::function<double(const Eigen::Vector3d &)> weight_T 
              = [&T, &weight](const Eigen::Vector3d &x)->double {
                  return weight.value(T, x);
                };
    w_mass_Pkpo_T = compute_weighted_gram_matrix(weight_T, basis_Pkpo_T_quad, basis_Pkpo_T_quad, quad_2kpo_pw_T, "sym");
  }


  // Compute matrix of L2 product  
  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(dimensionCell(iT), dimensionCell(iT));

  // Penalty for potential terms G_{q,V}
  for (size_t iV = 0; iV < T.n_vertices(); iV++) {
    const Vertex & V = *T.vertex(iV);

    // weight and scaling hV^4
    double w_hT4 = weight.value(T,V.coords()) * std::pow(T.diam(), 4);

    L2P.block(4*iV,4*iV,3,3) = w_hT4*Eigen::MatrixXd::Identity(3,3);
  } // iV

  // We need the potential in the cell
  Eigen::MatrixXd Potential_T = cellOperators(iT).potential;

  // Edge penalty terms
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
        
    QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * (degree()+1) );
    
    // weight and scaling hE^2
    double max_weight_quad_E = weight.value(T, quad_2kpo_E[0].vector());
    // If the weight is not constant, we want to take the largest along the edge
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2kpo_E.size(); iqn++) {
        max_weight_quad_E = std::max(max_weight_quad_E, weight.value(T, quad_2kpo_E[iqn].vector()));
      } // for
    }
    double w_hT2 = max_weight_quad_E * std::pow(T.diam(), 2);

    // The penalty term int_E (PT q - q_E) * (PT r - r_E) is computed by developping.
    auto basis_Pkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykpo, quad_2kpo_E);
    auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polykpo, quad_2kpo_E);
    Eigen::MatrixXd gram_PkpoT_PkpoE = compute_gram_matrix(basis_Pkpo_T_quad, basis_Pkpo_E_quad, quad_2kpo_E);
    
    Eigen::MatrixXd Potential_E = extendOperator(T, E, edgeOperators(E).potential);

    // Contribution of edge E
    Eigen::MatrixXd PTtrans_mass_PE = Potential_T.transpose() * gram_PkpoT_PkpoE * Potential_E;
    L2P += w_hT2 * ( Potential_T.transpose() * compute_gram_matrix(basis_Pkpo_T_quad, quad_2kpo_E) * Potential_T
                   - PTtrans_mass_PE - PTtrans_mass_PE.transpose()
                   + Potential_E.transpose() * compute_gram_matrix(basis_Pkpo_E_quad, quad_2kpo_E) * Potential_E );
  
    // Penalty for the potential terms G_{q,E}
    double w_hT3 = max_weight_quad_E * std::pow(T.diam(), 3);
    auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk, quad_2kpo_E);

    size_t offset_E = localOffset(T,E) + PolynomialSpaceDimension<Edge>::Poly(degree()-1);
    size_t dimPk = PolynomialSpaceDimension<Edge>::Poly(degree());
    Eigen::MatrixXd wgram_PkE = w_hT3 * compute_gram_matrix(basis_Pk_E_quad,quad_2kpo_E);
    L2P.block(offset_E,offset_E,dimPk,dimPk) += wgram_PkE;
    offset_E += PolynomialSpaceDimension<Edge>::Poly(degree());
    L2P.block(offset_E,offset_E,dimPk,dimPk) += wgram_PkE;
  } // for iE

  // Face penalty terms
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2 * (degree()+1) );

    // weight and scaling hF (we use quadrature nodes to evaluate the maximum of the weight)
    double max_weight_quad_F = weight.value(T, quad_2kpo_F[0].vector());
    // If the weight is not constant, we want to take the largest along the face
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2kpo_F.size(); iqn++) {
        max_weight_quad_F = std::max(max_weight_quad_F, weight.value(T, quad_2kpo_F[iqn].vector()));
      } // for
    }
    double w_hT = max_weight_quad_F * T.diam();

    // The penalty term int_F (PT q - gammaF q) * (PT r - gammaF r) is computed by developping.
    MonomialFaceIntegralsType int_monoF_2kp2 = IntegrateFaceMonomials(F, 2*degree()+2);
    DecomposePoly dec(F, MonomialScalarBasisFace(F, degree()+1));
    auto PkpoT_nodes = evaluate_quad<Function>::compute(*cellBases(iT).Polykpo, dec.get_nodes());
    auto PkpoT_family_PkpoF = dec.family(PkpoT_nodes);
    Eigen::MatrixXd gram_PkpoT_PkpoF = GramMatrix(F, PkpoT_family_PkpoF, *faceBases(F).Polykpo, int_monoF_2kp2);
  
    // Contribution of face F
    Eigen::MatrixXd Potential_F = extendOperator(T, F, faceOperators(F).potential);
    Eigen::MatrixXd PTtrans_mass_PF = Potential_T.transpose() * gram_PkpoT_PkpoF * Potential_F;
    L2P += w_hT * ( Potential_T.transpose() * GramMatrix(F, PkpoT_family_PkpoF, int_monoF_2kp2) * Potential_T
                 - PTtrans_mass_PF - PTtrans_mass_PF.transpose()
                 + Potential_F.transpose() * GramMatrix(F, *faceBases(F).Polykpo, int_monoF_2kp2) * Potential_F );

    // Penalty for the potential terms G_{q,F}
    if (degree() > 0) {
      double w_hT2 = max_weight_quad_F * std::pow(T.diam(), 2);
      size_t offset_F = localOffset(T,F) + PolynomialSpaceDimension<Face>::Poly(degree()-1);
      size_t dimPkmo = PolynomialSpaceDimension<Face>::Poly(degree()-1);
      L2P.block(offset_F,offset_F,dimPkmo,dimPkmo) 
        += w_hT2 * GramMatrix(F, *faceBases(F).Polykmo, int_monoF_2kp2);
    } // degree > 0

    // Following commented block does the same as above, but without DecomposePoly (which sometimes increases errors)
    /*
      auto basis_Pkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykpo, quad_2kpo_F);
      auto basis_Pkpo_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).Polykpo, quad_2kpo_F);
      Eigen::MatrixXd gram_PkpoT_PkpoF = compute_gram_matrix(basis_Pkpo_T_quad, basis_Pkpo_F_quad, quad_2kpo_F);
      
      Eigen::MatrixXd Potential_F = extendOperator(T, F, faceOperators(F).potential);
      Eigen::MatrixXd PTtrans_mass_PF = Potential_T.transpose() * gram_PkpoT_PkpoF * Potential_F;

      // Contribution of face F
      L2P += w_hF * ( Potential_T.transpose() * compute_gram_matrix(basis_Pkpo_T_quad, quad_2kpo_F) * Potential_T
                     - PTtrans_mass_PF - PTtrans_mass_PF.transpose()
                     + Potential_F.transpose() * GramMatrix(F, *faceBases(F).Polykpo) * Potential_F );
    */       
    
    
  } // for iF    


  L2P *= penalty_factor;

  // Cell term
  L2P += Potential_T.transpose() * w_mass_Pkpo_T * Potential_T;

  return L2P;

}

//------------------------------------------------------------------------------------
// Build the components of the gradient operator (probably never useful, actually...)
//------------------------------------------------------------------------------------
Eigen::MatrixXd XGradStokes::buildGradientComponentsFace(size_t iF) const
{
  const Face & F = *mesh().face(iF);
  
  size_t dim_xcurl_F
    = F.n_vertices() * 6
    + F.n_edges() * 3 * (PolynomialSpaceDimension<Edge>::Poly(degree()) + PolynomialSpaceDimension<Edge>::Poly(degree()+1))
    + PolynomialSpaceDimension<Face>::Roly(degree()-1) + PolynomialSpaceDimension<Face>::RolyCompl(degree())
      + PolynomialSpaceDimension<Face>::Poly(degree()-1)
      + PolynomialSpaceDimension<Face>::Goly(degree()) + PolynomialSpaceDimension<Face>::GolyCompl(degree());

  size_t dim_xgrad_F
    = dimensionFace(F);

  Eigen::MatrixXd uGF = Eigen::MatrixXd::Zero(dim_xcurl_F, dim_xgrad_F);

  size_t offset = 0;
  // Vertex components
  for (size_t iV = 0; iV < F.n_vertices(); iV++) {
    // const Vertex & V = *F.vertex(iV);
    offset += 3;
    uGF.block(offset, 4*iV, 3, 3) = Eigen::MatrixXd::Identity(3,3);
    offset += 3;
  }
  // Edge components
  size_t dimEXcurl = 3*(PolynomialSpaceDimension<Edge>::Poly(degree())+PolynomialSpaceDimension<Edge>::Poly(degree()+1));
  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    uGF.middleRows(offset, dimEXcurl)
      = extendOperator(F, E, edgeOperators(E).gradient).bottomRows(dimEXcurl);
    offset += dimEXcurl;
  } // for iE

  // Face components
    auto GF = faceOperators(F).gradient;
    if (m_stokes_core.degree() > 0) {
      // RF^{k} projections
      MonomialFaceIntegralsType int_monoF_2k = IntegrateFaceMonomials(F, 2*degree());
      Eigen::MatrixXd mass_Rck_F = GramMatrix(F, *faceBases(F).RolyComplk, int_monoF_2k);
      Eigen::MatrixXd mass_Rkmo_F = GramMatrix(F, *faceBases(F).Rolykmo, int_monoF_2k);

      Eigen::MatrixXd pi_Rkmo_GF_F = mass_Rkmo_F.ldlt().solve(GramMatrix(F, *faceBases(F).Rolykmo, *faceBases(F).Polyk2, int_monoF_2k) * GF);

      Eigen::MatrixXd pi_Rck_GF_F = mass_Rck_F.ldlt().solve(GramMatrix(F, *faceBases(F).RolyComplk, *faceBases(F).Polyk2, int_monoF_2k) * GF);

      uGF.block(offset, 0, PolynomialSpaceDimension<Face>::Roly(degree()-1), dim_xgrad_F) = pi_Rkmo_GF_F;
      offset += PolynomialSpaceDimension<Face>::Roly(degree()-1);
      uGF.block(offset, 0, PolynomialSpaceDimension<Face>::RolyCompl(degree()), dim_xgrad_F) = pi_Rck_GF_F;
      offset += PolynomialSpaceDimension<Face>::RolyCompl(degree());
      // v_F copy
      size_t dimPkmo = PolynomialSpaceDimension<Face>::Poly(degree()-1);
      size_t offset_F = localOffset(F) + dimPkmo;
      uGF.block(offset,offset_F,dimPkmo,dimPkmo) = Eigen::MatrixXd::Identity(dimPkmo,dimPkmo);
      offset += dimPkmo;
    }
    // G components
    MonomialFaceIntegralsType int_monoF_2kpo = IntegrateFaceMonomials(F, 2*degree()+1);
    Eigen::MatrixXd mass_Gk_F = GramMatrix(F, *faceBases(F).Golyk, int_monoF_2kpo);

    auto GFperp = faceOperators(F).gradient_perp;
    QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2*(degree()+1));
    auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Polyk2, quad_2kpo_F);
    auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Golyk, quad_2kpo_F);
    Eigen::MatrixXd pi_Gk_GFp_F = mass_Gk_F.ldlt().solve(compute_gram_matrix(basis_Gk_F_quad,basis_Pk2_F_quad, quad_2kpo_F) * GFperp);
    // TODO implement this specialization
    //Eigen::MatrixXd pi_Gk_GFp_F = mass_Gk_F.ldlt().solve(GramMatrix(F, *faceBases(F).Golyk, *faceBases(F).Polyk2, int_monoF_2kpo) * GFperp);

    uGF.block(offset, 0, PolynomialSpaceDimension<Face>::Goly(degree()), dim_xgrad_F) = pi_Gk_GFp_F;
    offset += PolynomialSpaceDimension<Face>::Goly(degree());
    if (degree() > 0) { 
      // TODO implement this specialization
      //Eigen::MatrixXd pi_Gck_GFp_F = mass_Gck_F.ldlt().solve(GramMatrix(F, *faceBases(F).GolyComplk, *faceBases(F).Polyk2, int_monoF_2kpo) * GFperp);

      Eigen::MatrixXd mass_Gck_F = GramMatrix(F, *faceBases(F).GolyComplk, int_monoF_2kpo);
      auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(F).GolyComplk, quad_2kpo_F);
      Eigen::MatrixXd pi_Gck_GFp_F = mass_Gck_F.ldlt().solve(compute_gram_matrix(basis_Gck_F_quad,basis_Pk2_F_quad, quad_2kpo_F) * GFperp);
      uGF.block(offset, 0, PolynomialSpaceDimension<Face>::GolyCompl(degree()), dim_xgrad_F) = pi_Gck_GFp_F;
      offset += PolynomialSpaceDimension<Face>::GolyCompl(degree());
    } // degree() > 0

  return uGF;
}

//------------------------------------------------------------------------------------
// Build the components of the gradient operator (probably never useful, actually...)
//------------------------------------------------------------------------------------
Eigen::MatrixXd XGradStokes::buildGradientComponentsCell(size_t iT) const
{
  const Cell & T = *mesh().cell(iT);
  
  size_t dim_xcurl_T
    = T.n_vertices() * 6
    + T.n_edges() * 3 * (PolynomialSpaceDimension<Edge>::Poly(degree()) + PolynomialSpaceDimension<Edge>::Poly(degree()+1))
    + T.n_faces() * (PolynomialSpaceDimension<Face>::Roly(degree()-1) + PolynomialSpaceDimension<Face>::RolyCompl(degree())
      + PolynomialSpaceDimension<Face>::Poly(degree()-1)
      + PolynomialSpaceDimension<Face>::Goly(degree()) + PolynomialSpaceDimension<Face>::GolyCompl(degree()))
    + PolynomialSpaceDimension<Cell>::Roly(degree()-1) + PolynomialSpaceDimension<Cell>::RolyCompl(degree());

  size_t dim_xgrad_T
    = dimensionCell(T);

  Eigen::MatrixXd uGT = Eigen::MatrixXd::Zero(dim_xcurl_T, dim_xgrad_T);

  size_t offset = 0;
  // Vertex components
  for (size_t iV = 0; iV < T.n_vertices(); iV++) {
    // const Vertex & V = *T.vertex(iV);
    offset += 3;
    uGT.block(offset, 4*iV, 3, 3) = Eigen::MatrixXd::Identity(3,3);
    offset += 3;
  }
  // Edge components
  size_t dimEXcurl = 3*(PolynomialSpaceDimension<Edge>::Poly(degree())+PolynomialSpaceDimension<Edge>::Poly(degree()+1));
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    uGT.middleRows(offset, dimEXcurl)
      = extendOperator(T, E, edgeOperators(E).gradient).bottomRows(dimEXcurl);
    offset += dimEXcurl;
  } // for iE

  // Face components
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    auto GF = extendOperator(T, F, faceOperators(F).gradient);
    if (m_stokes_core.degree() > 0) {
      // RT^{k} projections
      MonomialFaceIntegralsType int_monoF_2k = IntegrateFaceMonomials(F, 2*degree());
      Eigen::MatrixXd mass_Rck_F = GramMatrix(F, *faceBases(F).RolyComplk, int_monoF_2k);
      Eigen::MatrixXd mass_Rkmo_F = GramMatrix(F, *faceBases(F).Rolykmo, int_monoF_2k);

      Eigen::MatrixXd pi_Rkmo_GF_F = mass_Rkmo_F.ldlt().solve(GramMatrix(F, *faceBases(F).Rolykmo, *faceBases(F).Polyk2, int_monoF_2k) * GF);

      Eigen::MatrixXd pi_Rck_GF_F = mass_Rck_F.ldlt().solve(GramMatrix(F, *faceBases(F).RolyComplk, *faceBases(F).Polyk2, int_monoF_2k) * GF);

      uGT.block(offset, 0, PolynomialSpaceDimension<Face>::Roly(degree()-1), dim_xgrad_T) = pi_Rkmo_GF_F;
      offset += PolynomialSpaceDimension<Face>::Roly(degree()-1);
      uGT.block(offset, 0, PolynomialSpaceDimension<Face>::RolyCompl(degree()), dim_xgrad_T) = pi_Rck_GF_F;
      offset += PolynomialSpaceDimension<Face>::RolyCompl(degree());
      // v_F copy
      size_t dimPkmo = PolynomialSpaceDimension<Face>::Poly(degree()-1);
      size_t offset_F = localOffset(T,F) + dimPkmo;
      uGT.block(offset,offset_F,dimPkmo,dimPkmo) = Eigen::MatrixXd::Identity(dimPkmo,dimPkmo);
      offset += dimPkmo;
    }
    // G components
    MonomialFaceIntegralsType int_monoF_2kpo = IntegrateFaceMonomials(F, 2*degree()+1);
    Eigen::MatrixXd mass_Gk_F = GramMatrix(F, *faceBases(F).Golyk, int_monoF_2kpo);

    auto GFperp = extendOperator(T, F, faceOperators(F).gradient_perp);
    QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2*(degree()+1));
    auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Polyk2, quad_2kpo_F);
    auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Golyk, quad_2kpo_F);
    Eigen::MatrixXd pi_Gk_GFp_F = mass_Gk_F.ldlt().solve(compute_gram_matrix(basis_Gk_F_quad,basis_Pk2_F_quad, quad_2kpo_F) * GFperp);
    // TODO implement this specialization
    //Eigen::MatrixXd pi_Gk_GFp_F = mass_Gk_F.ldlt().solve(GramMatrix(F, *faceBases(F).Golyk, *faceBases(F).Polyk2, int_monoF_2kpo) * GFperp);

    uGT.block(offset, 0, PolynomialSpaceDimension<Face>::Goly(degree()), dim_xgrad_T) = pi_Gk_GFp_F;
    offset += PolynomialSpaceDimension<Face>::Goly(degree());
    if (degree() > 0) { 
      // TODO implement this specialization
      //Eigen::MatrixXd pi_Gck_GFp_F = mass_Gck_F.ldlt().solve(GramMatrix(F, *faceBases(F).GolyComplk, *faceBases(F).Polyk2, int_monoF_2kpo) * GFperp);

      Eigen::MatrixXd mass_Gck_F = GramMatrix(F, *faceBases(F).GolyComplk, int_monoF_2kpo);
      auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(F).GolyComplk, quad_2kpo_F);
      Eigen::MatrixXd pi_Gck_GFp_F = mass_Gck_F.ldlt().solve(compute_gram_matrix(basis_Gck_F_quad,basis_Pk2_F_quad, quad_2kpo_F) * GFperp);
      uGT.block(offset, 0, PolynomialSpaceDimension<Face>::GolyCompl(degree()), dim_xgrad_T) = pi_Gck_GFp_F;
      offset += PolynomialSpaceDimension<Face>::GolyCompl(degree());
    } // degree() > 0

  } // for iF

  if (m_stokes_core.degree() > 0) {
    // Cell component
    MonomialCellIntegralsType int_mono_2k = IntegrateCellMonomials(T, 2*degree());
    Eigen::MatrixXd mass_Rkmo_T = GramMatrix(T, *m_stokes_core.cellBases(iT).Rolykmo, int_mono_2k);
    Eigen::MatrixXd mass_Rck_T = GramMatrix(T, *m_stokes_core.cellBases(iT).RolyComplk, int_mono_2k);

    Eigen::MatrixXd pi_Rkmo_GT_T = mass_Rkmo_T.ldlt().solve(GramMatrix(T, *m_stokes_core.cellBases(iT).Rolykmo, *m_stokes_core.cellBases(iT).Polyk3, int_mono_2k) * cellOperators(iT).gradient);
    Eigen::MatrixXd pi_Rck_GT_T = mass_Rck_T.ldlt().solve(
                                      GramMatrix(T, *m_stokes_core.cellBases(iT).RolyComplk, *m_stokes_core.cellBases(iT).Polyk3, int_mono_2k)
                                      * cellOperators(iT).gradient
                                  );

    uGT.block(offset, 0, PolynomialSpaceDimension<Cell>::Roly(degree()-1), dim_xgrad_T) = pi_Rkmo_GT_T;
    offset += PolynomialSpaceDimension<Cell>::Roly(degree()-1);
    uGT.block(offset, 0, PolynomialSpaceDimension<Cell>::RolyCompl(degree()), dim_xgrad_T) = pi_Rck_GT_T;
    offset += PolynomialSpaceDimension<Cell>::RolyCompl(degree());
  }
  
  return uGT;
}
