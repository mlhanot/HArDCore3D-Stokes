#include <xvlstokes.hpp>
#include <basis.hpp>
#include <parallel_for.hpp>
#include <GMpoly_cell.hpp>
#include <GMpoly_face.hpp>
#include <GMpoly_edge.hpp>

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XVLStokes::XVLStokes(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : GlobalDOFSpace(
	     stokes_core.mesh(), 
       0, 
       3*PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree()+2),
	     PolynomialSpaceDimension<Face>::tildePoly(stokes_core.degree()+1),
	     PolynomialSpaceDimension<Cell>::RTb(stokes_core.degree() + 1)
	      ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output)
{
  // do nothing
}


//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XVLStokes::interpolate(const FunctionType & W,  const int doe_cell, const int doe_face, const int doe_edge) const
{
  Eigen::VectorXd Wh = Eigen::VectorXd::Zero(dimension());

  // Degrees of quadrature rules
  size_t dqr_cell = (doe_cell >= 0 ? doe_cell : 2 * degree() + 5);
  size_t dqr_face = (doe_face >= 0 ? doe_face : 2 * degree() + 4);
  size_t dqr_edge = (doe_edge >= 0 ? doe_edge : 2 * degree() + 4);

  // Interpolate at vertices

  // Interpolate at edges
  std::function<void(size_t, size_t)> interpolate_edges
    = [this, &Wh, W, &dqr_edge](size_t start, size_t end)->void
      {
	      for (size_t iE = start; iE < end; iE++) {
	        const Edge & E = *mesh().edge(iE);

          VectorRd tE = E.tangent();
	        QuadratureRule quad_dqr_E = generate_quadrature_rule(E, dqr_edge);
	        auto basis_Pk3p2_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk3p2, quad_dqr_E);
	        Wh.segment(globalOffset(E), edgeBases(iE).Polyk3p2->dimension())
	          = l2_projection([W,tE](const Eigen::Vector3d & x)->Eigen::Vector3d {
              return W(x)*tE;
            },
            *edgeBases(iE).Polyk3p2, quad_dqr_E, basis_Pk3p2_E_quad);
	      } // for iE
      };
  parallel_for(mesh().n_edges(), interpolate_edges, m_use_threads); 

  // Interpolate at faces
  std::function<void(size_t, size_t)> interpolate_faces
    = [this, &Wh, W, &dqr_face](size_t start, size_t end)->void
      {
      for (size_t iF = start; iF < end; iF++) {
        const Face & F = *mesh().face(iF);

        QuadratureRule quad_dqr_F = generate_quadrature_rule(F, dqr_face);

        auto basis_tildePkpo_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).tildePolykpo, quad_dqr_F);
        Wh.segment(globalOffset(F), PolynomialSpaceDimension<Face>::tildePoly(degree()+1))
          = l2_projection(W, *faceBases(iF).tildePolykpo, quad_dqr_F, basis_tildePkpo_F_quad);
      } // for iF
	  }; 
      
  parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);
  
  // Interpolate at cells
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &Wh, W, &dqr_cell](size_t start, size_t end)->void
      {
      for (size_t iT = start; iT < end; iT++) {		 
        const Cell & T = *mesh().cell(iT);

        QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr_cell);
        auto basis_RTbkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo,quad_dqr_T);
        Wh.segment(globalOffset(T), PolynomialSpaceDimension<Cell>::RTb(degree() + 1))
          = l2_projection(W, *cellBases(iT).RTbkpo, quad_dqr_T, basis_RTbkpo_T_quad);
      } // for iT
    };
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
	
  return Wh;
}

Eigen::MatrixXd XVLStokes::compute_Gram_Edge(size_t iE) const 
{
  const Edge & E = *mesh().edge(iE);
  QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2*(degree()+2));
  auto basis_Pkp2_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polykp2,quad_2kp2_E);
  Eigen::MatrixXd gram_Pkp2_E = compute_gram_matrix(basis_Pkp2_E_quad, quad_2kp2_E);
  // Polyk3p2 is made of 3 copies of Polykp2
  size_t dim_Pkp2_E = PolynomialSpaceDimension<Edge>::Poly(degree()+2);
  Eigen::MatrixXd L2E = Eigen::MatrixXd::Zero(3*dim_Pkp2_E,3*dim_Pkp2_E);
  for (size_t i = 0; i < 3; i++) {
    L2E.block(i*dim_Pkp2_E,i*dim_Pkp2_E,dim_Pkp2_E,dim_Pkp2_E) = gram_Pkp2_E;
  }

  return L2E;
}

Eigen::MatrixXd XVLStokes::compute_Gram_Face(size_t iF) const 
{
  const Face & F = *mesh().face(iF);
  QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2*(degree()+1));
  auto basis_tildePkpo_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).tildePolykpo,quad_2kpo_F);

  return compute_gram_matrix(basis_tildePkpo_F_quad, quad_2kpo_F);
}

Eigen::MatrixXd XVLStokes::compute_Gram_Cell(size_t iT) const 
{
  const Cell & T = *mesh().cell(iT);
  QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2*(degree()+1));
  auto basis_RTbkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(T.global_index()).RTbkpo,quad_2kpo_T);

  return compute_gram_matrix(basis_RTbkpo_T_quad, quad_2kpo_T);
}

Eigen::MatrixXd XVLStokes::computeL2Product_GG(
                                        const size_t iT,
                                        const XNablaStokes & xnabla,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & mass_RTbkpo_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 

  // create the weighted mass matrix, with simple product if weight is constant
  Eigen::MatrixXd w_mass_RTbkpo_T;
  if (weight.deg(T)==0){
    // constant weight
    if (mass_RTbkpo_T.rows()==1){
      // We have to compute the mass matrix
      QuadratureRule quad_2kpo_T = generate_quadrature_rule(T,2*(degree()+1));
      auto basis_RTbkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kpo_T);
      w_mass_RTbkpo_T = compute_gram_matrix(basis_RTbkpo_T_quad,quad_2kpo_T);
      //MonomialCellIntegralsType int_mono_2kpo = IntegrateCellMonomials(T, 2*(degree()+1));
      //w_mass_RTbkpo_T = weight.value(T, T.center_mass()) * GramMatrix(T, *cellBases(iT).RTbkpo, int_mono_2kpo);
    }else{
      w_mass_RTbkpo_T = weight.value(T, T.center_mass()) * mass_RTbkpo_T;
    }
  }else{
    // weight is not constant, we create a weighted mass matrix
    QuadratureRule quad_2kpw_T = generate_quadrature_rule(T, 2 * (degree()+1) + weight.deg(T));
    auto basis_RTbkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kpw_T);
    std::function<double(const Eigen::Vector3d &)> weight_T 
              = [&T, &weight](const Eigen::Vector3d &x)->double {
                  return weight.value(T, x);
                };
    w_mass_RTbkpo_T = compute_weighted_gram_matrix(weight_T, basis_RTbkpo_T_quad, basis_RTbkpo_T_quad, quad_2kpw_T, "sym");
  }

  // leftOp and rightOp come from nabla
  std::vector<Eigen::MatrixXd> nablaOp(T.n_edges() + T.n_faces() + 1);
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    nablaOp[iE] = xnabla.extendOperator(T, E, xnabla.edgeOperators(E.global_index()).nabla);
  }
  for (size_t iF = 0; iF < T.n_faces(); iF++){
    const Face & F = *T.face(iF);
    nablaOp[T.n_edges() + iF] = xnabla.extendOperator(T, F, xnabla.faceOperators(F.global_index()).nabla);
  }
  nablaOp[T.n_edges() + T.n_faces()] = xnabla.cellOperators(iT).nabla;

  return computeL2Product_with_Ops(iT, nablaOp, nablaOp, penalty_factor, w_mass_RTbkpo_T, weight);

}


Eigen::MatrixXd XVLStokes::computeL2Product_with_Ops(
                                        const size_t iT,
                                        const std::vector<Eigen::MatrixXd> & leftOp,
                                        const std::vector<Eigen::MatrixXd> & rightOp,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & w_mass_RTbkpo_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 

  // leftOp and rightOp must list the operators acting on the DOFs, and which we want to
  // use for the L2 product. Specifically, each one lists operators (matrices) returning
  // values in edges space P^k+2(E)^3, faces space tildeP^k+1(F) and element space RTb^k+1(T).
  // All these operators must have the same domain, so possibly being extended appropriately
  // using extendOperator from globaldofspace.

  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(leftOp[0].cols(), rightOp[0].cols());
  
  size_t offset_F = T.n_edges();
  size_t offset_T = T.n_edges() + T.n_faces();
  
  // Edge penalty terms
  // Counted twices outside the faces loop
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    VectorRd tE = E.tangent();

    // Compute gram matrices
    QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2 * (degree()+2));
    
    auto basis_Pk3p2_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk3p2, quad_2kp2_E);
    auto basis_RTbkpo_TE_quad = matrix_vector_product(evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kp2_E),tE);
    Eigen::MatrixXd gram_EE = compute_gram_matrix(basis_Pk3p2_E_quad,quad_2kp2_E);
    Eigen::MatrixXd gram_TT = compute_gram_matrix(basis_RTbkpo_TE_quad,quad_2kp2_E);
    Eigen::MatrixXd gram_TE = compute_gram_matrix(basis_RTbkpo_TE_quad,basis_Pk3p2_E_quad,quad_2kp2_E,"nonsym");

    // Weight including scaling hE (we compute the max over quadrature nodes to get an estimate of the max of the weight over the edge)
    double max_weight_quad_E = weight.value(T, quad_2kp2_E[0].vector());
    // If the weight is not constant, we want to take the largest along the edge
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2kp2_E.size(); iqn++) {
        max_weight_quad_E = std::max(max_weight_quad_E, weight.value(T, quad_2kp2_E[iqn].vector()));
      } // for
    }
    double w_hE2 = 2.*max_weight_quad_E * E.diam()*E.diam();

    // The penalty term int_E (leftOp - (leftOp)_E) * (rightOp - (rightOp)_E) is computed by developping
    // Contribution of edge E
    L2P += w_hE2 * ( leftOp[offset_T].transpose() * gram_TT * rightOp[offset_T]
                - leftOp[offset_T].transpose() * gram_TE * rightOp[iE] 
                - leftOp[iE].transpose() * gram_TE.transpose() * rightOp[offset_T]
                + leftOp[iE].transpose() * gram_EE * rightOp[iE]                  
                  );
  } // for iE

  // Face penalty terms
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    VectorRd nF = F.normal();

    // Compute gram matrices
    QuadratureRule quad_2kp2_F = generate_quadrature_rule(F, 2 * (degree()+2));
    
    auto basis_tildePkpo_F_quad = evaluate_quad<Function>::compute(*faceBases(F).tildePolykpo, quad_2kp2_F);
    auto basis_RTbkpo_TF_quad = tensor_tangent_product(evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kp2_F),nF);
    Eigen::MatrixXd gram_FF = compute_gram_matrix(basis_tildePkpo_F_quad,quad_2kp2_F);
    Eigen::MatrixXd gram_TT = compute_gram_matrix(basis_RTbkpo_TF_quad,quad_2kp2_F);
    Eigen::MatrixXd gram_TF = compute_gram_matrix(basis_RTbkpo_TF_quad,basis_tildePkpo_F_quad,quad_2kp2_F,"nonsym");

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
  L2P += leftOp[offset_T].transpose() * w_mass_RTbkpo_T * rightOp[offset_T];
 
  return L2P;
}

//------------------------------------------------------------------------------
// XSL
//------------------------------------------------------------------------------

XSLStokes::XSLStokes(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : GlobalDOFSpace(
	     stokes_core.mesh(), 
       0,
       0,
       0,
	     PolynomialSpaceDimension<Cell>::Poly(stokes_core.degree())
	      ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output)
{
  // do nothing
}


Eigen::VectorXd XSLStokes::interpolate(const FunctionType & v,  const int doe_cell) const
{
  size_t dim_Pk_T = PolynomialSpaceDimension<Cell>::Poly(degree());
  Eigen::VectorXd vh = Eigen::VectorXd::Zero(mesh().n_cells()*dim_Pk_T);

  // Degrees of quadrature rules
  size_t dqr_cell = (doe_cell >= 0 ? doe_cell : 2 * degree() + 5);

  // Interpolate at vertices

  // Interpolate at edges

  // Interpolate at faces
  
  // Interpolate at cells
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &vh, v, &dqr_cell, dim_Pk_T](size_t start, size_t end)->void
{
  for (size_t iT = start; iT < end; iT++) {		 
    const Cell & T = *mesh().cell(iT);
    QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr_cell);
    auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk, quad_dqr_T);
    MonomialCellIntegralsType int_mono_2k = IntegrateCellMonomials(T, 2*degree());
    vh.segment(iT*dim_Pk_T, dim_Pk_T) 
            = l2_projection(v, *cellBases(iT).Polyk, quad_dqr_T, basis_Pk_T_quad, GramMatrix(T, *cellBases(iT).Polyk, int_mono_2k));
  } // for iT
};
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
	
  return vh;
}

Eigen::MatrixXd XSLStokes::compute_Gram_Cell(size_t iT) const 
{
  const Cell & T = *mesh().cell(iT);
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2*(degree()));
  auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*cellBases(T.global_index()).Polyk,quad_2k_T);

  return compute_gram_matrix(basis_Pk_T_quad, quad_2k_T);
}

double XSLStokes::evaluatePotential(size_t iT, const Eigen::VectorXd & uh, const VectorRd & x) const
{
  double rv = 0.;
  for (size_t i = 0; i < PolynomialSpaceDimension<Cell>::Poly(degree()); i++) {
    rv += cellBases(iT).Polyk->function(i,x)*uh(i);
  }
  return rv;
}

