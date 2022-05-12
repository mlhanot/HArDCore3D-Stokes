#include <xsl.hpp>
#include <basis.hpp>
#include <parallel_for.hpp>
#include <GMpoly_cell.hpp>
#include <GMpoly_face.hpp>
#include <GMpoly_edge.hpp>

using namespace HArDCore3D;
//------------------------------------------------------------------------------
// XSL
//------------------------------------------------------------------------------

XSL::XSL(const DDRCore & ddr_core, bool use_threads, std::ostream & output)
  : GlobalDOFSpace(
	     ddr_core.mesh(), 
       0,
       0,
       0,
	     PolynomialSpaceDimension<Cell>::Poly(ddr_core.degree())
	      ),
    m_ddr_core(ddr_core),
    m_use_threads(use_threads),
    m_output(output)
{
  // do nothing
}


Eigen::VectorXd XSL::interpolate(const FunctionType & v,  const int doe_cell) const
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

Eigen::MatrixXd XSL::compute_Gram_Cell(size_t iT) const 
{
  const Cell & T = *mesh().cell(iT);
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2*(degree()));
  auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*cellBases(T.global_index()).Polyk,quad_2k_T);

  return compute_gram_matrix(basis_Pk_T_quad, quad_2k_T);
}

double XSL::evaluatePotential(size_t iT, const Eigen::VectorXd & uh, const VectorRd & x) const
{
  double rv = 0.;
  for (size_t i = 0; i < PolynomialSpaceDimension<Cell>::Poly(degree()); i++) {
    rv += cellBases(iT).Polyk->function(i,x)*uh(i);
  }
  return rv;
}

