#include "xcurlstokes.hpp"

#include <basis.hpp>
#include <parallel_for.hpp>
#include <GMpoly_cell.hpp>
#include <GMpoly_face.hpp>
#include <GMpoly_edge.hpp>

using namespace HArDCore3D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XCurlStokes::XCurlStokes(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : GlobalDOFSpace(
	     stokes_core.mesh(),
	     6,
	     3*PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree())
	      + 3*PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree() + 1),
	     PolynomialSpaceDimension<Face>::Roly(stokes_core.degree() - 1) + PolynomialSpaceDimension<Face>::RolyCompl(stokes_core.degree())
	      + PolynomialSpaceDimension<Face>::Poly(stokes_core.degree() - 1)
        + PolynomialSpaceDimension<Face>::Goly(stokes_core.degree()) + PolynomialSpaceDimension<Face>::GolyCompl(stokes_core.degree()),
	     PolynomialSpaceDimension<Cell>::Roly(stokes_core.degree() - 1) + PolynomialSpaceDimension<Cell>::RolyCompl(stokes_core.degree())	     
	     ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output),
    m_cell_operators(stokes_core.mesh().n_cells()),
    m_face_operators(stokes_core.mesh().n_faces()),
    m_edge_operators(stokes_core.mesh().n_edges())
{
  output << "[XCurlStokes] Initializing" << std::endl;
  if (use_threads) {
    m_output << "[XCurlStokes] Parallel execution" << std::endl;
  } else {
    m_output << "[XCurlStokes] Sequential execution" << std::endl;
  }
  // Construct edge curls and potentials
  std::function<void(size_t, size_t)> construct_all_edge_curls_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iE = start; iE < end; iE++) {
          m_edge_operators[iE].reset( new LocalOperators(_compute_edge_curl_potential(iE)) );
        } // for iE
      };

  m_output << "[XGradStokes] Constructing edge curls and potentials" << std::endl;
  parallel_for(mesh().n_edges(), construct_all_edge_curls_potentials, use_threads);

  // Construct face curls and potentials
  std::function<void(size_t, size_t)> construct_all_face_curls_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iF = start; iF < end; iF++) {
          m_face_operators[iF].reset( new LocalOperators(_compute_face_curl_potential(iF)) );
        } // for iF
      };

  m_output << "[XCurlStokes] Constructing face curls and potentials" << std::endl;
  parallel_for(mesh().n_faces(), construct_all_face_curls_potentials, use_threads);

  // Construct cell curls and potentials
  std::function<void(size_t, size_t)> construct_all_cell_curls_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          m_cell_operators[iT].reset( new LocalOperators(_compute_cell_curl_potential(iT)) );
        } // for iT
      };

  m_output << "[XCurlStokes] Constructing cell curls and potentials" << std::endl;
  parallel_for(mesh().n_cells(), construct_all_cell_curls_potentials, use_threads);  
}

//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XCurlStokes::interpolate(const FunctionType & v, const FunctionCurlType & Cv, const FunctionGradType & Gv, const int doe_cell, const int doe_face, const int doe_edge) const
{
  Eigen::VectorXd vh = Eigen::VectorXd::Zero(dimension());
  
  // Degrees of quadrature rules
  size_t dqr_cell = (doe_cell >= 0 ? doe_cell : 2 * degree() + 3);
  size_t dqr_face = (doe_face >= 0 ? doe_face : 2 * degree() + 3);
  size_t dqr_edge = (doe_edge >= 0 ? doe_edge : 2 * degree() + 3);
  
  // Interpolate at vertices
  std::function<void(size_t, size_t)> interpolate_vertices
    = [this, &vh, v, Cv](size_t start, size_t end)->void
      {
        for (size_t iV = start; iV < end; iV++) {
          vh.segment(6*iV,3) = Cv(mesh().vertex(iV)->coords());
          vh.segment(6*iV + 3,3) = v(mesh().vertex(iV)->coords());
        } // for iV
      };
  parallel_for(mesh().n_vertices(), interpolate_vertices, m_use_threads); 

  // Interpolate at edges
  std::function<void(size_t, size_t)> interpolate_edges
    = [this, &vh, v, Cv, Gv, &dqr_edge](size_t start, size_t end)->void
      {
	      for (size_t iE = start; iE < end; iE++) {
	        const Edge & E = *mesh().edge(iE);
	        Eigen::Vector3d tE = E.tangent();
          std::vector<Eigen::Vector3d> basisE = E.edge_normalbasis();
	        
          QuadratureRule quad_dqr_E = generate_quadrature_rule(E, dqr_edge);
	        auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_dqr_E);
          auto v_E_tE = [&tE, v](const Eigen::Vector3d & x)->double {
			          return v(x).dot(tE);
			        };
          auto v_E_nE1 = [&basisE, v](const Eigen::Vector3d & x)->double {
			          return v(x).dot(basisE[0]);
			        };
          auto v_E_nE2 = [&basisE, v](const Eigen::Vector3d & x)->double {
			          return v(x).dot(basisE[1]);
			        };

          size_t offset_E = globalOffset(E);
	        vh.segment(offset_E, edgeBases(iE).Polyk->dimension())
	          = l2_projection(v_E_tE, *edgeBases(iE).Polyk, quad_dqr_E, basis_Pk_E_quad);
          offset_E += PolynomialSpaceDimension<Edge>::Poly(degree());
	        vh.segment(offset_E, edgeBases(iE).Polyk->dimension())
	          = l2_projection(v_E_nE1, *edgeBases(iE).Polyk, quad_dqr_E, basis_Pk_E_quad);
          offset_E += PolynomialSpaceDimension<Edge>::Poly(degree());
	        vh.segment(offset_E, edgeBases(iE).Polyk->dimension())
	          = l2_projection(v_E_nE2, *edgeBases(iE).Polyk, quad_dqr_E, basis_Pk_E_quad);

	        auto R_v_E_x = [&tE, Cv, Gv](const Eigen::Vector3d & x)->double {
			          return (Cv(x).dot(tE)*tE + (Gv(x).transpose()*tE).cross(tE))(0);
			        };
	        auto R_v_E_y = [&tE, Cv, Gv](const Eigen::Vector3d & x)->double {
			          return (Cv(x).dot(tE)*tE + (Gv(x).transpose()*tE).cross(tE))(1);
			        };
	        auto R_v_E_z = [&tE, Cv, Gv](const Eigen::Vector3d & x)->double {
			          return (Cv(x).dot(tE)*tE + (Gv(x).transpose()*tE).cross(tE))(2);
			        };
	        auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykpo, quad_dqr_E);
          offset_E += PolynomialSpaceDimension<Edge>::Poly(degree());
	        vh.segment(offset_E,edgeBases(iE).Polykpo->dimension())
	          = l2_projection(R_v_E_x, *edgeBases(iE).Polykpo, quad_dqr_E, basis_Pkpo_E_quad);
          offset_E += PolynomialSpaceDimension<Edge>::Poly(degree()+1);
	        vh.segment(offset_E,edgeBases(iE).Polykpo->dimension())
	          = l2_projection(R_v_E_y, *edgeBases(iE).Polykpo, quad_dqr_E, basis_Pkpo_E_quad);
          offset_E += PolynomialSpaceDimension<Edge>::Poly(degree()+1);
	        vh.segment(offset_E,edgeBases(iE).Polykpo->dimension())
	          = l2_projection(R_v_E_z, *edgeBases(iE).Polykpo, quad_dqr_E, basis_Pkpo_E_quad);

	      } // for iE
      };
  parallel_for(mesh().n_edges(), interpolate_edges, m_use_threads);

  if (degree() == 0 ) {
    // Interpolate at faces
    std::function<void(size_t, size_t)> interpolate_faces
      = [this, &vh, Gv, &dqr_face](size_t start, size_t end)->void
	      {
	        for (size_t iF = start; iF < end; iF++) {
	          const Face & F = *mesh().face(iF);

	          Eigen::Vector3d nF = F.normal();
	          auto nF_cross_Gv_nF = [&nF, Gv](const Eigen::Vector3d & x)->Eigen::Vector3d {
                                         return nF.cross(Gv(x)*nF);
                                       };

	          QuadratureRule quad_dqr_F = generate_quadrature_rule(F, dqr_face);

	          size_t offset_F = globalOffset(F);
	          auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Golyk, quad_dqr_F);
	          vh.segment(offset_F, PolynomialSpaceDimension<Face>::Goly(degree()))
	            = l2_projection(nF_cross_Gv_nF, *faceBases(iF).Golyk, quad_dqr_F, basis_Gk_F_quad);

	          offset_F += PolynomialSpaceDimension<Face>::Goly(degree());
            // Gck = 0 when k = 0
	          //auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).GolyComplk, quad_dqr_F);
	          //vh.segment(offset_F, PolynomialSpaceDimension<Face>::GolyCompl(degree()))
	            //= l2_projection(nF_cross_Gv_nF, *faceBases(iF).GolyComplk, quad_dqr_F, basis_Gck_F_quad);
	        } // for iF
	      };
    parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);
  } else { // degree > 0
    // Interpolate at faces
    std::function<void(size_t, size_t)> interpolate_faces
      = [this, &vh, v, Gv, &dqr_face](size_t start, size_t end)->void
	      {
	        for (size_t iF = start; iF < end; iF++) {
	          const Face & F = *mesh().face(iF);

	          Eigen::Vector3d nF = F.normal();
	          auto nF_cross_v_cross_nF = [&nF, v](const Eigen::Vector3d & x)->Eigen::Vector3d {
                                         return nF.cross(v(x).cross(nF));
                                       };

	          QuadratureRule quad_dqr_F = generate_quadrature_rule(F, dqr_face);

	          size_t offset_F = globalOffset(F);
	          auto basis_Rkmo_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Rolykmo, quad_dqr_F);
	          vh.segment(offset_F, PolynomialSpaceDimension<Face>::Roly(degree() - 1))
	            = l2_projection(nF_cross_v_cross_nF, *faceBases(iF).Rolykmo, quad_dqr_F, basis_Rkmo_F_quad);

	          offset_F += PolynomialSpaceDimension<Face>::Roly(degree() - 1);
	          auto basis_Rck_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).RolyComplk, quad_dqr_F);
	          vh.segment(offset_F, PolynomialSpaceDimension<Face>::RolyCompl(degree()))
	            = l2_projection(nF_cross_v_cross_nF, *faceBases(iF).RolyComplk, quad_dqr_F, basis_Rck_F_quad);

	          auto v_dot_nF = [&nF, v](const Eigen::Vector3d & x)->double {
                                         return v(x).dot(nF);
                                       };

	          offset_F += PolynomialSpaceDimension<Face>::RolyCompl(degree());
	          auto basis_Pkmo_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Polykmo, quad_dqr_F);
	          vh.segment(offset_F, PolynomialSpaceDimension<Face>::Poly(degree()-1))
	            = l2_projection(v_dot_nF, *faceBases(iF).Polykmo, quad_dqr_F, basis_Pkmo_F_quad);

            // R_{v,F}
	          auto nF_cross_Gv_nF = [&nF, Gv](const Eigen::Vector3d & x)->Eigen::Vector3d {
                                         return nF.cross(Gv(x)*nF);
                                       };

            offset_F += PolynomialSpaceDimension<Face>::Poly(degree()-1);
	          auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).Golyk, quad_dqr_F);
	          vh.segment(offset_F, PolynomialSpaceDimension<Face>::Goly(degree()))
	            = l2_projection(nF_cross_Gv_nF, *faceBases(iF).Golyk, quad_dqr_F, basis_Gk_F_quad);

	          offset_F += PolynomialSpaceDimension<Face>::Goly(degree());
	          auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(iF).GolyComplk, quad_dqr_F);
	          vh.segment(offset_F, PolynomialSpaceDimension<Face>::GolyCompl(degree()))
	            = l2_projection(nF_cross_Gv_nF, *faceBases(iF).GolyComplk, quad_dqr_F, basis_Gck_F_quad);

	        } // for iF
	      };
    parallel_for(mesh().n_faces(), interpolate_faces, m_use_threads);

    // Interpolate at cells
    std::function<void(size_t, size_t)> interpolate_cells
      = [this, &vh, v, &dqr_cell](size_t start, size_t end)->void
	      {
	        for (size_t iT = start; iT < end; iT++) {
	          const Cell & T = *mesh().cell(iT);

	          QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr_cell);
	          MonomialCellIntegralsType int_mono_2k = IntegrateCellMonomials(T, 2*degree());

	          Eigen::Index offset_T = globalOffset(T);
	          auto basis_Rkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Rolykmo, quad_dqr_T);
	          vh.segment(offset_T, PolynomialSpaceDimension<Cell>::Roly(degree() - 1))
	            = l2_projection(v, *cellBases(iT).Rolykmo, quad_dqr_T, basis_Rkmo_T_quad, GramMatrix(T, *cellBases(iT).Rolykmo, int_mono_2k));

	          offset_T += PolynomialSpaceDimension<Cell>::Roly(degree() - 1);
	          auto basis_Rck_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).RolyComplk, quad_dqr_T);
	          vh.segment(offset_T, PolynomialSpaceDimension<Cell>::RolyCompl(degree()))
	            = l2_projection(v, *cellBases(iT).RolyComplk, quad_dqr_T, basis_Rck_T_quad, GramMatrix(T, *cellBases(iT).RolyComplk, int_mono_2k));
	        } // for iT
	      };
    parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);
  } // if degree() > 0
  
  return vh;
}

//------------------------------------------------------------------------------
// Curl and potential reconstruction
//------------------------------------------------------------------------------

XCurlStokes::LocalOperators XCurlStokes::_compute_edge_curl_potential(size_t iE)
{  
  const Edge & E = *mesh().edge(iE);

  //------------------------------------------------------------------------------
  // Curl
  //------------------------------------------------------------------------------
  
  // v_E'
  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  MonomialEdgeIntegralsType int_mono_2kp3_E = IntegrateEdgeMonomials(E, 2*degree()+3);
  // TODO
  //Eigen::MatrixXd MCE_vE = GramMatrix(E, *edgeBases(iE).Polyk3po, int_mono_2kp3_E);
  QuadratureRule quad_2kp3_E = generate_quadrature_rule(E,2*degree()+3);
  Eigen::MatrixXd MCE_vE = compute_gram_matrix(evaluate_quad<Function>::compute(*edgeBases(iE).Polyk3po, quad_2kp3_E),quad_2kp3_E);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BCE_vE 
    = Eigen::MatrixXd::Zero(edgeBases(iE).Polyk3po->dimension(), dimensionEdge(iE));
  for (size_t i = 0; i < edgeBases(iE).Polyk3po->dimension(); i++) {
    BCE_vE(i, 3) = -edgeBases(iE).Polyk3po->function(i,mesh().edge(iE)->vertex(0)->coords())(0); // x component
    BCE_vE(i, 4) = -edgeBases(iE).Polyk3po->function(i,mesh().edge(iE)->vertex(0)->coords())(1); // y component
    BCE_vE(i, 5) = -edgeBases(iE).Polyk3po->function(i,mesh().edge(iE)->vertex(0)->coords())(2); // z component
    BCE_vE(i, 9) = edgeBases(iE).Polyk3po->function(i,mesh().edge(iE)->vertex(1)->coords())(0); // x component
    BCE_vE(i, 10) = edgeBases(iE).Polyk3po->function(i,mesh().edge(iE)->vertex(1)->coords())(1); // y component
    BCE_vE(i, 11) = edgeBases(iE).Polyk3po->function(i,mesh().edge(iE)->vertex(1)->coords())(2); // z component
  } // for i 
  GradientBasis<StokesCore::PolyBasisEdgeType> grad_Pkpo_E(*edgeBases(iE).Polykpo);
  Eigen::MatrixXd BCE_GPkpo_Pk = GramMatrix(E, grad_Pkpo_E, *edgeBases(iE).Polyk, int_mono_2kp3_E);
  // Columns are stored as tE, nE1, nE2
  // Rows as x, y, z
  size_t dim_Pk_E = PolynomialSpaceDimension<Edge>::Poly(degree());
  size_t dim_Pkpo_E = PolynomialSpaceDimension<Edge>::Poly(degree()+1);
  Eigen::Vector3d tE = E.tangent();
  std::vector<Eigen::Vector3d> basisE = E.edge_normalbasis();
  BCE_vE.block(0*dim_Pkpo_E,12+0*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -tE(0)*BCE_GPkpo_Pk; // tE . x
  BCE_vE.block(0*dim_Pkpo_E,12+1*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -basisE[0](0)*BCE_GPkpo_Pk; // nE1 . x
  BCE_vE.block(0*dim_Pkpo_E,12+2*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -basisE[1](0)*BCE_GPkpo_Pk; // nE2 . x
  // Rows y
  BCE_vE.block(1*dim_Pkpo_E,12+0*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -tE(1)*BCE_GPkpo_Pk; // tE . y
  BCE_vE.block(1*dim_Pkpo_E,12+1*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -basisE[0](1)*BCE_GPkpo_Pk; // nE1 . y
  BCE_vE.block(1*dim_Pkpo_E,12+2*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -basisE[1](1)*BCE_GPkpo_Pk; // nE2 . y
  // Rows z
  BCE_vE.block(2*dim_Pkpo_E,12+0*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -tE(2)*BCE_GPkpo_Pk; // tE . z
  BCE_vE.block(2*dim_Pkpo_E,12+1*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -basisE[0](2)*BCE_GPkpo_Pk; // nE1 . z
  BCE_vE.block(2*dim_Pkpo_E,12+2*dim_Pk_E,dim_Pkpo_E,dim_Pk_E) = -basisE[1](2)*BCE_GPkpo_Pk; // nE2 . z
  
  // v_E' = MCE_vE^-1 BCE_vE  // [x y z]
  // Matrix to map [x//y//z] -> [x//y//z//]xtE : M = [[0 & t3 & -t2],[-t3 & 0 & t1],[t2 & -t1 & 0]]
  Eigen::MatrixXd M_map_xtE = Eigen::MatrixXd::Zero(3*dim_Pkpo_E,3*dim_Pkpo_E);
  M_map_xtE.block(0*dim_Pkpo_E,1*dim_Pkpo_E,dim_Pkpo_E,dim_Pkpo_E) = tE(2)*Eigen::MatrixXd::Identity(dim_Pkpo_E,dim_Pkpo_E);
  M_map_xtE.block(0*dim_Pkpo_E,2*dim_Pkpo_E,dim_Pkpo_E,dim_Pkpo_E) = -tE(1)*Eigen::MatrixXd::Identity(dim_Pkpo_E,dim_Pkpo_E);
  M_map_xtE.block(1*dim_Pkpo_E,0*dim_Pkpo_E,dim_Pkpo_E,dim_Pkpo_E) = -tE(2)*Eigen::MatrixXd::Identity(dim_Pkpo_E,dim_Pkpo_E);
  M_map_xtE.block(1*dim_Pkpo_E,2*dim_Pkpo_E,dim_Pkpo_E,dim_Pkpo_E) = tE(0)*Eigen::MatrixXd::Identity(dim_Pkpo_E,dim_Pkpo_E);
  M_map_xtE.block(2*dim_Pkpo_E,0*dim_Pkpo_E,dim_Pkpo_E,dim_Pkpo_E) = tE(1)*Eigen::MatrixXd::Identity(dim_Pkpo_E,dim_Pkpo_E);
  M_map_xtE.block(2*dim_Pkpo_E,1*dim_Pkpo_E,dim_Pkpo_E,dim_Pkpo_E) = -tE(0)*Eigen::MatrixXd::Identity(dim_Pkpo_E,dim_Pkpo_E);
  // v_E' x tE = M_map_xtE * MCE_vE^-1 * BCE_vE 

  // map R_{v,V} & R_{v,E} and add - v_E' x tE
  Eigen::MatrixXd CE_vE 
    = Eigen::MatrixXd::Zero(6+edgeBases(iE).Polyk3po->dimension(), dimensionEdge(iE));
  CE_vE.block(0,0,3,3) = Eigen::Matrix3d::Identity(); // First vertex
  CE_vE.block(3,6,3,3) = Eigen::Matrix3d::Identity(); // Second vertex
  CE_vE.block(6,12+3*dim_Pk_E,3*dim_Pkpo_E,3*dim_Pkpo_E) = Eigen::MatrixXd::Identity(3*dim_Pkpo_E,3*dim_Pkpo_E);
  
  CE_vE.bottomRows(3*dim_Pkpo_E) -= M_map_xtE * MCE_vE.ldlt().solve(BCE_vE);


  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  // Pk+2 [x][y][z]
  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BPE
    = Eigen::MatrixXd::Zero(3*(PolynomialSpaceDimension<Edge>::Poly(degree()) + 2), dimensionEdge(iE));

  // Enforce the gradient of the potential reconstruction
  BPE.topRows(3*PolynomialSpaceDimension<Edge>::Poly(degree()+1)) = BCE_vE;

  // Enforce the average value of the potential reconstruction
  QuadratureRule quad_k_E = generate_quadrature_rule(E, degree());
  auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_k_E);
  
  // We set the average value of the potential equal to the average of the edge unknown
  for (size_t i = 0; i < PolynomialSpaceDimension<Edge>::Poly(degree()); i++) {
    for (size_t iqn = 0; iqn < quad_k_E.size(); iqn++) {
      for (size_t j = 0; j < 3; j++) {
        BPE.middleRows(3*dim_Pkpo_E+j,1)(0, 12 + 0*dim_Pk_E + i) += tE(j)*quad_k_E[iqn].w * basis_Pk_E_quad[i][iqn];
        BPE.middleRows(3*dim_Pkpo_E+j,1)(0, 12 + 1*dim_Pk_E + i) += basisE[0](j)*quad_k_E[iqn].w * basis_Pk_E_quad[i][iqn];
        BPE.middleRows(3*dim_Pkpo_E+j,1)(0, 12 + 2*dim_Pk_E + i) += basisE[1](j)*quad_k_E[iqn].w * basis_Pk_E_quad[i][iqn];
      } // for j 
    } // for iqn
  } // for i
  
  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  Eigen::MatrixXd MPE
    = Eigen::MatrixXd::Zero(3*PolynomialSpaceDimension<Edge>::Poly(degree()+2), 3*PolynomialSpaceDimension<Edge>::Poly(degree() + 2));

  GradientBasis<StokesCore::PolyBasisEdgeType> grad_Pkp2_E(*edgeBases(iE).Polykp2);
  Eigen::MatrixXd M_Pkpo_GPkp2 = GramMatrix(E, *edgeBases(iE).Polykpo, grad_Pkp2_E, int_mono_2kp3_E);
  size_t dim_Pkp2_E = PolynomialSpaceDimension<Edge>::Poly(degree()+2);
  MPE.block(0*dim_Pkpo_E,0*dim_Pkp2_E,dim_Pkpo_E,dim_Pkp2_E) = M_Pkpo_GPkp2; // x
  MPE.block(1*dim_Pkpo_E,1*dim_Pkp2_E,dim_Pkpo_E,dim_Pkp2_E) = M_Pkpo_GPkp2; // y
  MPE.block(2*dim_Pkpo_E,2*dim_Pkp2_E,dim_Pkpo_E,dim_Pkp2_E) = M_Pkpo_GPkp2; // z

  // Average
  MonomialScalarBasisEdge basis_P0_E(E, 0);
  Eigen::MatrixXd M_P0_Pkp2 = GramMatrix(E, basis_P0_E, *edgeBases(iE).Polykp2, int_mono_2kp3_E);
  MPE.block(3*dim_Pkpo_E+0,0*dim_Pkp2_E,1,dim_Pkp2_E) = M_P0_Pkp2; // x
  MPE.block(3*dim_Pkpo_E+1,1*dim_Pkp2_E,1,dim_Pkp2_E) = M_P0_Pkp2; // y
  MPE.block(3*dim_Pkpo_E+2,2*dim_Pkp2_E,1,dim_Pkp2_E) = M_P0_Pkp2; // z

  return LocalOperators(CE_vE, Eigen::MatrixXd(), MPE.partialPivLu().solve(BPE));
}

  
XCurlStokes::LocalOperators XCurlStokes::_compute_face_curl_potential(size_t iF)
{
  const Face & F = *mesh().face(iF);
  VectorRd nF = F.normal();
  
  //------------------------------------------------------------------------------
  // Curl
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  MonomialFaceIntegralsType int_monoF_2kpo = IntegrateFaceMonomials(F, 2*degree()+1);
  Eigen::MatrixXd MCF = GramMatrix(F, *faceBases(iF).Polyk, int_monoF_2kpo);
  
  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BCF
    = Eigen::MatrixXd::Zero(faceBases(iF).Polyk->dimension(), dimensionFace(iF));

  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    QuadratureRule quad_2k_E = generate_quadrature_rule(E, 2 * degree());
    BCF.block(0, localOffset(F, E), faceBases(iF).Polyk->dimension(), edgeBases(E.global_index()).Polyk->dimension())
      -= F.edge_orientation(iE) * compute_gram_matrix(
						      evaluate_quad<Function>::compute(*faceBases(iF).Polyk, quad_2k_E),
						      evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk, quad_2k_E),
						      quad_2k_E
						      );
  } // for iE

  if (degree() > 0) {
    CurlBasis<StokesCore::PolyBasisFaceType> rot_PkF(*faceBases(iF).Polyk);
    BCF.block(0, localOffset(F), faceBases(iF).Polyk->dimension(), faceBases(iF).Rolykmo->dimension())
      += GramMatrix(F, rot_PkF, *faceBases(iF).Rolykmo, int_monoF_2kpo);
  } // if degree() > 0
 
  Eigen::MatrixXd CF = MCF.ldlt().solve(BCF);

  //------------------------------------------------------------------------------
  // Curl vec
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  // TODO implement monomial
  QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2 * degree() + 1);
  Eigen::MatrixXd MCF_vec = compute_gram_matrix(evaluate_quad<Function>::compute(*faceBases(iF).Polyk2, quad_2kpo_F),quad_2kpo_F);
  
  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BCF_vec
    = Eigen::MatrixXd::Zero(faceBases(iF).Polyk2->dimension(), dimensionFace(iF));

  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    std::vector<Eigen::Vector3d> basisE = E.edge_normalbasis();
    size_t dim_Pk_E = edgeBases(E.global_index()).Polyk->dimension();
    QuadratureRule quad_2k_E = generate_quadrature_rule(E, 2 * degree());
    auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk, quad_2k_E);
    BCF_vec.block(0, localOffset(F, E) + dim_Pk_E, faceBases(iF).Polyk2->dimension(), dim_Pk_E) // v_E nE1 . nF * rF . tE
      += F.edge_orientation(iE) * scalar_product(basisE[0],nF) * compute_gram_matrix(
						      scalar_product(evaluate_quad<Function>::compute(*faceBases(iF).Polyk2, quad_2k_E),E.tangent()),
						      basis_Pk_E_quad,
						      quad_2k_E
						      );
    BCF_vec.block(0, localOffset(F, E) + 2*dim_Pk_E, faceBases(iF).Polyk2->dimension(), dim_Pk_E) // v_E nE2 . nF * rF . tE
      += F.edge_orientation(iE) * scalar_product(basisE[1],nF) * compute_gram_matrix(
						      scalar_product(evaluate_quad<Function>::compute(*faceBases(iF).Polyk2, quad_2k_E),E.tangent()),
						      basis_Pk_E_quad,
						      quad_2k_E
						      );
  } // for iE

  if (degree() > 0) {
    size_t offset_F = faceBases(iF).Rolykmo->dimension() + faceBases(iF).RolyComplk->dimension();
    RotBasis<StokesCore::Poly2BasisFaceType> rot_Pk2F(*faceBases(iF).Polyk2);
    BCF_vec.block(0, localOffset(F) + offset_F, faceBases(iF).Polyk2->dimension(), faceBases(iF).Polykmo->dimension())
      += compute_gram_matrix(evaluate_quad<Function>::compute(rot_Pk2F, quad_2kpo_F),
        evaluate_quad<Function>::compute(*faceBases(iF).Polykmo,quad_2kpo_F),quad_2kpo_F);
      //+= GramMatrix(F, rot_Pk2F, *faceBases(iF).Polykmo, int_monoF_2kpo);
  } // if degree() > 0
 
  Eigen::MatrixXd CF_vec = MCF_vec.ldlt().solve(BCF_vec);
 
  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  auto basis_Pkpo0_F = ShiftedBasis<typename StokesCore::PolyBasisFaceType>(*faceBases(iF).Polykpo, 1);

  Eigen::MatrixXd MPF
    = Eigen::MatrixXd::Zero(faceBases(iF).Polyk2->dimension(),
			    faceBases(iF).Polyk2->dimension());
  Eigen::MatrixXd BPF
    = Eigen::MatrixXd::Zero(faceBases(iF).Polyk2->dimension(), dimensionFace(iF));

  CurlBasis<decltype(basis_Pkpo0_F)> rot_Pkp0_F(basis_Pkpo0_F);
  MPF.topLeftCorner(basis_Pkpo0_F.dimension(), faceBases(iF).Polyk2->dimension())
    = GramMatrix(F, rot_Pkp0_F, *faceBases(iF).Polyk2, int_monoF_2kpo);

  if (degree() > 0) {
    MPF.bottomLeftCorner(faceBases(iF).RolyComplk->dimension(), faceBases(iF).Polyk2->dimension())
      = GramMatrix(F, *faceBases(iF).RolyComplk, *faceBases(iF).Polyk2, int_monoF_2kpo);
    //BPF.bottomRightCorner(faceBases(iF).RolyComplk->dimension(), faceBases(iF).RolyComplk->dimension())
    BPF.block(basis_Pkpo0_F.dimension(),localOffset(F)+PolynomialSpaceDimension<Face>::Roly(degree()-1),
              faceBases(iF).RolyComplk->dimension(), faceBases(iF).RolyComplk->dimension())
      += GramMatrix(F, *faceBases(iF).RolyComplk, int_monoF_2kpo);    
  } // if degree() > 0
 
  BPF.topLeftCorner(basis_Pkpo0_F.dimension(), dimensionFace(iF))
    += GramMatrix(F, basis_Pkpo0_F, *faceBases(iF).Polyk, int_monoF_2kpo) * CF;
 
  for (size_t iE = 0; iE < F.n_edges(); iE++) {
    const Edge & E = *F.edge(iE);
    //QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * degree() + 1);
    //BPF.block(0, localOffset(F, E), basis_Pkpo0_F.dimension(), edgeBases(E.global_index()).Polyk->dimension())
    //  += F.edge_orientation(iE) * compute_gram_matrix(
		//				      evaluate_quad<Function>::compute(basis_Pkpo0_F, quad_2kpo_E),
		//				      evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk, quad_2kpo_E),
		//				      quad_2kpo_E
		//				      );    
    size_t dim_Pkp2_E = PolynomialSpaceDimension<Edge>::Poly(degree()+2);
    Eigen::Vector3d tE = E.tangent();
    Eigen::MatrixXd xyz2tE = Eigen::MatrixXd::Zero(dim_Pkp2_E,3*dim_Pkp2_E);
    for (size_t i = 0; i < 3; i++) {
      xyz2tE.block(0,i*dim_Pkp2_E,dim_Pkp2_E,dim_Pkp2_E) = tE(i)*Eigen::MatrixXd::Identity(dim_Pkp2_E,dim_Pkp2_E);
    }
    QuadratureRule quad_2kp3_E = generate_quadrature_rule(E, 2 * degree() + 3);

    BPF.middleRows(0,basis_Pkpo0_F.dimension())
      += F.edge_orientation(iE) * compute_gram_matrix(
						      evaluate_quad<Function>::compute(basis_Pkpo0_F, quad_2kp3_E),
						      evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polykp2, quad_2kp3_E),
						      quad_2kp3_E
						      ) * xyz2tE * extendOperator(F,E, edgeOperators(E.global_index()).potential);
  } // for iE

  return LocalOperators(CF, CF_vec, MPF.partialPivLu().solve(BPF));
}

//------------------------------------------------------------------------------

XCurlStokes::LocalOperators XCurlStokes::_compute_cell_curl_potential(size_t iT)
{
  const Cell & T = *mesh().cell(iT);

  //------------------------------------------------------------------------------
  // Curl
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  // Compute all integrals of monomial powers to degree 2k and the mass matrix
  MonomialCellIntegralsType int_mono_2kpo = IntegrateCellMonomials(T, 2*degree()+1);
  Eigen::MatrixXd gram_Pk3_T = GramMatrix(T, *cellBases(iT).Polyk3, int_mono_2kpo);
  Eigen::LDLT<Eigen::MatrixXd> ldlt_gram_Pk3_T(gram_Pk3_T);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BCT
    = Eigen::MatrixXd::Zero(cellBases(iT).Polyk3->dimension(), dimensionCell(iT));

  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    Eigen::Vector3d nF = F.normal();

    MonomialScalarBasisFace basis_Pk_F(F, degree());
    DecomposePoly dec(F, TangentFamily<MonomialScalarBasisFace>(basis_Pk_F, basis_Pk_F.coordinates_system()));
    boost::multi_array<VectorRd, 2> Pk3_T_cross_nF_nodes 
          = vector_product(evaluate_quad<Function>::compute(*cellBases(iT).Polyk3, dec.get_nodes()), nF);
    auto Pk3_T_cross_nF_family = dec.family(Pk3_T_cross_nF_nodes);
    Eigen::MatrixXd PF = extendOperator(T, F, m_face_operators[F.global_index()]->potential);
    MonomialFaceIntegralsType int_mono_2k_F = IntegrateFaceMonomials(F, 2*degree());
    BCT += T.face_orientation(iF) * GramMatrix(F, Pk3_T_cross_nF_family, *faceBases(F).Polyk2, int_mono_2k_F) * PF;

    // The following commented block replaces the previous one, without using DecomposePoly
    /*
      QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * degree());
      Eigen::MatrixXd PF = extendOperator(T, F, m_face_operators[F.global_index()]->potential);
      BCT += T.face_orientation(iF) * compute_gram_matrix(
					    vector_product(evaluate_quad<Function>::compute(*cellBases(iT).Polyk3, quad_2k_F), nF),
			        evaluate_quad<Function>::compute(*faceBases(F).Polyk2, quad_2k_F),
					    quad_2k_F
					    ) * PF;
    */
    
  } // for iF

  if (degree() > 0) {
    CurlBasis<StokesCore::Poly3BasisCellType> curl_Pk3_basis(*cellBases(iT).Polyk3);
    BCT.block(0, localOffset(T), cellBases(iT).Polyk3->dimension(), cellBases(iT).Rolykmo->dimension())
      += GramMatrix(T, curl_Pk3_basis, *cellBases(iT).Rolykmo, int_mono_2kpo);
  } // if degree() > 0 

  Eigen::MatrixXd CT = ldlt_gram_Pk3_T.solve(BCT);
  
  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------
  
  Eigen::MatrixXd MPT
    = Eigen::MatrixXd::Zero(cellBases(iT).Polyk3->dimension(),
			    cellBases(iT).Polyk3->dimension());  
  
  Eigen::MatrixXd BPT
    = Eigen::MatrixXd::Zero(cellBases(iT).Polyk3->dimension(), dimensionCell(iT));

  CurlBasis<StokesCore::GolyComplBasisCellType> Rolyk_basis(*cellBases(iT).GolyComplkpo);
  MPT.topRows(cellBases(iT).GolyComplkpo->dimension()) = GramMatrix(T, Rolyk_basis, *cellBases(iT).Polyk3, int_mono_2kpo);
  
  if (degree() > 0) {
    MPT.bottomRows(cellBases(iT).RolyComplk->dimension()) = GramMatrix(T, *cellBases(iT).RolyComplk, *cellBases(iT).Polyk3, int_mono_2kpo);
    BPT.bottomRightCorner(cellBases(iT).RolyComplk->dimension(), cellBases(iT).RolyComplk->dimension())
      = GramMatrix(T, *cellBases(iT).RolyComplk, int_mono_2kpo);
  } // if degree() > 0

  
  BPT.topRows(cellBases(iT).GolyComplkpo->dimension())
    += GramMatrix(T, *cellBases(iT).GolyComplkpo, *cellBases(iT).Polyk3, int_mono_2kpo) * CT;
    
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    Eigen::Vector3d nF = F.normal();
    Eigen::MatrixXd PF = extendOperator(T, F, m_face_operators[F.global_index()]->potential);
  
    MonomialScalarBasisFace basis_Pkpo_F(F, degree()+1);
    DecomposePoly dec(F, TangentFamily<MonomialScalarBasisFace>(basis_Pkpo_F, basis_Pkpo_F.coordinates_system()));
    boost::multi_array<VectorRd, 2> Gkpo_T_cross_nF_nodes 
            = vector_product(evaluate_quad<Function>::compute(*cellBases(iT).GolyComplkpo, dec.get_nodes()), nF);
    auto Gkpo_T_cross_nF_family = dec.family(Gkpo_T_cross_nF_nodes);
    MonomialFaceIntegralsType int_mono_2kpo_F = IntegrateFaceMonomials(F, 2*degree()+1);
    BPT.topRows(cellBases(iT).GolyComplkpo->dimension())
      -= T.face_orientation(iF) * GramMatrix(F, Gkpo_T_cross_nF_family, *faceBases(F).Polyk2, int_mono_2kpo_F) * PF;

    // The following commented block replaces the previous one, without DecomposePoly
    /*
      QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2 * degree() + 1);
      BPT.topRows(cellBases(iT).GolyComplkpo->dimension())
        -= T.face_orientation(iF) * compute_gram_matrix(
						      vector_product(evaluate_quad<Function>::compute(*cellBases(iT).GolyComplkpo, quad_2kpo_F), nF),
						      evaluate_quad<Function>::compute(*faceBases(F).Polyk2, quad_2kpo_F),
						      quad_2kpo_F
						      ) * PF;
    */

  } // for iF
  
  return LocalOperators(CT, Eigen::MatrixXd(), MPT.partialPivLu().solve(BPT));
}

//------------------------------------------------------------------------------
//        Functions to compute matrices for local L2 products on Xcurl
//------------------------------------------------------------------------------

Eigen::MatrixXd XCurlStokes::computeL2Product(
                                        const size_t iT,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & mass_Pk3_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 
  
  // create the weighted mass matrix, with simple product if weight is constant
  Eigen::MatrixXd w_mass_Pk3_T;
  if (weight.deg(T)==0){
    // constant weight
    if (mass_Pk3_T.rows()==1){
      // We have to compute the mass matrix
      MonomialCellIntegralsType int_mono_2kp2 = IntegrateCellMonomials(T, 2*degree()+2);
      w_mass_Pk3_T = weight.value(T, T.center_mass()) * GramMatrix(T, *cellBases(iT).Polyk3, int_mono_2kp2);
    }else{
      w_mass_Pk3_T = weight.value(T, T.center_mass()) * mass_Pk3_T;
    }
  }else{
    // weight is not constant, we create a weighted mass matrix
    QuadratureRule quad_2kpw_T = generate_quadrature_rule(T, 2 * degree() + weight.deg(T));
    auto basis_Pk3_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk3, quad_2kpw_T);
    std::function<double(const Eigen::Vector3d &)> weight_T 
              = [&T, &weight](const Eigen::Vector3d &x)->double {
                  return weight.value(T, x);
                };
    w_mass_Pk3_T = compute_weighted_gram_matrix(weight_T, basis_Pk3_T_quad, basis_Pk3_T_quad, quad_2kpw_T, "sym");
  }

  
  // The leftOp and rightOp will come from the potentials
  std::vector<Eigen::MatrixXd> potentialOp(T.n_edges()+T.n_faces()+1);
  for (size_t iE = 0; iE < T.n_edges(); iE++){
    const Edge & E = *T.edge(iE);
    potentialOp[iE] = extendOperator(T, E, m_edge_operators[E.global_index()]->potential);
  }
  for (size_t iF = 0; iF < T.n_faces(); iF++){
    const Face & F = *T.face(iF);
    potentialOp[T.n_edges()+iF] = extendOperator(T, F, m_face_operators[F.global_index()]->potential);
  }
  potentialOp[T.n_edges()+T.n_faces()] = m_cell_operators[iT]->potential;


  return computeL2Product_with_Ops(iT, potentialOp, potentialOp, penalty_factor, w_mass_Pk3_T, weight);

}

Eigen::MatrixXd XCurlStokes::computeL2Product_with_Ops(
                                        const size_t iT,
                                        const std::vector<Eigen::MatrixXd> & leftOp,
                                        const std::vector<Eigen::MatrixXd> & rightOp,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & w_mass_Pk3_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 

  // leftOp and rightOp must list the operators acting on the DOFs, and which we want to
  // use for the L2 product. Specifically, each one lists operators (matrices) returning
  // values in edges space P^k(E), faces space P^k(F)^2 (tangent) and element space P^k(T)^3.
  // For the standard Xcurl L2 product, these will respectively be identity (for each edge),
  // gamma_tF (for each F) and PT. 
  // To compute the Xcurl L2 product applied (left or right) to the discrete gradient,
  // leftOp or rightOp must list the edge, face and element (full) gradient operators.
  // All these operators must have the same domain, so possibly being extended appropriately
  // using extendOperator from globaldofspace.

  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(leftOp[0].cols(), rightOp[0].cols());
  
  size_t offset_F = T.n_edges();
  size_t offset_T = T.n_edges() + T.n_faces();

  // Edge penalty terms
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
        
    QuadratureRule quad_2kp4_E = generate_quadrature_rule(E, 2 * degree() + 4);
    
    // weight and scaling hE^2
    double max_weight_quad_E = weight.value(T, quad_2kp4_E[0].vector());
    // If the weight is not constant, we want to take the largest along the edge
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2kp4_E.size(); iqn++) {
        max_weight_quad_E = std::max(max_weight_quad_E, weight.value(T, quad_2kp4_E[iqn].vector()));
      } // for
    }
    double w_hE2 = max_weight_quad_E * std::pow(E.measure(), 2);

    // The penalty term int_E (PT w  - w_E) * (PT v  - v_E) is computed by developping.
    auto basis_Pk3_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk3, quad_2kp4_E);
    auto basis_Pk3p2_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk3p2, quad_2kp4_E);
    Eigen::MatrixXd gram_Pk3T_Pk3p2E = compute_gram_matrix(basis_Pk3_T_quad, basis_Pk3p2_E_quad, quad_2kp4_E);
    
    // Contribution of edge E
    L2P += w_hE2 * ( leftOp[offset_T].transpose() * compute_gram_matrix(basis_Pk3_T_quad, quad_2kp4_E) * rightOp[offset_T]
                   - leftOp[offset_T].transpose() * gram_Pk3T_Pk3p2E * rightOp[iE]
                   - leftOp[iE].transpose() * gram_Pk3T_Pk3p2E.transpose() * rightOp[offset_T]
                   + leftOp[iE].transpose() * compute_gram_matrix(basis_Pk3p2_E_quad, quad_2kp4_E) * rightOp[iE]);

  } // for iE

  // Face penalty terms
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * degree());

    // Compute mass-matrices: Polyk2-Polyk2, Polyk2-Polyk3 (also serves to take the tangential component of PT)
    MonomialScalarBasisFace basis_Pk_F(F, degree());
    DecomposePoly dec(F, TangentFamily<MonomialScalarBasisFace>(basis_Pk_F, basis_Pk_F.coordinates_system()));
    const VectorRd nF = F.normal();
    // Values of Pk3T at the nodes and projected on the tangent space to F
    auto Pk3T_tangent_nodes = transform_values_quad<VectorRd>(
                                evaluate_quad<Function>::compute(*cellBases(iT).Polyk3, dec.get_nodes()),
                                [&nF](const VectorRd &z)->VectorRd { return z-(z.dot(nF))*nF;});
    auto Pk3T_tangent_family_Pk2F = dec.family(Pk3T_tangent_nodes);
    MonomialFaceIntegralsType int_monoF_2kp2 = IntegrateFaceMonomials(F, 2*degree()+2);
    Eigen::MatrixXd mass_Pk2F_Pk2F = GramMatrix(F, *faceBases(F).Polyk2, int_monoF_2kp2);
    Eigen::MatrixXd gram_Pk2F_Pk3T = GramMatrix(F, *faceBases(F).Polyk2, Pk3T_tangent_family_Pk2F, int_monoF_2kp2);

    // This commented block does the same as above, without DecomposePoly
    /*
      MonomialFaceIntegralsType int_monoF_2k = IntegrateFaceMonomials(F, 2*degree());
      Eigen::MatrixXd mass_Pk2F_Pk2F = GramMatrix(F, *faceBases(F).Polyk2, int_monoF_2kp2);
      auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).Polyk2, quad_2k_F);
      auto basis_Pk3_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk3, quad_2k_F);
      Eigen::MatrixXd gram_Pk2F_Pk3T = compute_gram_matrix(basis_Pk2_F_quad, basis_Pk3_T_quad, quad_2k_F, "nonsym");
    */
    
    // Weight coming from permeability and scaling hF
    double max_weight_quad_F = weight.value(T, quad_2k_F[0].vector());
    // If the weight is not constant, we want to take the largest along the edge
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2k_F.size(); iqn++) {
        max_weight_quad_F = std::max(max_weight_quad_F, weight.value(T, quad_2k_F[iqn].vector()));
      } // for
    }
    double w_hF = max_weight_quad_F * F.diam();

    // The penalty term int_T ((leftOp)_{tF}-leftOp_{tF}) * ((rightOp)_{tF}-rightOp_{tF}) is computed by developping,
    // and using the fact that (mass_Pk2F_Pk2F)^{-1} * gram_Pk2F_Pk3T is the matrix that represents the projection from
    // P^k(T)^3 to the tangent space P^k(F)^2
 
    // Contribution of face F
    L2P += w_hF * ( leftOp[offset_T].transpose() * gram_Pk2F_Pk3T.transpose() * mass_Pk2F_Pk2F.ldlt().solve(gram_Pk2F_Pk3T) * rightOp[offset_T]
                - leftOp[offset_T].transpose() * gram_Pk2F_Pk3T.transpose() * rightOp[offset_F+iF] 
                - leftOp[offset_F+iF].transpose() * gram_Pk2F_Pk3T * rightOp[offset_T]
                + leftOp[offset_F+iF].transpose() * mass_Pk2F_Pk2F * rightOp[offset_F+iF]                  
                  );

  } // for iF

  L2P *= penalty_factor;
  
  // Consistent (cell) term
  L2P += leftOp[offset_T].transpose() * w_mass_Pk3_T * rightOp[offset_T];
  
   
  return L2P;
}

Eigen::MatrixXd XCurlStokes::computeL2opnVertex(double h) const {
  Eigen::DiagonalMatrix<double, 6> rv;
  double h2 = h*h;
  rv.diagonal() << h2,h2,h2,1.,1.,1.;
  return rv;
}

Eigen::MatrixXd XCurlStokes::computeL2opnEdge(size_t iE) const {
  const Edge & E = *mesh().edge(iE);
  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(numLocalDofsEdge(),numLocalDofsEdge());
  // L2 product of each components
  QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2 * degree()+2);
  auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk, quad_2kp2_E);
  auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polykpo, quad_2kp2_E);
  Eigen::MatrixXd gram_Pk_E = compute_gram_matrix(basis_Pk_E_quad,quad_2kp2_E);
  Eigen::MatrixXd gram_Pkpo_E = compute_gram_matrix(basis_Pkpo_E_quad,quad_2kp2_E);
  size_t dimPk = PolynomialSpaceDimension<Edge>::Poly(degree());
  size_t dimPkpo = PolynomialSpaceDimension<Edge>::Poly(degree() +1 );
  size_t offset_E = 0;
  for (size_t i = 0; i < 3; i++) {
    L2P.block(offset_E,offset_E,dimPk,dimPk) = gram_Pk_E;
    offset_E += dimPk;
  }
  for (size_t i = 0; i < 3; i++) {
    L2P.block(offset_E,offset_E,dimPkpo,dimPkpo) = E.diam()*E.diam()*gram_Pkpo_E;
    offset_E += dimPkpo;
  }
  return L2P;
}

Eigen::MatrixXd XCurlStokes::computeL2opnuEdge(size_t iE) const {
  const Edge & E = *mesh().edge(iE);
  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(dimensionEdge(iE),dimensionEdge(iE));
  // Vertices
  L2P.block(3,3,3,3) = E.diam()*Eigen::MatrixXd::Identity(3,3);
  L2P.block(9,9,3,3) = E.diam()*Eigen::MatrixXd::Identity(3,3);
  L2P.block(0,0,3,3) = E.diam()*E.diam()*E.diam()*Eigen::MatrixXd::Identity(3,3);
  L2P.block(6,6,3,3) = E.diam()*E.diam()*E.diam()*Eigen::MatrixXd::Identity(3,3);
  // L2 product of each components
  QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2 * degree()+2);
  auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk, quad_2kp2_E);
  auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polykpo, quad_2kp2_E);
  Eigen::MatrixXd gram_Pk_E = compute_gram_matrix(basis_Pk_E_quad,quad_2kp2_E);
  Eigen::MatrixXd gram_Pkpo_E = compute_gram_matrix(basis_Pkpo_E_quad,quad_2kp2_E);
  size_t dimPk = PolynomialSpaceDimension<Edge>::Poly(degree());
  size_t dimPkpo = PolynomialSpaceDimension<Edge>::Poly(degree() +1 );
  size_t offset_E = 12;
  for (size_t i = 0; i < 3; i++) {
    L2P.block(offset_E,offset_E,dimPk,dimPk) = gram_Pk_E;
    offset_E += dimPk;
  }
  for (size_t i = 0; i < 3; i++) {
    L2P.block(offset_E,offset_E,dimPkpo,dimPkpo) = E.diam()*E.diam()*gram_Pkpo_E;
    offset_E += dimPkpo;
  }
  return L2P;
}

Eigen::MatrixXd XCurlStokes::computeL2opnFace(size_t iF) const {
  if (degree() == 0) {
    const Face & F = *mesh().face(iF);
    QuadratureRule quad_2k_F = generate_quadrature_rule(F, 1);
    auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).Golyk, quad_2k_F);
    Eigen::MatrixXd L2P = compute_gram_matrix(basis_Gk_F_quad, quad_2k_F);

    return L2P; 
  } else {
    const Face & F = *mesh().face(iF);
    QuadratureRule quad_2k_F = generate_quadrature_rule(F, 2 * degree());
    auto basis_Rkmo_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).Rolykmo, quad_2k_F);
    auto basis_Rck_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).RolyComplk, quad_2k_F);
    auto basis_Pkmo_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).Polykmo, quad_2k_F);
    auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).Golyk, quad_2k_F);
    auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(F.global_index()).GolyComplk, quad_2k_F);
    Eigen::MatrixXd gram_Rkmo_F = compute_gram_matrix(basis_Rkmo_F_quad, quad_2k_F);
    Eigen::MatrixXd gram_Rck_F = compute_gram_matrix(basis_Rck_F_quad, quad_2k_F);
    Eigen::MatrixXd gram_Pkmo_F = compute_gram_matrix(basis_Pkmo_F_quad, quad_2k_F);
    Eigen::MatrixXd gram_Gk_F = compute_gram_matrix(basis_Gk_F_quad, quad_2k_F);
    Eigen::MatrixXd gram_Gck_F = compute_gram_matrix(basis_Gck_F_quad, quad_2k_F);
    size_t dimRkmo = PolynomialSpaceDimension<Face>::Roly(degree()-1);
    size_t dimRck = PolynomialSpaceDimension<Face>::RolyCompl(degree());
    size_t dimPkmo = PolynomialSpaceDimension<Face>::Poly(degree()-1);
    size_t dimGk = PolynomialSpaceDimension<Face>::Goly(degree());
    size_t dimGck = PolynomialSpaceDimension<Face>::GolyCompl(degree());

    Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(numLocalDofsFace(),numLocalDofsFace());
    size_t offset = 0;
    L2P.block(offset,offset,dimRkmo,dimRkmo) = gram_Rkmo_F;
    offset += dimRkmo;
    L2P.block(offset,offset,dimRck,dimRck) = gram_Rck_F;
    offset += dimRck;
    L2P.block(offset,offset,dimPkmo,dimPkmo) = gram_Pkmo_F;
    offset += dimPkmo;
    L2P.block(offset,offset,dimGk,dimGk) = F.diam()*F.diam()*gram_Gk_F;
    offset += dimGk;
    L2P.block(offset,offset,dimGck,dimGck) = F.diam()*F.diam()*gram_Gck_F;
    
    return L2P;
  }
}

Eigen::MatrixXd XCurlStokes::computeL2opnCell(size_t iT) const {
  if (degree() == 0) return Eigen::MatrixXd();
  const Cell & T = *mesh().cell(iT);
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * degree());
  auto basis_Rkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(T.global_index()).Rolykmo, quad_2k_T);
  auto basis_Rck_T_quad = evaluate_quad<Function>::compute(*cellBases(T.global_index()).RolyComplk, quad_2k_T);
  Eigen::MatrixXd gram_Rkmo_T = compute_gram_matrix(basis_Rkmo_T_quad, quad_2k_T);
  Eigen::MatrixXd gram_Rck_T = compute_gram_matrix(basis_Rck_T_quad, quad_2k_T);
  size_t dimRkmo = PolynomialSpaceDimension<Cell>::Roly(degree()-1);
  size_t dimRck = PolynomialSpaceDimension<Cell>::RolyCompl(degree());

  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(numLocalDofsCell(),numLocalDofsCell());
  size_t offset = 0;
  L2P.block(offset,offset,dimRkmo,dimRkmo) = gram_Rkmo_T;
  offset += dimRkmo;
  L2P.block(offset,offset,dimRck,dimRck) = gram_Rck_T;

  return L2P;
}

Eigen::MatrixXd XCurlStokes::buildCurlComponentsCell(size_t iT) const {
  const Cell & T = *mesh().cell(iT);

  size_t dim_xnabla_T = T.n_vertices() * 3 
                      + T.n_edges() * 3 * PolynomialSpaceDimension<Edge>::Poly(degree()+1)
                      + T.n_faces() * (PolynomialSpaceDimension<Face>::Poly(degree())
                            + PolynomialSpaceDimension<Face>::Goly(degree()) + PolynomialSpaceDimension<Face>::GolyCompl(degree()))
                      + PolynomialSpaceDimension<Cell>::Goly(degree()-1) + PolynomialSpaceDimension<Cell>::GolyCompl(degree());
  size_t dim_xcurl_T = dimensionCell(T);

  Eigen::MatrixXd uCT = Eigen::MatrixXd::Zero(dim_xnabla_T,dim_xcurl_T);

  size_t offset = 0;
  // Vertex components
  for (size_t iV = 0; iV < T.n_vertices(); iV++) {
    uCT.block(offset,6*iV,3,3) = Eigen::MatrixXd::Identity(3,3);
    offset += 3;
  }
  // Edge components
  size_t dimEXnabla = 3*PolynomialSpaceDimension<Edge>::Poly(degree()+1);
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    uCT.middleRows(offset, dimEXnabla)
      = extendOperator(T, E, edgeOperators(E).curl).bottomRows(dimEXnabla); // discard vertices
    offset += dimEXnabla;
  } // for iE
  // Face components
  for (size_t iF = 0; iF < T.n_faces(); iF++) {
    const Face & F = *T.face(iF);
    size_t dim_Pk_F = PolynomialSpaceDimension<Face>::Poly(degree());
    uCT.middleRows(offset,dim_Pk_F) = extendOperator(T, F, faceOperators(F).curl);
    offset += dim_Pk_F;
    // Gk, Gck components
    Eigen::MatrixXd CF_vec = extendOperator(T, F, faceOperators(F).curl_vec);
    MonomialFaceIntegralsType int_monoF_2kpo = IntegrateFaceMonomials(F, 2*degree()+1);
    Eigen::MatrixXd mass_Gk_F = GramMatrix(F, *faceBases(F).Golyk, int_monoF_2kpo);

    QuadratureRule quad_2kpo_F = generate_quadrature_rule(F, 2*(degree()+1));
    auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Polyk2, quad_2kpo_F);
    auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*faceBases(F).Golyk, quad_2kpo_F);
    Eigen::MatrixXd tmp = compute_gram_matrix(basis_Gk_F_quad,basis_Pk2_F_quad, quad_2kpo_F) * CF_vec;
    Eigen::MatrixXd pi_Gk_GFp_F = mass_Gk_F.ldlt().solve(compute_gram_matrix(basis_Gk_F_quad,basis_Pk2_F_quad, quad_2kpo_F) * CF_vec);
    
    // TODO implement this specialization
    //Eigen::MatrixXd pi_Gk_GFp_F = mass_Gk_F.ldlt().solve(GramMatrix(F, *faceBases(F).Golyk, *faceBases(F).Polyk2, int_monoF_2kpo) * GFperp);

    // TODO implement this specialization
    //Eigen::MatrixXd pi_Gck_GFp_F = mass_Gck_F.ldlt().solve(GramMatrix(F, *faceBases(F).GolyComplk, *faceBases(F).Polyk2, int_monoF_2kpo) * GFperp);

    size_t offset_FGk = PolynomialSpaceDimension<Face>::Roly(degree()-1) + PolynomialSpaceDimension<Face>::RolyCompl(degree())
                      + PolynomialSpaceDimension<Face>::Poly(degree()-1);
    size_t dim_Gk_F = PolynomialSpaceDimension<Face>::Goly(degree());
    size_t dim_Gck_F = PolynomialSpaceDimension<Face>::GolyCompl(degree());

    uCT.block(offset, 0,dim_Gk_F, dim_xcurl_T) = pi_Gk_GFp_F;
    // Add R_{v,G,F}
    uCT.block(offset,localOffset(T,F)+offset_FGk,dim_Gk_F,dim_Gk_F) = Eigen::MatrixXd::Identity(dim_Gk_F,dim_Gk_F);
    offset += dim_Gk_F;
    if (degree() > 0) {
      Eigen::MatrixXd mass_Gck_F = GramMatrix(F, *faceBases(F).GolyComplk, int_monoF_2kpo);
      auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*faceBases(F).GolyComplk, quad_2kpo_F);
      Eigen::MatrixXd pi_Gck_GFp_F = mass_Gck_F.ldlt().solve(compute_gram_matrix(basis_Gck_F_quad,basis_Pk2_F_quad, quad_2kpo_F) * CF_vec);

      uCT.block(offset, 0, dim_Gck_F, dim_xcurl_T) = pi_Gck_GFp_F;
      // Add R_{v,G,F}^c
      uCT.block(offset,localOffset(T,F)+offset_FGk+dim_Gk_F,dim_Gck_F,dim_Gck_F) = Eigen::MatrixXd::Identity(dim_Gck_F,dim_Gck_F);
      offset += dim_Gck_F;
    }

  } // for iF

  if (m_stokes_core.degree() > 0) {
    // Cell component
    MonomialCellIntegralsType int_mono_2k = IntegrateCellMonomials(T, 2*degree());
    Eigen::MatrixXd mass_Gkmo_T = GramMatrix(T, *m_stokes_core.cellBases(iT).Golykmo, int_mono_2k);
    Eigen::MatrixXd mass_Gck_T = GramMatrix(T, *m_stokes_core.cellBases(iT).GolyComplk, int_mono_2k);

    Eigen::MatrixXd pi_Gkmo_CT_T = mass_Gkmo_T.ldlt().solve(GramMatrix(T, *m_stokes_core.cellBases(iT).Golykmo, *m_stokes_core.cellBases(iT).Polyk3, int_mono_2k) * cellOperators(iT).curl);
    Eigen::MatrixXd pi_Gck_CT_T = mass_Gck_T.ldlt().solve(
                                      GramMatrix(T, *m_stokes_core.cellBases(iT).GolyComplk, *m_stokes_core.cellBases(iT).Polyk3, int_mono_2k)
                                      * cellOperators(iT).curl
                                  );

    uCT.block(offset, 0, PolynomialSpaceDimension<Cell>::Goly(degree()-1), dim_xcurl_T) = pi_Gkmo_CT_T;
    offset += PolynomialSpaceDimension<Cell>::Goly(degree()-1);
    uCT.block(offset, 0, PolynomialSpaceDimension<Cell>::GolyCompl(degree()), dim_xcurl_T) = pi_Gck_CT_T;
    offset += PolynomialSpaceDimension<Cell>::GolyCompl(degree());
  }
  
  return uCT;
}

