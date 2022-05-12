#include <cassert>

#include "stokescore.hpp"
#include <parallel_for.hpp>
#include <GMpoly_cell.hpp>
#include <GMpoly_face.hpp>
#include <GMpoly_edge.hpp>

using namespace HArDCore3D;

//------------------------------------------------------------------------------

StokesCore::StokesCore(const Mesh & mesh, size_t K, bool use_threads, std::ostream & output)
  : m_mesh(mesh),
    m_K(K),
    m_output(output),
    m_cell_bases(mesh.n_cells()),
    m_face_bases(mesh.n_faces()),
    m_edge_bases(mesh.n_edges())
{
  m_output << "[StokesCore] Initializing" << std::endl;
  
  // Construct element bases
  std::function<void(size_t, size_t)> construct_all_cell_bases
    = [this](size_t start, size_t end)->void
      {
	      for (size_t iT = start; iT < end; iT++) {
	        this->m_cell_bases[iT].reset( new CellBases(this->_construct_cell_bases(iT)) );
	      } // for iT
      };

  m_output << "[StokesCore] Constructing element bases" << std::endl;
  parallel_for(mesh.n_cells(), construct_all_cell_bases, use_threads);
  
  // Construct face bases
  std::function<void(size_t, size_t)> construct_all_face_bases
    = [this](size_t start, size_t end)->void
      {
	      for (size_t iF = start; iF < end; iF++) {
	        this->m_face_bases[iF].reset( new FaceBases(_construct_face_bases(iF)) );
	      } // for iF
      };
  
  m_output << "[StokesCore] Constructing face bases" << std::endl;
  parallel_for(mesh.n_faces(), construct_all_face_bases, use_threads);

  // Construct edge bases
  std::function<void(size_t, size_t)> construct_all_edge_bases   
    = [this](size_t start, size_t end)->void
      {
	      for (size_t iE = start; iE < end; iE++) {
	        this->m_edge_bases[iE].reset( new EdgeBases(_construct_edge_bases(iE)) );
	      } // for iF
      };
  
  m_output << "[StokesCore] Constructing edge bases" << std::endl;
  parallel_for(mesh.n_edges(), construct_all_edge_bases, use_threads);
}

//------------------------------------------------------------------------------

StokesCore::CellBases StokesCore::_construct_cell_bases(size_t iT)
{
  const Cell & T = *m_mesh.cell(iT);

  CellBases bases_T;
  
  MonomialCellIntegralsType int_monoT_2kp4 = IntegrateCellMonomials(T, 2*(m_K+2));
  
  //------------------------------------------------------------------------------
  // Basis for Pk+1(T), Pk(T), Pk-1(T)
  //------------------------------------------------------------------------------
  
  MonomialScalarBasisCell basis_Pkpo_T(T, m_K + 1);
  bases_T.Polykpo.reset( new PolyBasisCellType(l2_orthonormalize(basis_Pkpo_T, GramMatrix(T, basis_Pkpo_T, int_monoT_2kp4))) );  

  MonomialScalarBasisCell basis_Pk_T(T, m_K);
  bases_T.Polyk.reset( new PolyBasisCellType(l2_orthonormalize(basis_Pk_T, GramMatrix(T, basis_Pk_T, int_monoT_2kp4))) );  

  // Check that we got the dimensions right
  assert( bases_T.Polykpo->dimension() == PolynomialSpaceDimension<Cell>::Poly(m_K + 1) );
  assert( bases_T.Polyk->dimension() == PolynomialSpaceDimension<Cell>::Poly(m_K) );

  if (PolynomialSpaceDimension<Cell>::Poly(m_K - 1) > 0) {
    MonomialScalarBasisCell basis_Pkmo_T(T, m_K-1);
    bases_T.Polykmo.reset( new PolyBasisCellType(l2_orthonormalize(basis_Pkmo_T, GramMatrix(T, basis_Pkmo_T, int_monoT_2kp4))) );  
    assert( bases_T.Polykmo->dimension() == PolynomialSpaceDimension<Cell>::Poly(m_K-1) );
  }  
  
  //------------------------------------------------------------------------------
  // Basis for Pk+1(T)^3, Pk(T)^3
  //------------------------------------------------------------------------------

  bases_T.Polyk3po.reset( new Poly3BasisCellType(*bases_T.Polykpo) );
  assert( bases_T.Polyk3po->dimension() == 3 * PolynomialSpaceDimension<Cell>::Poly(m_K+1) );

  bases_T.Polyk3.reset( new Poly3BasisCellType(*bases_T.Polyk) );
  assert( bases_T.Polyk3->dimension() == 3 * PolynomialSpaceDimension<Cell>::Poly(m_K) );
  
  //------------------------------------------------------------------------------
  // Basis for Gk-1(T)
  //------------------------------------------------------------------------------

  if (PolynomialSpaceDimension<Cell>::Goly(m_K - 1) > 0) {
    GradientBasis<ShiftedBasis<MonomialScalarBasisCell> >
      basis_Gkmo_T(ShiftedBasis<MonomialScalarBasisCell>(MonomialScalarBasisCell(T, m_K), 1));

    // Orthonormalize and store the basis
    bases_T.Golykmo.reset( new GolyBasisCellType(l2_orthonormalize(basis_Gkmo_T, GramMatrix(T, basis_Gkmo_T, int_monoT_2kp4))) );

    // Check that we got the dimension right
    assert( bases_T.Golykmo->dimension() == PolynomialSpaceDimension<Cell>::Goly(m_K - 1) );
  } // if

  //------------------------------------------------------------------------------
  // Bases for Gck(T), Gck+1(T), and Rk-1(T)
  //------------------------------------------------------------------------------
 
  // Gck+1(T) (orthonormalised)
  GolyComplBasisCell basis_Gckpo_T(T, m_K+1);
  bases_T.GolyComplkpo.reset( new GolyComplBasisCellType(l2_orthonormalize(basis_Gckpo_T, GramMatrix(T, basis_Gckpo_T, int_monoT_2kp4))) );

  // check dimension
  assert( bases_T.GolyComplkpo->dimension() == PolynomialSpaceDimension<Cell>::GolyCompl(m_K + 1) );
 

  if (PolynomialSpaceDimension<Cell>::GolyCompl(m_K) > 0) {
    // Gck(T)
    GolyComplBasisCell basis_Gck_T(T, m_K);
    bases_T.GolyComplk.reset( new GolyComplBasisCellType(l2_orthonormalize(basis_Gck_T, GramMatrix(T, basis_Gck_T, int_monoT_2kp4))) );
    assert( bases_T.GolyComplk->dimension() == PolynomialSpaceDimension<Cell>::GolyCompl(m_K) );

    // Basis for curl Gck. We do not want to restart from bases_T.GolyComplk because it is orthonormalised (so a 
    // Family of other bases); if we started from this one, after orthonormalisation, the basis of Rk-1(T) would be
    // a Family of a Family, for which any evaluation could be quite expensive. 
    CurlBasis<GolyComplBasisCell> basis_curl_Gck_T(basis_Gck_T);
    bases_T.Rolykmo.reset( new RolyBasisCellType(l2_orthonormalize(basis_curl_Gck_T, GramMatrix(T, basis_curl_Gck_T, int_monoT_2kp4))) );   
    assert( bases_T.Rolykmo->dimension() == PolynomialSpaceDimension<Cell>::Roly(m_K - 1));
  } // if


  //------------------------------------------------------------------------------
  // Basis for Rck(T) and Rck+2(T)
  //------------------------------------------------------------------------------
  // Rck+2(T) (orthonormalised)
  RolyComplBasisCell basis_Rckp2_T(T, m_K+2);
  bases_T.RolyComplkp2.reset( new RolyComplBasisCellType(l2_orthonormalize(basis_Rckp2_T, GramMatrix(T, basis_Rckp2_T, int_monoT_2kp4))) );
  assert ( bases_T.RolyComplkp2->dimension() == PolynomialSpaceDimension<Cell>::RolyCompl(m_K+2) );

  // Rck(T) (orthonormalised). 
  if (PolynomialSpaceDimension<Cell>::RolyCompl(m_K) > 0) { 
    RolyComplBasisCell basis_Rck_T(T, m_K);
    bases_T.RolyComplk.reset( new RolyComplBasisCellType(l2_orthonormalize(basis_Rck_T, GramMatrix(T, basis_Rck_T, int_monoT_2kp4))) );
    assert ( bases_T.RolyComplk->dimension() == PolynomialSpaceDimension<Cell>::RolyCompl(m_K) );
  } // if

  //------------------------------------------------------------------------------
  // Basis for RTbkpo(T)
  //------------------------------------------------------------------------------
  // Rk3(T) (orthonormalised)
  CurlBasis<GolyComplBasisCell> basis_curl_Gckpo_T(basis_Gckpo_T);
  bases_T.Roly3k.reset( new Roly3BasisCellType( RolyBasisCellType(l2_orthonormalize(basis_curl_Gckpo_T, GramMatrix(T, basis_curl_Gckpo_T, int_monoT_2kp4)))));
  assert( bases_T.Roly3k->dimension() == 3*PolynomialSpaceDimension<Cell>::Roly(m_K));

  // Rbckpo(T) (orthonormalised)
  // TODO implement monomial integration
  QuadratureRule quad_2kp4_T = generate_quadrature_rule(T,2*m_K + 4);
  if (m_K > 0) {
    RolybComplBasisCell basis_Rbckpo_T(T,m_K+1);
    //bases_T.RolybComplkpo.reset( new RolybComplBasisCellType(l2_orthonormalize(basis_Rbckpo_T, GramMatrix(T, basis_Rbckpo_T, int_monoT_2kp4))) );
    auto basis_Rbckpo_T_quad = evaluate_quad<Function>::compute(basis_Rbckpo_T, quad_2kp4_T);
    bases_T.RolybComplkpo.reset( new RolybComplBasisCellType(l2_orthonormalize(basis_Rbckpo_T, quad_2kp4_T,basis_Rbckpo_T_quad)) );
    assert(bases_T.RolybComplkpo->dimension() == PolynomialSpaceDimension<Cell>::RolybCompl(m_K+1));

    // Rbk(T) (orthonormalised)
    RolybBasisCell basis_Rbk_T(T,m_K);
    //bases_T.Rolybk.reset( new RolybBasisCellType(l2_orthonormalize(basis_Rbk_T, GramMatrix(T, basis_Rbk_T, int_monoT_2kp4))) );
    auto basis_Rbk_T_quad = evaluate_quad<Function>::compute(basis_Rbk_T, quad_2kp4_T);
    bases_T.Rolybk.reset( new RolybBasisCellType(l2_orthonormalize(basis_Rbk_T, quad_2kp4_T,basis_Rbk_T_quad)) );
    assert(bases_T.Rolybk->dimension() == PolynomialSpaceDimension<Cell>::Rolyb(m_K));
  } else { // m_K == 0
    bases_T.RolybComplkpo.reset(new RolybComplBasisCellType());
    bases_T.Rolybk.reset(new RolybBasisCellType());
  } // m_K > 0

  // RTbkpo(T)
  bases_T.RTbkpo.reset( new RTbBasisCellType(SumFamily<RolybComplBasisCellType,RolybBasisCellType>(*bases_T.RolybComplkpo,*bases_T.Rolybk),*bases_T.Roly3k) );
  assert(bases_T.RTbkpo->dimension() == PolynomialSpaceDimension<Cell>::RTb(m_K+1));

  //------------------------------------------------------------------------------
  // Basis for Rck3+2(T)
  //------------------------------------------------------------------------------

  //RolyComplBasisCell basis_Rckp2_T(T, m_K+2);
  bases_T.RolyComplk3p2.reset( new RolyCompl3BasisCellType (
    RolyComplBasisCellType(l2_orthonormalize(basis_Rckp2_T, GramMatrix(T, basis_Rckp2_T, int_monoT_2kp4)))) );
  assert ( bases_T.RolyComplk3p2->dimension() == 3*PolynomialSpaceDimension<Cell>::RolyCompl(m_K+2) );

  return bases_T;
}

//------------------------------------------------------------------------------

StokesCore::FaceBases StokesCore::_construct_face_bases(size_t iF)
{
  const Face & F = *m_mesh.face(iF);
  
  FaceBases bases_F;

  MonomialFaceIntegralsType int_monoF_2kp4 = IntegrateFaceMonomials(F, 2*(m_K+2));

  //------------------------------------------------------------------------------
  // Basis for Pk+1(F), Pk(F), Pk-1(F)
  //------------------------------------------------------------------------------
  MonomialScalarBasisFace basis_Pkpo_F(F, m_K + 1);
  bases_F.Polykpo.reset( new PolyBasisFaceType(l2_orthonormalize(basis_Pkpo_F, GramMatrix(F, basis_Pkpo_F, int_monoF_2kp4))) );
  
  MonomialScalarBasisFace basis_Pk_F(F, m_K);
  bases_F.Polyk.reset( new PolyBasisFaceType(l2_orthonormalize(basis_Pk_F, GramMatrix(F, basis_Pk_F, int_monoF_2kp4))) );

  // Check that we got the dimensions right
  assert( bases_F.Polykpo->dimension() == PolynomialSpaceDimension<Face>::Poly(m_K + 1) );
  assert( bases_F.Polyk->dimension() == PolynomialSpaceDimension<Face>::Poly(m_K) );

  if (PolynomialSpaceDimension<Face>::Poly(m_K - 1) > 0) {
    MonomialScalarBasisFace basis_Pkmo_F(F, m_K-1);
    bases_F.Polykmo.reset( new PolyBasisFaceType(l2_orthonormalize(basis_Pkmo_F, GramMatrix(F, basis_Pkmo_F, int_monoF_2kp4))) );
    assert( bases_F.Polykmo->dimension() == PolynomialSpaceDimension<Face>::Poly(m_K-1) );
  }
  
  //------------------------------------------------------------------------------
  // Basis Pk(F)^2
  //------------------------------------------------------------------------------

  // We use the system of coordinates of the basis on the face as generators of the face
  bases_F.Polyk2.reset( new Poly2BasisFaceType(*bases_F.Polyk, basis_Pk_F.coordinates_system()) );
  // Check dimension
  assert( bases_F.Polyk2->dimension() == 2 * PolynomialSpaceDimension<Face>::Poly(m_K) );
  
  
  //------------------------------------------------------------------------------
  // Basis for Rk-1(F)
  //------------------------------------------------------------------------------

  if (PolynomialSpaceDimension<Face>::Roly(m_K - 1) > 0) {
    // Non-orthonormalised basis of Rk-1(F). 
    MonomialScalarBasisFace basis_Pk_F(F, m_K);
    ShiftedBasis<MonomialScalarBasisFace> basis_Pk0_F(basis_Pk_F,1);
    CurlBasis<ShiftedBasis<MonomialScalarBasisFace>> basis_Rkmo_F(basis_Pk0_F);
    // Orthonormalise, store and check dimension
    bases_F.Rolykmo.reset( new RolyBasisFaceType(l2_orthonormalize(basis_Rkmo_F, GramMatrix(F, basis_Rkmo_F, int_monoF_2kp4))) );
    assert( bases_F.Rolykmo->dimension() == PolynomialSpaceDimension<Face>::Roly(m_K - 1) );
  }
  
  //------------------------------------------------------------------------------
  // Basis for Rck(F)
  //------------------------------------------------------------------------------

  if (PolynomialSpaceDimension<Face>::RolyCompl(m_K) > 0) {
    RolyComplBasisFace basis_Rck_F(F, m_K);
    bases_F.RolyComplk.reset( new RolyComplBasisFaceType(l2_orthonormalize(basis_Rck_F, GramMatrix(F, basis_Rck_F, int_monoF_2kp4))) );
    assert ( bases_F.RolyComplk->dimension() == PolynomialSpaceDimension<Face>::RolyCompl(m_K) );
  }

  //------------------------------------------------------------------------------
  // Basis for Rck+2(F)
  //------------------------------------------------------------------------------

  RolyComplBasisFace basis_Rckp2_F(F, m_K+2);
  bases_F.RolyComplkp2.reset( new RolyComplBasisFaceType(l2_orthonormalize(basis_Rckp2_F, GramMatrix(F, basis_Rckp2_F, int_monoF_2kp4))) );
  assert ( bases_F.RolyComplkp2->dimension() == PolynomialSpaceDimension<Face>::RolyCompl(m_K+2) );
  
  //------------------------------------------------------------------------------
  // Basis for Gk(F)
  //------------------------------------------------------------------------------

  if (PolynomialSpaceDimension<Face>::Goly(m_K) > 0) {
    // Non-orthonormalised basis of Gk(F). 
    MonomialScalarBasisFace basis_Pkpo_F(F, m_K+1);
    ShiftedBasis<MonomialScalarBasisFace> basis_Pkpo0_F(basis_Pkpo_F,1);
    GradientBasis<ShiftedBasis<MonomialScalarBasisFace>> basis_Gk_F(basis_Pkpo0_F);
    // Orthonormalise, store and check dimension
    bases_F.Golyk.reset( new GolyBasisFaceType(l2_orthonormalize(basis_Gk_F, GramMatrix(F, basis_Gk_F, int_monoF_2kp4))) );
    assert( bases_F.Golyk->dimension() == PolynomialSpaceDimension<Face>::Goly(m_K) );
  }
  
  //------------------------------------------------------------------------------
  // Basis for Gck(F)
  //------------------------------------------------------------------------------

  if (PolynomialSpaceDimension<Face>::GolyCompl(m_K) > 0) {
    GolyComplBasisFace basis_Gck_F(F, m_K);
    bases_F.GolyComplk.reset( new GolyComplBasisFaceType(l2_orthonormalize(basis_Gck_F, GramMatrix(F, basis_Gck_F, int_monoF_2kp4))) );
    assert ( bases_F.GolyComplk->dimension() == PolynomialSpaceDimension<Face>::GolyCompl(m_K) );
  }
  
  //------------------------------------------------------------------------------
  // Basis for RTbkpo(F)
  //------------------------------------------------------------------------------
  // bPkpo(F)
  bases_F.bPolykpo.reset( new bPolyBasisFaceType(*bases_F.Polykpo,F) );
  assert(bases_F.bPolykpo->dimension() == 2*PolynomialSpaceDimension<Face>::Poly(m_K+1));

  // Rk2(F) (orthonormalised)
    MonomialScalarBasisFace basis_Pkp2_F(F, m_K+2);
    ShiftedBasis<MonomialScalarBasisFace> basis_Pk0p2_F(basis_Pkp2_F,1);
    CurlBasis<ShiftedBasis<MonomialScalarBasisFace>> basis_Rkpo_F(basis_Pk0p2_F);
  bases_F.Rolyk2po.reset( new Roly2BasisFaceType( RolyBasisFaceType(l2_orthonormalize(basis_Rkpo_F, GramMatrix(F, basis_Rkpo_F, int_monoF_2kp4))),F) );
  assert( bases_F.Rolyk2po->dimension() == 2*PolynomialSpaceDimension<Face>::Roly(m_K+1));

  // Rbckpo(F) (orthonormalised)
  QuadratureRule quad_2kp4_F = generate_quadrature_rule(F,2*m_K + 4);
  if (m_K > 0) {
    RolybComplBasisFace basis_Rbckpo_F(F,m_K+1);
    // TODO implement monomial integration
    //bases_F.RolybComplkpo.reset( new RolybComplBasisFaceType(l2_orthonormalize(basis_Rbckpo_F, GramMatrix(F, basis_Rbckpo_F, int_monoF_2kp4))) );
    auto basis_Rbckpo_F_quad = evaluate_quad<Function>::compute(basis_Rbckpo_F,quad_2kp4_F);
    bases_F.RolybComplkpo.reset( new RolybComplBasisFaceType(l2_orthonormalize(basis_Rbckpo_F, quad_2kp4_F,basis_Rbckpo_F_quad)) );
    assert(bases_F.RolybComplkpo->dimension() == PolynomialSpaceDimension<Face>::RolybCompl(m_K+1));
  } else { // m_K == 0
    bases_F.RolybComplkpo.reset( new RolybComplBasisFaceType());
  } // m_K > 0

  // Rbkpo(F) (orthonormalised)
  RolybBasisFace basis_Rbkpo_F(F,m_K+1);
  //bases_F.Rolybkpo.reset( new RolybBasisFaceType(l2_orthonormalize(basis_Rbkpo_F, GramMatrix(F, basis_Rbkpo_F, int_monoF_2kp4))) );
  auto basis_Rbkpo_F_quad = evaluate_quad<Function>::compute(basis_Rbkpo_F,quad_2kp4_F);
  bases_F.Rolybkpo.reset( new RolybBasisFaceType(l2_orthonormalize(basis_Rbkpo_F, quad_2kp4_F,basis_Rbkpo_F_quad)) );
  assert(bases_F.Rolybkpo->dimension() == PolynomialSpaceDimension<Face>::Rolyb(m_K+1));

  // tildePkpo(F)
  bases_F.tildePolykpo.reset( new tildePolyBasisFaceType(
        *bases_F.bPolykpo,
        SumFamily<SumFamily<RolybComplBasisFaceType,RolybBasisFaceType>,Roly2BasisFaceType>(
          SumFamily<RolybComplBasisFaceType,RolybBasisFaceType>(*bases_F.RolybComplkpo,*bases_F.Rolybkpo),
          *bases_F.Rolyk2po)) );
  assert(bases_F.tildePolykpo->dimension() == PolynomialSpaceDimension<Face>::tildePoly(m_K+1));

  //------------------------------------------------------------------------------
  // Basis for Pk3+2(F)
  //------------------------------------------------------------------------------
  //MonomialScalarBasisFace basis_Pkp2_F(F, m_K + 2);
  bases_F.Polyk3p2.reset( new Poly3BasisFaceType(
    PolyBasisFaceType(l2_orthonormalize(basis_Pkp2_F, GramMatrix(F, basis_Pkp2_F, int_monoF_2kp4)))) );
  assert( bases_F.Polyk3p2->dimension() == 3*PolynomialSpaceDimension<Face>::Poly(m_K + 2) );

  //------------------------------------------------------------------------------
  // Basis for Rck3+3(F)
  //------------------------------------------------------------------------------

  MonomialFaceIntegralsType int_monoF_2kp6 = IntegrateFaceMonomials(F, 2*(m_K+3));

  RolyComplBasisFace basis_Rckp3_F(F, m_K+3);
  bases_F.RolyComplk3p3.reset( new RolyCompl3BasisFaceType (
    RolyComplBasisFaceType(l2_orthonormalize(basis_Rckp3_F, GramMatrix(F, basis_Rckp3_F, int_monoF_2kp6)))) );
  assert ( bases_F.RolyComplk3p3->dimension() == 3*PolynomialSpaceDimension<Face>::RolyCompl(m_K+3) );
  
  return bases_F;
}

//------------------------------------------------------------------------------

StokesCore::EdgeBases StokesCore::_construct_edge_bases(size_t iE)
{
  const Edge & E = *m_mesh.edge(iE);

  EdgeBases bases_E;
  
  MonomialEdgeIntegralsType int_monoE_2kp6 = IntegrateEdgeMonomials(E, 2*(degree()+3));

  // Basis for Pk+3(E)
  MonomialScalarBasisEdge basis_Pkp3_E(E, m_K + 3);
  bases_E.Polykp3.reset( new PolyBasisEdgeType(l2_orthonormalize(basis_Pkp3_E, GramMatrix(E, basis_Pkp3_E, int_monoE_2kp6))) );

  // Basis for Pk+2(E)
  MonomialScalarBasisEdge basis_Pkp2_E(E, m_K + 2);
  bases_E.Polykp2.reset( new PolyBasisEdgeType(l2_orthonormalize(basis_Pkp2_E, GramMatrix(E, basis_Pkp2_E, int_monoE_2kp6))) );

  // Basis for Pk+1(E)
  MonomialScalarBasisEdge basis_Pkpo_E(E, m_K + 1);
  bases_E.Polykpo.reset( new PolyBasisEdgeType(l2_orthonormalize(basis_Pkpo_E, GramMatrix(E, basis_Pkpo_E, int_monoE_2kp6))) );

  // Basis for Pk(E)
  MonomialScalarBasisEdge basis_Pk_E(E, m_K);
  bases_E.Polyk.reset( new PolyBasisEdgeType(l2_orthonormalize(basis_Pk_E, GramMatrix(E, basis_Pk_E, int_monoE_2kp6))) );

  // Basis for Pk-1(E)
  if (PolynomialSpaceDimension<Edge>::Poly(m_K - 1) > 0) {
    MonomialScalarBasisEdge basis_Pkmo_E(E, m_K - 1);
    bases_E.Polykmo.reset( new PolyBasisEdgeType(l2_orthonormalize(basis_Pkmo_E, GramMatrix(E, basis_Pkmo_E, int_monoE_2kp6))) );
  }

  //------------------------------------------------------------------------------
  // Basis for Pkp3(E)^3, Pkp2(E)^3, Pkpo(E)^3
  //------------------------------------------------------------------------------

  bases_E.Polyk3p3.reset( new Poly3BasisEdgeType(*bases_E.Polykp3) );
  assert( bases_E.Polyk3p3->dimension() == 3 * PolynomialSpaceDimension<Edge>::Poly(m_K+3) );

  bases_E.Polyk3p2.reset( new Poly3BasisEdgeType(*bases_E.Polykp2) );
  assert( bases_E.Polyk3p2->dimension() == 3 * PolynomialSpaceDimension<Edge>::Poly(m_K+2) );

  bases_E.Polyk3po.reset( new Poly3BasisEdgeType(*bases_E.Polykpo) );
  assert( bases_E.Polyk3po->dimension() == 3 * PolynomialSpaceDimension<Edge>::Poly(m_K+1) );

  return bases_E;
}
