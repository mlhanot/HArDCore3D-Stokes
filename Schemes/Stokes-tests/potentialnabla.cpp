// Tests for the nabla operator.
// It computes otherwise useless quantities and is not optimized in any way.
// Warning : Ttrig requieres an extremely high degree for the interpolation.

#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xcurlstokes.hpp>
#include <xnablastokes.hpp>
#include <xvlstokes.hpp>
#include "testfunction.hpp"

#include <parallel_for.hpp>

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;

const std::string mesh_file = "./meshes/" "Voro-small-2/RF_fmt/voro.2";

constexpr bool use_threads = true;
// Foward declare
template<typename T> double TestCDGE(const XNablaStokes &xnabla,const XVLStokes &xvl,T &v,bool = true);
template<typename T> double TestContI(const XNablaStokes &xnabla,T &v,bool = true);
template<typename T> double TestCDGF(const XNablaStokes &xnabla,const XVLStokes &xvl,T &v,bool = true);
template<typename T> double TestCDGT(const XNablaStokes &xnabla,const XVLStokes &xvl,T &v,bool = true);
template<typename T> double TestCDDT(const XNablaStokes &xnabla,const XSLStokes &xsl,T &v,bool = true);
template<typename T> double TestGammaI(const XNablaStokes &xnabla, T &v,bool = true);
template<typename T> double TestPotentialI(const XNablaStokes &xnabla, T &v,bool = true);
double TestpiGamma_P(const XNablaStokes &xnabla,bool = true);
double TestpiGamma_Gck(const XNablaStokes &xnabla,bool = true);
double TestpiPotential_Gck(const XNablaStokes &xnabla,bool = true);
double TestpiGamma_Gk(const XNablaStokes &xnabla,bool = true);
double TestpiPotential_Gk(const XNablaStokes &xnabla,bool = true);

template<typename T>
XNablaStokes::FunctionType FormalFunction(T &v) {
  return [&v](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = v[0].evaluate(x,0,0,0);
    rv(1) = v[1].evaluate(x,0,0,0);
    rv(2) = v[2].evaluate(x,0,0,0);
    return rv;};
}
template<typename T>
XVLStokes::FunctionType FormalGrad(T &v) {
  return [&v](const VectorRd &x)->Eigen::Matrix3d {
    Eigen::Matrix3d rv;
    rv(0,0) = v[0].evaluate(x,1,0,0);
    rv(0,1) = v[0].evaluate(x,0,1,0);
    rv(0,2) = v[0].evaluate(x,0,0,1);
    rv(1,0) = v[1].evaluate(x,1,0,0);
    rv(1,1) = v[1].evaluate(x,0,1,0);
    rv(1,2) = v[1].evaluate(x,0,0,1);
    rv(2,0) = v[2].evaluate(x,1,0,0);
    rv(2,1) = v[2].evaluate(x,0,1,0);
    rv(2,2) = v[2].evaluate(x,0,0,1);
    return rv;};
}
template<typename T>
XSLStokes::FunctionType FormalDivergence(T &v) {
  return [&v](const VectorRd &x)->double {
    double rv = 0;
    rv += v[0].evaluate(x,1,0,0);
    rv += v[1].evaluate(x,0,1,0);
    rv += v[2].evaluate(x,0,0,1);
    return rv;};
}

template<size_t > int validate_potential();

int main() {
  std::cout << std::endl << "[main] Test with degree 0" << std::endl; 
  validate_potential<0>();
  std::cout << std::endl << "[main] Test with degree 1" << std::endl;
  validate_potential<1>();
  std::cout << std::endl << "[main] Test with degree 2" << std::endl;
  validate_potential<2>();
  std::cout << std::endl << "Number of unexpected result : "<< nb_errors << std::endl;
  return nb_errors;
}
 
template<size_t degree>
int validate_potential() {

  // Build the mesh
  MeshBuilder builder = MeshBuilder(mesh_file);
  std::unique_ptr<Mesh> mesh_ptr = builder.build_the_mesh();
  std::cout << FORMAT(25) << "[main] Mesh size" << mesh_ptr->h_max() << std::endl;
  
  // Create core 
  StokesCore stokes_core(*mesh_ptr,degree);
  std::cout << "[main] StokesCore constructed" << std::endl;

  // Create discrete space XCurlStokes
  XCurlStokes xcurl(stokes_core);
  std::cout << "[main] XCurlStokes constructed" << std::endl;

  // Create discrete space XNablaStokes
  XNablaStokes xnabla(stokes_core);
  std::cout << "[main] XNablaStokes constructed" << std::endl;

  // Create discrete space XVLStokes
  XVLStokes xvl(stokes_core);
  std::cout << "[main] XVLStokes constructed" << std::endl;

  // Create discrete space XSLStokes
  XSLStokes xsl(stokes_core);
  std::cout << "[main] XSLStokes constructed" << std::endl;


  // Create test functions
  std::vector<PolyTest<degree>> Pkx{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 1>> Pkpox{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 2>> Pkp2x{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 3>> Pkp3x{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<TrigTest<degree>> Ttrigx{Initialization::Random,Initialization::Random,Initialization::Random};
  
  // Test 1 : CD : GI = IG
  std::cout << "[main] Begining of test E" << std::endl;
  std::cout << "CD : CE" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "Error for Pk :"<< TestCDGE(xnabla, xvl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDGE(xnabla, xvl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDGE(xnabla, xvl, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDGE(xnabla, xvl, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDGE(xnabla, xvl, Ttrigx) << endls;

  std::cout << "Potential E" << std::endl;
  std::cout << "We expected zero up to degree k+3" << std::endl;
  std::cout << "Error for Pk :"<< TestContI(xnabla, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestContI(xnabla, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestContI(xnabla, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestContI(xnabla, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestContI(xnabla, Ttrigx,false) << endls;

  std::cout << "[main] Begining of test F" << std::endl;
  std::cout << "CD : CF" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "Error for Pk :"<< TestCDGF(xnabla, xvl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDGF(xnabla, xvl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDGF(xnabla, xvl, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDGF(xnabla, xvl, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDGF(xnabla, xvl, Ttrigx) << endls;

  // Test 2 : Potential Consistency : gamma pIv = v
  std::cout << "We expect zero up to degree k+2" << std::endl;
  std::cout << "Potential Consistency : GammaI" << std::endl;
  std::cout << "Error for Pk :" << TestGammaI(xnabla, Pkx) << endls;
  std::cout << "Error for Pkpo :" << TestGammaI(xnabla, Pkpox) << endls;
  std::cout << "Error for Pkp2 :" << TestGammaI(xnabla, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :" << TestGammaI(xnabla, Pkp3x,false) << endls;
  std::cout << "Error for Ttrig :" << TestGammaI(xnabla, Ttrigx,false) << endls;

  // Test 3 : pipv = vF
  std::cout << "Potential Consistency : piGamma" << std::endl;
  std::cout << "We expect everything to be zero" << std::endl;
  std::cout << "Error for pi_Pk :" << TestpiGamma_P(xnabla) << endls; 
  std::cout << "Error for pi_Gck :" << TestpiGamma_Gck(xnabla) << endls; 
  std::cout << "Error for pi_Gk :" << TestpiGamma_Gk(xnabla) << endls; 

  std::cout << "[main] Begining of test T" << std::endl;
  std::cout << "CD : CT" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "Error for Pk :"<< TestCDGT(xnabla, xvl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDGT(xnabla, xvl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDGT(xnabla, xvl, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDGT(xnabla, xvl, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDGT(xnabla, xvl, Ttrigx) << endls;

  std::cout << "CD : DT" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "Error for Pk :"<< TestCDDT(xnabla, xsl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDDT(xnabla, xsl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDDT(xnabla, xsl, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDDT(xnabla, xsl, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDDT(xnabla, xsl, Ttrigx) << endls;

  // Test 2 : Potential Consistency : gamma pIv = v
  std::cout << "We expect zero up to degree k+1" << std::endl;
  std::cout << "Potential Consistency : PotentialI" << std::endl;
  std::cout << "Error for Pk :" << TestPotentialI(xnabla, Pkx) << endls;
  std::cout << "Error for Pkpo :" << TestPotentialI(xnabla, Pkpox) << endls;
  std::cout << "Error for Pkp2 :" << TestPotentialI(xnabla, Pkp2x,false) << endls;
  std::cout << "Error for Pkp3 :" << TestPotentialI(xnabla, Pkp3x,false) << endls;
  std::cout << "Error for Ttrig :" << TestPotentialI(xnabla, Ttrigx,false) << endls;

  // Test 3 : pipv = vF
  std::cout << "Potential Consistency : piPotential" << std::endl;
  std::cout << "We expect everything to be zero" << std::endl;
  std::cout << "Error for pi_Gck :" << TestpiPotential_Gck(xnabla) << endls; 
  std::cout << "Error for pi_Gk :" << TestpiPotential_Gk(xnabla) << endls;
  return 0;
}

template<typename Type> double TestCDGE(const XNablaStokes &xnabla,const XVLStokes &xvl, Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xnabla.interpolate(FormalFunction(v),1,1,14);
  Eigen::VectorXd IGv = xvl.interpolate(FormalGrad(v),1,1,14);
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_edges());
  
  // 
  parallel_for(xnabla.mesh().n_edges(),
    [&xnabla,&xvl,v, &Iv, &IGv,&err](size_t start, size_t end)->void
    {
      for (size_t iE = start; iE < end; iE++){
        Eigen::VectorXd GIv = xnabla.edgeOperators(iE).nabla*xnabla.restrictEdge(iE,Iv);
        // Difference
        GIv -= xvl.restrictEdge(iE,IGv);
        // L2 product
        Eigen::MatrixXd gram_Edge = xvl.compute_Gram_Edge(iE);
        err(iE) = std::sqrt(GIv.transpose()*gram_Edge*GIv);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename T> double TestContI(const XNablaStokes &xnabla,T &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xnabla.interpolate(FormalFunction(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_edges());
  
  // 
  parallel_for(xnabla.mesh().n_edges(),
    [&xnabla,v, &Iv,&err](size_t start, size_t end)->void
    {
      for (size_t iE = start; iE < end; iE++){
        Eigen::VectorXd PIv = xnabla.edgeOperators(iE).potential*xnabla.restrictEdge(iE,Iv);
        // Difference
        // interpolate
        const Edge &E = *xnabla.mesh().edge(iE);
        QuadratureRule quad_dqr_E = generate_quadrature_rule(E,2*(xnabla.degree() + 4));
        auto basis_Pk3p3_E_quad = evaluate_quad<Function>::compute(*xnabla.edgeBases(iE).Polyk3p3, quad_dqr_E);
        Eigen::VectorXd Ip3v = l2_projection(FormalFunction(v),*xnabla.edgeBases(iE).Polyk3p3,quad_dqr_E,basis_Pk3p3_E_quad);
        Eigen::VectorXd ErrV = PIv - Ip3v;
        // L2 product
        Eigen::MatrixXd gram_Pk3p3 = compute_gram_matrix(basis_Pk3p3_E_quad,quad_dqr_E);
        err(iE) = std::sqrt(ErrV.transpose()*gram_Pk3p3*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestCDGF(const XNablaStokes &xnabla,const XVLStokes &xvl, Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xnabla.interpolate(FormalFunction(v),1,14,14);
  Eigen::VectorXd IGv = xvl.interpolate(FormalGrad(v),1,14,14);
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_faces());
  
  // 
  parallel_for(xnabla.mesh().n_faces(),
    [&xnabla,&xvl,v, &Iv, &IGv,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        Eigen::VectorXd GIv = xnabla.faceOperators(iF).nabla*xnabla.restrictFace(iF,Iv);
        // Difference
        GIv -= xvl.restrictFace(iF,IGv).tail(xvl.numLocalDofsFace());
        // L2 product
        Eigen::MatrixXd gram_Face = xvl.compute_Gram_Face(iF);
        err(iF) = std::sqrt(GIv.transpose()*gram_Face*GIv);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestCDGT(const XNablaStokes &xnabla,const XVLStokes &xvl, Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xnabla.interpolate(FormalFunction(v),14,14,14);
  Eigen::VectorXd IGv = xvl.interpolate(FormalGrad(v),14,14,14);
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  
  // 
  parallel_for(xnabla.mesh().n_cells(),
    [&xnabla,&xvl,v, &Iv, &IGv,&err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        Eigen::VectorXd GIv = xnabla.cellOperators(iT).nabla*xnabla.restrictCell(iT,Iv);
        // Difference
        GIv -= xvl.restrictCell(iT,IGv).tail(xvl.numLocalDofsCell());
        // L2 product
        Eigen::MatrixXd gram_Cell = xvl.compute_Gram_Cell(iT);
        err(iT) = std::sqrt(GIv.transpose()*gram_Cell*GIv);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestCDDT(const XNablaStokes &xnabla,const XSLStokes &xsl, Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xnabla.interpolate(FormalFunction(v),14,14,14);
  Eigen::VectorXd IGv = xsl.interpolate(FormalDivergence(v),14);
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  
  // 
  parallel_for(xnabla.mesh().n_cells(),
    [&xnabla,&xsl,v, &Iv, &IGv,&err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        Eigen::VectorXd GIv = xnabla.cellOperators(iT).divergence*xnabla.restrictCell(iT,Iv);
        // Difference
        size_t dim_Pk_T = PolynomialSpaceDimension<Cell>::Poly(xnabla.degree());
        GIv -= IGv.segment(iT*dim_Pk_T,dim_Pk_T);
        // L2 product
        Eigen::MatrixXd gram_Cell = xsl.compute_Gram_Cell(iT);
        err(iT) = std::sqrt(GIv.transpose()*gram_Cell*GIv);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename T> double TestGammaI(const XNablaStokes &xnabla,T &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xnabla.interpolate(FormalFunction(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_faces());
  
  // 
  parallel_for(xnabla.mesh().n_faces(),
    [&xnabla,v,&Iv,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face &F = *xnabla.mesh().face(iF);
        Eigen::MatrixXd GIv = xnabla.faceOperators(iF).potential * xnabla.restrictFace(iF,Iv);

        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xnabla.degree()+4);
        auto basis_Pk3p2_F_quad = evaluate_quad<Function>::compute(*xnabla.faceBases(F).Polyk3p2, quad_2kp2_F);
        Eigen::VectorXd pi_v = l2_projection(FormalFunction(v),*xnabla.faceBases(F).Polyk3p2, quad_2kp2_F, basis_Pk3p2_F_quad);
        Eigen::VectorXd ErrV = pi_v - GIv;
        err(iF) = std::sqrt(ErrV.transpose()*compute_gram_matrix(basis_Pk3p2_F_quad,quad_2kp2_F)*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestPotentialI(const XNablaStokes &xnabla,Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xnabla.interpolate(FormalFunction(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  
  // 
  parallel_for(xnabla.mesh().n_cells(),
    [&xnabla,v,&Iv,&err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        const Cell &T = *xnabla.mesh().cell(iT);
        Eigen::MatrixXd GIv = xnabla.cellOperators(iT).potential * xnabla.restrictCell(iT,Iv);

        QuadratureRule quad_2kp2_T = generate_quadrature_rule(T,2*xnabla.degree()+4);
        auto basis_Pk3po_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(T).Polyk3po, quad_2kp2_T);
        Eigen::VectorXd pi_v = l2_projection(FormalFunction(v),*xnabla.cellBases(T).Polyk3po, quad_2kp2_T, basis_Pk3po_T_quad);
        Eigen::VectorXd ErrV = pi_v - GIv;
        err(iT) = std::sqrt(ErrV.transpose()*compute_gram_matrix(basis_Pk3po_T_quad,quad_2kp2_T)*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestpiGamma_P(const XNablaStokes &xnabla,bool expect_zero) {
   Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_faces());
  //
  parallel_for(xnabla.mesh().n_faces(),
    [&xnabla,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face & F = *xnabla.mesh().face(iF);
        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xnabla.degree()+4);
        auto basis_Pk_F_quad = evaluate_quad<Function>::compute(*xnabla.faceBases(F).Polyk,quad_2kp2_F);
        auto basis_Pk3p2_nF_F_quad = scalar_product(evaluate_quad<Function>::compute(*xnabla.faceBases(F).Polyk3p2,quad_2kp2_F),F.normal());
        
        Eigen::MatrixXd gram_Pk = compute_gram_matrix(basis_Pk_F_quad,quad_2kp2_F);
        Eigen::MatrixXd gram_Pk_Pk3p2_nF = compute_gram_matrix(basis_Pk_F_quad,basis_Pk3p2_nF_F_quad,quad_2kp2_F,"nonsym");
        Eigen::MatrixXd pi_gamma = gram_Pk.ldlt().solve(gram_Pk_Pk3p2_nF) * xnabla.faceOperators(iF).potential;

        // Extract wF
        size_t dim_Pk_F = PolynomialSpaceDimension<Face>::Poly(xnabla.degree());
        size_t offset_F = xnabla.localOffset(F);
        Eigen::MatrixXd ex_wF = Eigen::MatrixXd::Zero(dim_Pk_F,xnabla.dimensionFace(F));
        ex_wF.middleCols(offset_F,dim_Pk_F) = Eigen::MatrixXd::Identity(dim_Pk_F,dim_Pk_F);
        
        // Error
        err(iF) = (ex_wF - pi_gamma).norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestpiGamma_Gck(const XNablaStokes &xnabla,bool expect_zero) {
  if (xnabla.degree()==0) return 0.;
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_faces());
  //
  parallel_for(xnabla.mesh().n_faces(),
    [&xnabla,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face & F = *xnabla.mesh().face(iF);
        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xnabla.degree()+4);
        auto basis_Gck_F_quad = evaluate_quad<Function>::compute(*xnabla.faceBases(F).GolyComplk,quad_2kp2_F);
        auto basis_Pk3p2_F_quad = evaluate_quad<Function>::compute(*xnabla.faceBases(F).Polyk3p2,quad_2kp2_F);
        
        Eigen::MatrixXd gram_Gck = compute_gram_matrix(basis_Gck_F_quad,quad_2kp2_F);
        Eigen::MatrixXd gram_Gck_Pk3p2 = compute_gram_matrix(basis_Gck_F_quad,basis_Pk3p2_F_quad,quad_2kp2_F,"nonsym");
        Eigen::MatrixXd pi_gamma = gram_Gck.ldlt().solve(gram_Gck_Pk3p2) * xnabla.faceOperators(iF).potential;

        // Extract wF
        size_t dim_Pk_F = PolynomialSpaceDimension<Face>::Poly(xnabla.degree());
        size_t dim_Gk_F = PolynomialSpaceDimension<Face>::Goly(xnabla.degree());
        size_t dim_Gck_F = PolynomialSpaceDimension<Face>::GolyCompl(xnabla.degree());
        size_t offset_F = xnabla.localOffset(F);
        Eigen::MatrixXd ex_wF = Eigen::MatrixXd::Zero(dim_Gck_F,xnabla.dimensionFace(F));
        ex_wF.middleCols(offset_F+dim_Pk_F+dim_Gk_F,dim_Gck_F) = Eigen::MatrixXd::Identity(dim_Gck_F,dim_Gck_F);
        
        // Error
        err(iF) = (ex_wF - pi_gamma).norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestpiPotential_Gck(const XNablaStokes &xnabla,bool expect_zero) {
  if (xnabla.degree()==0) return 0.;
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  //
  parallel_for(xnabla.mesh().n_cells(),
    [&xnabla,&err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        const Cell & T = *xnabla.mesh().cell(iT);
        QuadratureRule quad_2kp2_T = generate_quadrature_rule(T,2*xnabla.degree()+4);
        auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(T).GolyComplk,quad_2kp2_T);
        auto basis_Pk3po_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(T).Polyk3po,quad_2kp2_T);
        
        Eigen::MatrixXd gram_Gck = compute_gram_matrix(basis_Gck_T_quad,quad_2kp2_T);
        Eigen::MatrixXd gram_Gck_Pk3po = compute_gram_matrix(basis_Gck_T_quad,basis_Pk3po_T_quad,quad_2kp2_T,"nonsym");
        Eigen::MatrixXd pi_gamma = gram_Gck.ldlt().solve(gram_Gck_Pk3po) * xnabla.cellOperators(iT).potential;

        // Extract wT
        size_t dim_Gkmo_T = PolynomialSpaceDimension<Cell>::Goly(xnabla.degree()-1);
        size_t dim_Gck_T = PolynomialSpaceDimension<Cell>::GolyCompl(xnabla.degree());
        size_t offset_T = xnabla.localOffset(T);
        Eigen::MatrixXd ex_wT = Eigen::MatrixXd::Zero(dim_Gck_T,xnabla.dimensionCell(T));
        ex_wT.middleCols(offset_T+dim_Gkmo_T,dim_Gck_T) = Eigen::MatrixXd::Identity(dim_Gck_T,dim_Gck_T);
        
        // Error
        err(iT) = (ex_wT - pi_gamma).norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestpiGamma_Gk(const XNablaStokes &xnabla,bool expect_zero) {
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_faces());
  //
  parallel_for(xnabla.mesh().n_faces(),
    [&xnabla,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face & F = *xnabla.mesh().face(iF);
        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xnabla.degree()+4);
        auto basis_Gk_F_quad = evaluate_quad<Function>::compute(*xnabla.faceBases(F).Golyk,quad_2kp2_F);
        auto basis_Pk3p2_F_quad = evaluate_quad<Function>::compute(*xnabla.faceBases(F).Polyk3p2,quad_2kp2_F);
        
        Eigen::MatrixXd gram_Gk = compute_gram_matrix(basis_Gk_F_quad,quad_2kp2_F);
        Eigen::MatrixXd gram_Gk_Pk3p2 = compute_gram_matrix(basis_Gk_F_quad,basis_Pk3p2_F_quad,quad_2kp2_F,"nonsym");
        Eigen::MatrixXd pi_gamma = gram_Gk.ldlt().solve(gram_Gk_Pk3p2) * xnabla.faceOperators(iF).potential;

        // Extract wF
        size_t dim_Pk_F = PolynomialSpaceDimension<Face>::Poly(xnabla.degree());
        size_t dim_Gk_F = PolynomialSpaceDimension<Face>::Goly(xnabla.degree());
        size_t offset_F = xnabla.localOffset(F);
        Eigen::MatrixXd ex_wF = Eigen::MatrixXd::Zero(dim_Gk_F,xnabla.dimensionFace(F));
        ex_wF.middleCols(offset_F+dim_Pk_F,dim_Gk_F) = Eigen::MatrixXd::Identity(dim_Gk_F,dim_Gk_F);
        
        // Error
        err(iF) = (ex_wF - pi_gamma).norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestpiPotential_Gk(const XNablaStokes &xnabla,bool expect_zero) {
  if (xnabla.degree()==0) return 0.;
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  //
  parallel_for(xnabla.mesh().n_cells(),
    [&xnabla,&err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        const Cell & T = *xnabla.mesh().cell(iT);
        QuadratureRule quad_2kp2_T = generate_quadrature_rule(T,2*xnabla.degree()+4);
        auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(T).Golykmo,quad_2kp2_T);
        auto basis_Pk3po_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(T).Polyk3po,quad_2kp2_T);
        
        Eigen::MatrixXd gram_Gkmo = compute_gram_matrix(basis_Gkmo_T_quad,quad_2kp2_T);
        Eigen::MatrixXd gram_Gkmo_Pk3po = compute_gram_matrix(basis_Gkmo_T_quad,basis_Pk3po_T_quad,quad_2kp2_T,"nonsym");
        Eigen::MatrixXd pi_gamma = gram_Gkmo.ldlt().solve(gram_Gkmo_Pk3po) * xnabla.cellOperators(iT).potential;

        // Extract wT
        size_t dim_Gkmo_T = PolynomialSpaceDimension<Cell>::Goly(xnabla.degree()-1);
        size_t offset_T = xnabla.localOffset(T);
        Eigen::MatrixXd ex_wT = Eigen::MatrixXd::Zero(dim_Gkmo_T,xnabla.dimensionCell(T));
        ex_wT.middleCols(offset_T,dim_Gkmo_T) = Eigen::MatrixXd::Identity(dim_Gkmo_T,dim_Gkmo_T);
        
        // Error
        err(iT) = (ex_wT - pi_gamma).norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}


