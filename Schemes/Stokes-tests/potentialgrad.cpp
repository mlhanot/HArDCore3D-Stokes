// Tests for the gradient operator.
// It computes otherwise useless quantities and is not optimized in any way.
// Warning : Ttrig requieres an extremely high degree for the interpolation.

#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xgradstokes.hpp>
#include <xcurlstokes.hpp>
#include "testfunction.hpp"

#include <parallel_for.hpp>

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;

const std::string mesh_file = "./meshes/" "Voro-small-2/RF_fmt/voro.2";

constexpr bool use_threads = true;
// Foward declare
template<typename T> double TestCDuGE(const XGradStokes &xgrad, const XCurlStokes &xcurl,T &v,bool = true);
template<typename T> double TestCDGF(const XGradStokes &xgrad, const XCurlStokes &xcurl,T &v,bool = true);
template<typename T> double TestCDGpF(const XGradStokes &xgrad, const XCurlStokes &xcurl,T &v,bool = true);
template<typename T> double TestCDGT(const XGradStokes &xgrad, const XCurlStokes &xcurl,T &v,bool = true);
template<typename T> double TestCDuGT(const XGradStokes &xgrad, const XCurlStokes &xcurl,T &v,bool = true);
template<typename T> double TestGammaI(const XGradStokes &xgrad, T &v,bool = true);
double TestpiGamma(const XGradStokes &xgrad,const Eigen::VectorXd &,bool = true);

template<typename T>
XGradStokes::FunctionType FormalFunction(T &v) {
  return [&v](const VectorRd &x)->double {
    return v.evaluate(x,0,0,0);};
}
template<typename T>
XGradStokes::FunctionGradType FormalGrad(T &v) {
  return [&v](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = v.evaluate(x,1,0,0);
    rv(1) = v.evaluate(x,0,1,0);
    rv(2) = v.evaluate(x,0,0,1);
    return rv;};
}
template<typename T>
XCurlStokes::FunctionGradType FormalGradGrad(T &v) {
  return [&v](const VectorRd &x)->Eigen::Matrix3d {
    Eigen::Matrix3d rv;
    rv(0,0) = v.evaluate(x,2,0,0);
    rv(0,1) = v.evaluate(x,1,1,0);
    rv(0,2) = v.evaluate(x,1,0,1);
    rv(1,0) = v.evaluate(x,1,1,0);
    rv(1,1) = v.evaluate(x,0,2,0);
    rv(1,2) = v.evaluate(x,0,1,1);
    rv(2,0) = v.evaluate(x,1,0,1);
    rv(2,1) = v.evaluate(x,0,1,1);
    rv(2,2) = v.evaluate(x,0,0,2);
    return rv;};
}
template<typename T>
XCurlStokes::FunctionCurlType FormalCurlGrad(T &v) {
  return [&v](const VectorRd &x)->VectorRd {
    return VectorRd::Zero(3);};
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

  // Create discrete space XGradStokes
  XGradStokes xgrad(stokes_core);
  std::cout << "[main] XGradStokes constructed" << std::endl;

  // Create discrete space XCurlStokes
  XCurlStokes xcurl(stokes_core);
  std::cout << "[main] XCurlStokes constructed" << std::endl;

  // Create test functions
  PolyTest<degree> Pkx(Initialization::Random);
  PolyTest<degree + 1> Pkpox(Initialization::Random);
  PolyTest<degree + 2> Pkp2x(Initialization::Random);
  PolyTest<degree + 3> Pkp3x(Initialization::Random);
  TrigTest<degree> Ttrigx(Initialization::Random);
  
  // Test 1 : CD : GI = IG
  std::cout << "[main] Begining of test CD" << std::endl;
  std::cout << "CD : uGE" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "Error for Pk :"<< TestCDuGE(xgrad,xcurl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDuGE(xgrad, xcurl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDuGE(xgrad, xcurl, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDuGE(xgrad, xcurl, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDuGE(xgrad, xcurl, Ttrigx) << endls;
  std::cout << "CD : CurlF" << std::endl;
  std::cout << "We expected zero up to degree k+1" << std::endl;
  std::cout << "Error for Pk :"<< TestCDGF(xgrad, xcurl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDGF(xgrad, xcurl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDGF(xgrad, xcurl, Pkp2x,false) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDGF(xgrad, xcurl, Pkp3x,false) << endls;
  std::cout << "Error for Ttrig :"<< TestCDGF(xgrad, xcurl, Ttrigx,false) << endls;
  /*
  std::cout << "CD : CurlperpF" << std::endl;
  std::cout << "We expected zero up to degree k+1" << std::endl;
  std::cout << "Error for Pk :"<< TestCDGpF(xgrad, xcurl, Pk) << endls;
  std::cout << "Error for Pkpo :"<< TestCDGpF(xgrad, xcurl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDGpF(xgrad, xcurl, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDGpF(xgrad, xcurl, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDGpF(xgrad, xcurl, Ttrigx) << endls;
  */
  std::cout << "CD : GT" << std::endl;
  std::cout << "We expected zero up to degree k+1" << std::endl;
  std::cout << "Error for Pk :"<< TestCDGT(xgrad, xcurl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDGT(xgrad, xcurl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDGT(xgrad, xcurl, Pkp2x,false) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDGT(xgrad, xcurl, Pkp3x,false) << endls;
  std::cout << "Error for Ttrig :"<< TestCDGT(xgrad, xcurl, Ttrigx,false) << endls;

  std::cout << "CD : uGT" << std::endl;
  std::cout << "We expected everything to be zero up" << std::endl;
  std::cout << "Error for Pk :"<< TestCDuGT(xgrad, xcurl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDuGT(xgrad, xcurl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDuGT(xgrad, xcurl, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDuGT(xgrad, xcurl, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDuGT(xgrad, xcurl, Ttrigx) << endls;

  // Test 2 : Potential Consistency : gamma pIv = v, v dans Pkpo
  std::cout << "[main] Begining of test Potential Consistency" << std::endl;
  std::cout << "We expect zero up to degree k+1" << std::endl;
  std::cout << "Potential Consistency : Face" << std::endl;
  std::cout << "Error for Pk :" << TestGammaI(xgrad, Pkx) << endls;
  std::cout << "Error for Pkpo :" << TestGammaI(xgrad, Pkpox) << endls;
  std::cout << "Error for Pkp2 :" << TestGammaI(xgrad, Pkp2x,false) << endls;
  std::cout << "Error for Pkp3 :" << TestGammaI(xgrad, Pkp3x,false) << endls;
  std::cout << "Error for Ttrig :" << TestGammaI(xgrad, Ttrigx,false) << endls;

  // Test 3 : pipv = vF
  std::cout << "[main] Begining of test Potential Consistency 2" << std::endl;
  std::cout << "We expect everything to be zero" << std::endl;
  Eigen::VectorXd randomdofs = Eigen::VectorXd::Zero(xgrad.dimension());
  fill_random_vector(randomdofs);
  std::cout << "Error for pi_Gkmo, pi_Gck :" << TestpiGamma(xgrad, randomdofs) << endls; 
  fill_random_vector(randomdofs);
  std::cout << "Error for pi_Gkmo, pi_Gck :" << TestpiGamma(xgrad, randomdofs) << endls; 


  return 0;
}

// Return weighted Linf discrete norm
template<typename T> double TestCDuGE(const XGradStokes &xgrad, const XCurlStokes &xcurl,T &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iq = xgrad.interpolate(FormalFunction(v),FormalGrad(v),-1,-1,16);
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_edges());
  Eigen::VectorXd IGq = xcurl.interpolate(FormalGrad(v),FormalCurlGrad(v),FormalGradGrad(v),-1,-1,16);
  // Compute IGq - GIq
  parallel_for(xcurl.mesh().n_edges(), 
    [&xgrad,&xcurl,&Iq,&IGq,&err,v](size_t start,size_t end)->void
    {
      for (size_t iE = start; iE < end; iE++) {
        Eigen::MatrixXd uhr = xgrad.restrictEdge(iE,Iq);
        Eigen::MatrixXd ErrV = xcurl.restrictEdge(iE,IGq) - xgrad.edgeOperators(iE).gradient*uhr;
        err(iE) = std::sqrt((ErrV.transpose()*xcurl.computeL2opnuEdge(iE)*ErrV)(0,0));
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

Eigen::VectorXd interpolate_P2k_F(const XCurlStokes &xcurl, size_t iF, const std::function<VectorRd(const VectorRd & x,size_t iF)> &f) {
  Face const &F = *xcurl.mesh().face(iF);
  size_t dqr = 2*xcurl.degree() + 3;
  QuadratureRule quad_dqr_F = generate_quadrature_rule(F,dqr);
  auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*xcurl.faceBases(F).Polyk2, quad_dqr_F);
  auto f_iF = [&f,iF](const Eigen::Vector3d & x)->VectorRd {
    return f(x,iF);
  };
  return l2_projection(f_iF,*xcurl.faceBases(F).Polyk2, quad_dqr_F,basis_Pk2_F_quad);
}

template<typename T> double TestCDGF(const XGradStokes &xgrad, const XCurlStokes &xcurl,T &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iq = xgrad.interpolate(FormalFunction(v),FormalGrad(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_faces());
  // 
  auto local_function = [&v,&xcurl](const VectorRd & x, size_t iF)->VectorRd
    {
      VectorRd nF = xcurl.mesh().face(iF)->normal();
      return nF.cross(FormalGrad(v)(x).cross(nF));
    };
  //
  parallel_for(xcurl.mesh().n_faces(),
    [&xgrad,&xcurl,&Iq,local_function,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face &F = *xcurl.mesh().face(iF);
        Eigen::VectorXd IGq = interpolate_P2k_F(xcurl,iF,local_function);
        Eigen::VectorXd uhr = xgrad.restrictFace(iF,Iq);
        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xcurl.degree()+2);
        auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*xcurl.faceBases(F).Polyk2, quad_2kp2_F);
        Eigen::VectorXd ErrV = IGq - xgrad.faceOperators(iF).gradient*uhr;
        err(iF) = std::sqrt(ErrV.transpose()*compute_gram_matrix(basis_Pk2_F_quad,quad_2kp2_F)*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestCDGT(const XGradStokes &xgrad, const XCurlStokes &xcurl,Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iq = xgrad.interpolate(FormalFunction(v),FormalGrad(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  
  // 
  parallel_for(xcurl.mesh().n_cells(),
    [&xgrad,&xcurl,v, &Iq,&err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        const Cell &T = *xcurl.mesh().cell(iT);
        QuadratureRule quad_2kp2_T = generate_quadrature_rule(T,2*xcurl.degree()+3);
        auto basis_Pk3_T_quad = evaluate_quad<Function>::compute(*xcurl.cellBases(T).Polyk3, quad_2kp2_T);
        Eigen::VectorXd IGq = l2_projection(FormalGrad(v),*xcurl.cellBases(T).Polyk3, quad_2kp2_T, basis_Pk3_T_quad);
        Eigen::VectorXd uhr = xgrad.restrictCell(iT,Iq);
        Eigen::VectorXd ErrV = IGq - xgrad.cellOperators(iT).gradient*uhr;
        err(iT) = std::sqrt(ErrV.transpose()*compute_gram_matrix(basis_Pk3_T_quad,quad_2kp2_T)*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestCDuGT(const XGradStokes &xgrad, const XCurlStokes &xcurl,Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iq = xgrad.interpolate(FormalFunction(v),FormalGrad(v),14,14,16);
  Eigen::VectorXd IGq = xcurl.interpolate(FormalGrad(v),FormalCurlGrad(v),FormalGradGrad(v),14,14,16);
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  
  // Compute IGq - GIq
  parallel_for(xcurl.mesh().n_cells(), 
    [&xgrad,&xcurl,&Iq,&IGq,&err,v](size_t start,size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++) {
        Eigen::MatrixXd uhr = xgrad.restrictCell(iT,Iq);
        Eigen::MatrixXd ErrV = xcurl.restrictCell(iT,IGq) - xgrad.buildGradientComponentsCell(iT)*uhr;
        err(iT) = ErrV.norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename T> double TestGammaI(const XGradStokes &xgrad,T &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iq = xgrad.interpolate(FormalFunction(v),FormalGrad(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xgrad.mesh().n_faces());
  
  // 
  parallel_for(xgrad.mesh().n_faces(),
    [&xgrad,v,&Iq,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face &F = *xgrad.mesh().face(iF);
        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xgrad.degree()+3);
        auto basis_Pkpo_F_quad = evaluate_quad<Function>::compute(*xgrad.faceBases(F).Polykpo, quad_2kp2_F);
        Eigen::VectorXd IGq = l2_projection(FormalFunction(v),*xgrad.faceBases(F).Polykpo, quad_2kp2_F, basis_Pkpo_F_quad);
        Eigen::VectorXd uhr = xgrad.restrictFace(iF,Iq);
        Eigen::VectorXd ErrV = IGq - xgrad.faceOperators(iF).potential*uhr;
        err(iF) = std::sqrt(ErrV.transpose()*compute_gram_matrix(basis_Pkpo_F_quad,quad_2kp2_F)*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestpiGamma(const XGradStokes &xgrad,const Eigen::VectorXd &uqh,bool expect_zero) {
   Eigen::VectorXd err = Eigen::VectorXd::Zero(xgrad.mesh().n_faces());
  //
  if (xgrad.degree()==0) return 0.;
  parallel_for(xgrad.mesh().n_faces(),
    [&xgrad,&uqh,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face &F = *xgrad.mesh().face(iF);
        Eigen::VectorXd uhr = xgrad.restrictFace(iF,uqh);
        size_t dimPkmo = PolynomialSpaceDimension<Face>::Poly(xgrad.degree()-1);
        Eigen::VectorXd ErrV = uqh.segment(xgrad.globalOffset(F),dimPkmo) - (xgrad.faceOperators(iF).potential*uhr).head(dimPkmo);
        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xgrad.degree()+1);
        auto basis_Pkmo_F_quad = evaluate_quad<Function>::compute(*xgrad.faceBases(F).Polykmo, quad_2kp2_F);
        err(iF) = std::sqrt(ErrV.transpose()*compute_gram_matrix(basis_Pkmo_F_quad,quad_2kp2_F)*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

