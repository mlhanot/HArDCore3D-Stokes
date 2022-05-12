// Tests for the gradient operator.
// It computes otherwise useless quantities and is not optimized in any way.
// Warning : Ttrig requieres an extremely high degree for the interpolation.

#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xgradstokes.hpp>
#include <xcurlstokes.hpp>
#include <xnablastokes.hpp>
#include "testfunction.hpp"

#include <parallel_for.hpp>

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;

const std::string mesh_file = "./meshes/" "Voro-small-2/RF_fmt/voro.2";

constexpr bool use_threads = true;
// Foward declare
template<typename T> double TestCDCT(const XCurlStokes &xcurl,T &v,bool = true);
template<typename T> double TestCDuCE(const XCurlStokes &xcurl,const XNablaStokes &xnabla, T &v,bool = true);
template<typename T> double TestCDuCT(const XCurlStokes &xcurl, const XNablaStokes &xnabla,T &v,bool = true);
template<typename T,typename S> double TestGammaI(const XCurlStokes &xcurl, T &v, S &u,bool = true);
double TestpiGamma(const XCurlStokes &xcurl,const Eigen::VectorXd &,bool = true);
double TestpiGammaGF(const XGradStokes &xgrad,const XCurlStokes &xcurl,const Eigen::VectorXd &,bool = true);
double TestGammaGF(const XGradStokes &xgrad,const XCurlStokes &xcurl,bool = true);

template<typename T> // Used for N^k
XCurlStokes::FunctionType FormalSGrad(T &v,const Face &F) {
  return [&v,F](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = v.evaluate(x,1,0,0);
    rv(1) = v.evaluate(x,0,1,0);
    rv(2) = v.evaluate(x,0,0,1);
    return F.normal().cross(rv).cross(F.normal());};
}
template<typename T>
XCurlStokes::FunctionGradType FormalSGradGrad(T &v,const Face &F) {
  return [&v,F](const VectorRd &x)->Eigen::Matrix3d {
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
    Eigen::Vector3d t = F.normal();
    Eigen::Matrix3d t_hat;
    t_hat << 0, -t(2), t(1),
      t(2), 0, -t(0),
      -t(1), t(0), 0;
    for (size_t i = 0; i < 3;i++) {
      rv.middleRows(i,1) = (t_hat.transpose()*t_hat*rv.middleRows(i,1).transpose()).transpose();
    }
    for (size_t i = 0; i < 3;i++) {
      rv.middleCols(i,1) = t_hat.transpose()*t_hat*rv.middleCols(i,1);
    }
    return rv;};
} // Used for N^k
XCurlStokes::FunctionCurlType FormalZero() {
  return [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = 0;
    rv(1) = 0;
    rv(2) = 0;
    return rv;};
}

template<typename T>
XCurlStokes::FunctionType FormalFunction(T &v) {
  return [&v](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = v[0].evaluate(x,0,0,0);
    rv(1) = v[1].evaluate(x,0,0,0);
    rv(2) = v[2].evaluate(x,0,0,0);
    return rv;};
}
template<typename T>
XCurlStokes::FunctionGradType FormalGrad(T &v) {
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
XCurlStokes::FunctionCurlType FormalCurl(T &v) {
  return [&v](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = v[2].evaluate(x,0,1,0) - v[1].evaluate(x,0,0,1);
    rv(1) = v[0].evaluate(x,0,0,1) - v[2].evaluate(x,1,0,0);
    rv(2) = v[1].evaluate(x,1,0,0) - v[0].evaluate(x,0,1,0);
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

  // Create discrete space XGradStokes
  XGradStokes xgrad(stokes_core);
  std::cout << "[main] XGradStokes constructed" << std::endl;

  // Create discrete space XCurlStokes
  XCurlStokes xcurl(stokes_core);
  std::cout << "[main] XCurlStokes constructed" << std::endl;

  // Create discrete space XNablaStokes
  XNablaStokes xnabla(stokes_core);
  std::cout << "[main] XNablaStokes constructed" << std::endl;

  // Create test functions
  std::vector<PolyTest<degree>> Pkx{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 1>> Pkpox{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 2>> Pkp2x{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 3>> Pkp3x{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 4>> Pkp4x{Initialization::Random,Initialization::Random,Initialization::Random};// Nkp3
  std::vector<TrigTest<degree>> Ttrigx{Initialization::Random,Initialization::Random,Initialization::Random};
  
  // Test 1 : CD : GI = IG
  std::cout << "[main] Begining of test CD" << std::endl;
  std::cout << "CD : CT" << std::endl;
  std::cout << "We expected zero up to degree k+1" << std::endl;
  std::cout << "Error for Pk :"<< TestCDCT(xcurl, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDCT(xcurl, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDCT(xcurl, Pkp2x,false) << endls;
  //std::cout << "Error for Pkp3 :"<< TestCDCT(xcurl, Pkp3x,false) << endls;
  //std::cout << "Error for Ttrig :"<< TestCDCT(xcurl, Ttrigx,false) << endls;

  std::cout << "CD : uCE" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "Error for Pk :"<< TestCDuCE(xcurl, xnabla, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDuCE(xcurl, xnabla, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDuCE(xcurl, xnabla, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDuCE(xcurl, xnabla, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDuCE(xcurl, xnabla, Ttrigx) << endls;

  std::cout << "CD : uCT" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "Error for Pk :"<< TestCDuCT(xcurl, xnabla, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestCDuCT(xcurl, xnabla, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDuCT(xcurl, xnabla, Pkp2x) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDuCT(xcurl, xnabla, Pkp3x) << endls;
  std::cout << "Error for Ttrig :"<< TestCDuCT(xcurl, xnabla, Ttrigx) << endls;

  // Test 2 : Potential Consistency : gamma pIv = v, v dans Pk2
  std::cout << "[main] Begining of test Potential Consistency" << std::endl;
  std::cout << "We expect zero up to degree k+1" << std::endl;
  std::cout << "Potential Consistency : Face" << std::endl;
  std::cout << "Error for Pk :" << TestGammaI(xcurl, Pkx, Pkpox) << endls;
  std::cout << "Error for Nkpo :" << TestGammaI(xcurl, Pkx, Pkp2x) << endls;
  std::cout << "Error for Nkp2 :" << TestGammaI(xcurl, Pkpox, Pkp3x,false) << endls;
  std::cout << "Error for Nkp3 :" << TestGammaI(xcurl, Pkp2x, Pkp4x,false) << endls;
  std::cout << "Error for Ttrig :" << TestGammaI(xcurl, Ttrigx, Pkp3x,false) << endls;

  // Test 3 : pipv = vF
  std::cout << "[main] Begining of test Potential Consistency 2" << std::endl;
  std::cout << "We expect everything to be zero" << std::endl;
  Eigen::VectorXd randomdofs = Eigen::VectorXd::Zero(xcurl.dimension());
  fill_random_vector(randomdofs);
  std::cout << "Error for pi_Gkmo, pi_Gck :" << TestpiGamma(xcurl, randomdofs) << endls; 
  fill_random_vector(randomdofs);
  std::cout << "Error for pi_Gkmo, pi_Gck :" << TestpiGamma(xcurl, randomdofs) << endls; 

  // Test 4 : pipG = piG
  std::cout << "[main] Begining of test Potential Consistency 3" << std::endl;
  std::cout << "We expect everything to be zero" << std::endl;
  Eigen::VectorXd randomdofs2 = Eigen::VectorXd::Zero(xgrad.dimension());
  fill_random_vector(randomdofs2);
  std::cout << "Error for pi_RTk :" << TestpiGammaGF(xgrad, xcurl, randomdofs2) << endls; 
  fill_random_vector(randomdofs2);
  std::cout << "Error for pi_RTk :" << TestpiGammaGF(xgrad, xcurl, randomdofs2) << endls; 
  // std::cout << "Error for gammaG - G :" << TestGammaGF(xgrad, xcurl,false) << endls; // Not 0 as expected

  return 0;
}

template<typename Type> double TestCDCT(const XCurlStokes &xcurl,Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xcurl.interpolate(FormalFunction(v),FormalCurl(v),FormalGrad(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  
  // 
  parallel_for(xcurl.mesh().n_cells(),
    [&xcurl,v, &Iv,&err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        const Cell &T = *xcurl.mesh().cell(iT);
        QuadratureRule quad_2kp2_T = generate_quadrature_rule(T,2*xcurl.degree()+3);
        auto basis_Pk3_T_quad = evaluate_quad<Function>::compute(*xcurl.cellBases(T).Polyk3, quad_2kp2_T);
        Eigen::VectorXd CIv = xcurl.cellOperators(iT).curl*xcurl.restrictCell(iT,Iv);
        double rv = 0;
        for (size_t iqn = 0; iqn < quad_2kp2_T.size();iqn++) {
          VectorRd valu = VectorRd::Zero();
          for (size_t i = 0; i < xcurl.cellBases(T).Polyk3->dimension();i++) {
            valu += basis_Pk3_T_quad[i][iqn]*CIv(i);
          }
          rv += quad_2kp2_T[iqn].w * (valu - FormalCurl(v)(quad_2kp2_T[iqn].vector())).dot(valu - FormalCurl(v)(quad_2kp2_T[iqn].vector()));
        }
        err(iT) = std::sqrt(rv);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestCDuCE(const XCurlStokes &xcurl,const XNablaStokes &xnabla, Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xcurl.interpolate(FormalFunction(v),FormalCurl(v),FormalGrad(v),14,14,14);
  Eigen::VectorXd ICv = xnabla.interpolate(FormalCurl(v),14,14,14);
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_edges());
  
  // Compute ICv - CIv
  parallel_for(xcurl.mesh().n_edges(), 
    [&xcurl,&xnabla,&Iv,&ICv,&err,v](size_t start,size_t end)->void
    {
      for (size_t iE = start; iE < end; iE++) {
        Eigen::MatrixXd uhr = xcurl.restrictEdge(iE,Iv);
        Eigen::MatrixXd ErrV = xnabla.restrictEdge(iE,ICv) - xcurl.edgeOperators(iE).curl*uhr;
        err(iE) = ErrV.norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestCDuCT(const XCurlStokes &xcurl,const XNablaStokes &xnabla, Type &v,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xcurl.interpolate(FormalFunction(v),FormalCurl(v),FormalGrad(v),14,14,14);
  Eigen::VectorXd ICv = xnabla.interpolate(FormalCurl(v),14,14,14);
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  
  // Compute ICv - CIv
  parallel_for(xcurl.mesh().n_cells(), 
    [&xcurl,&xnabla,&Iv,&ICv,&err,v](size_t start,size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++) {
        Eigen::MatrixXd uhr = xcurl.restrictCell(iT,Iv);
        Eigen::MatrixXd ErrV = xnabla.restrictCell(iT,ICv) - xcurl.buildCurlComponentsCell(iT)*uhr;
        err(iT) = ErrV.norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename T,typename S> double TestGammaI(const XCurlStokes &xcurl,T &v,S &u,bool expect_zero) {
  // Compute interpolates
  Eigen::VectorXd Iv = xcurl.interpolate(FormalFunction(v),FormalCurl(v),FormalGrad(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_faces());
  
  // 
  parallel_for(xcurl.mesh().n_faces(),
    [&xcurl,v,u,&Iv,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face &F = *xcurl.mesh().face(iF);
        Eigen::MatrixXd uhr = xcurl.restrictFace(iF,Iv)
          + xcurl.restrictFace(iF,xcurl.interpolate(FormalSGrad(u[0],F),FormalZero(),FormalSGradGrad(u[0],F))); // P^k + G^{k+1}

        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xcurl.degree()+4);
        auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*xcurl.faceBases(F).Polyk2, quad_2kp2_F);
        Eigen::VectorXd pi_v = l2_projection(FormalFunction(v),*xcurl.faceBases(F).Polyk2, quad_2kp2_F, basis_Pk2_F_quad);
        pi_v += l2_projection(FormalSGrad(u[0],F),*xcurl.faceBases(F).Polyk2, quad_2kp2_F, basis_Pk2_F_quad);
        Eigen::VectorXd ErrV = pi_v - xcurl.faceOperators(iF).potential*uhr;
        err(iF) = std::sqrt(ErrV.transpose()*compute_gram_matrix(basis_Pk2_F_quad,quad_2kp2_F)*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestpiGamma(const XCurlStokes &xcurl,const Eigen::VectorXd &uqh,bool expect_zero) {
   Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_faces());
  //
  if (xcurl.degree()==0) return 0.;
  parallel_for(xcurl.mesh().n_faces(),
    [&xcurl,&uqh,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face &F = *xcurl.mesh().face(iF);
        // pi_Rkmo pi_Rck
        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xcurl.degree()+1);
        auto basis_Rkmo_F_quad = evaluate_quad<Function>::compute(*xcurl.faceBases(F).Rolykmo, quad_2kp2_F);
        auto basis_Rck_F_quad = evaluate_quad<Function>::compute(*xcurl.faceBases(F).RolyComplk, quad_2kp2_F);
        auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*xcurl.faceBases(F).Polyk2, quad_2kp2_F);
        Eigen::MatrixXd LHS_Rkmo = compute_gram_matrix(basis_Rkmo_F_quad,quad_2kp2_F);
        Eigen::MatrixXd LHS_Rck = compute_gram_matrix(basis_Rck_F_quad,quad_2kp2_F);
        Eigen::MatrixXd RHS_Rkmo = compute_gram_matrix(basis_Rkmo_F_quad,basis_Pk2_F_quad,quad_2kp2_F);
        Eigen::MatrixXd RHS_Rck = compute_gram_matrix(basis_Rck_F_quad,basis_Pk2_F_quad,quad_2kp2_F);
        Eigen::MatrixXd pi_Rkmo = LHS_Rkmo.ldlt().solve(RHS_Rkmo);
        Eigen::MatrixXd pi_Rck = LHS_Rck.ldlt().solve(RHS_Rck);

        Eigen::VectorXd uhr = xcurl.faceOperators(iF).potential*xcurl.restrictFace(iF,uqh);
        size_t dimRkmo = PolynomialSpaceDimension<Face>::Roly(xcurl.degree()-1);
        Eigen::VectorXd ErrVkmo = uqh.segment(xcurl.globalOffset(F),dimRkmo) - pi_Rkmo*uhr;
        Eigen::VectorXd ErrVck = uqh.segment(xcurl.globalOffset(F)+dimRkmo,PolynomialSpaceDimension<Face>::RolyCompl(xcurl.degree())) - pi_Rck*uhr;

        err(iF) = std::sqrt((ErrVkmo.transpose()*LHS_Rkmo*ErrVkmo + ErrVck.transpose()*LHS_Rck*ErrVck)(0,0));
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestpiGammaGF(const XGradStokes &xgrad, const XCurlStokes &xcurl,const Eigen::VectorXd &uqh,bool expect_zero) {
   Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_faces());
  //
  if (xcurl.degree()==0) return 0.;
  parallel_for(xcurl.mesh().n_faces(),
    [&xgrad,&xcurl,&uqh,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        const Face &F = *xcurl.mesh().face(iF);
        // Construct RT
        StokesCore::RTBasisFaceType RTk(*xcurl.faceBases(F).Rolykmo,*xcurl.faceBases(F).RolyComplk);
        
        // pi_RTk
        QuadratureRule quad_2kp2_F = generate_quadrature_rule(F,2*xcurl.degree()+1);
        auto basis_RTk_F_quad = evaluate_quad<Function>::compute(RTk, quad_2kp2_F);
        auto basis_Pk2_F_quad = evaluate_quad<Function>::compute(*xcurl.faceBases(F).Polyk2, quad_2kp2_F);
        Eigen::MatrixXd LHS_RTk = compute_gram_matrix(basis_RTk_F_quad,quad_2kp2_F);
        Eigen::MatrixXd RHS_RTk = compute_gram_matrix(basis_RTk_F_quad,basis_Pk2_F_quad,quad_2kp2_F);
        Eigen::MatrixXd pi_RTk = LHS_RTk.ldlt().solve(RHS_RTk);
        // pi_RTk Gq
        Eigen::MatrixXd pi_Gq = pi_RTk*xgrad.faceOperators(iF).gradient*xgrad.restrictFace(iF,uqh);
        // errV
        Eigen::VectorXd ErrV = pi_RTk*xcurl.faceOperators(iF).potential*xgrad.buildGradientComponentsFace(iF)*xgrad.restrictFace(iF,uqh) - pi_Gq;

        err(iF) = std::sqrt(ErrV.transpose()*LHS_RTk*ErrV);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestGammaGF(const XGradStokes &xgrad, const XCurlStokes &xcurl,bool expect_zero) {
   Eigen::VectorXd err = Eigen::VectorXd::Zero(xcurl.mesh().n_faces());
  //
  if (xcurl.degree()==0) return 0.;
  parallel_for(xcurl.mesh().n_faces(),
    [&xgrad,&xcurl,&err](size_t start, size_t end)->void
    {
      for (size_t iF = start; iF < end; iF++){
        Eigen::MatrixXd errV = xgrad.faceOperators(iF).gradient - xcurl.faceOperators(iF).potential*xgrad.buildGradientComponentsFace(iF);

        err(iF) = errV.norm();
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

