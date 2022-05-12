// Tests for the local complex property.

#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xgradstokes.hpp>
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
double ComplexEdgeGI(const XGradStokes &xgrad);
double ComplexFaceGI(const XGradStokes &xgrad);
double ComplexCellGI(const XGradStokes &xgrad);

double ComplexEdgeCG(const XGradStokes &xgrad, const XCurlStokes &xcurl);
double ComplexCellCG(const XGradStokes &xgrad, const XCurlStokes &xcurl);

double ComplexCellDC(const XCurlStokes &xcurl, const XNablaStokes & xnabla);

XGradStokes::FunctionType sOne = [](const VectorRd &x)->double {return 1.;};
XGradStokes::FunctionGradType dOne = [](const VectorRd &x)->VectorRd {return VectorRd::Zero();};

template<size_t > int validate_potential();

// Check the complex property on each edge and cells
int main() {
  std::cout << std::endl << "[main] Test with degree 0" << std::endl; 
  validate_potential<0>();
  std::cout << std::endl << "[main] Test with degree 1" << std::endl;
  validate_potential<1>();
  std::cout << std::endl << "[main] Test with degree 2" << std::endl;
  validate_potential<2>();
  std::cout << std::endl << "[main] Test with degree 3" << std::endl;
  validate_potential<3>();
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
  XGradStokes xgrad(stokes_core);
  std::cout << "[main] XGradStokes constructed" << std::endl;

  // Create discrete space XCurlStokes
  XCurlStokes xcurl(stokes_core);
  std::cout << "[main] XCurlStokes constructed" << std::endl;

  // Create discrete space XNablaStokes
  XNablaStokes xnabla(stokes_core);
  std::cout << "[main] XNablaStokes constructed" << std::endl;

  // Test 1 : GI
  std::cout << "[main] Begining of test GI" << std::endl;
  std::cout << "E :" << ComplexEdgeGI(xgrad) << endls;
  std::cout << "F :" << ComplexCellGI(xgrad) << endls;
  std::cout << "T :" << ComplexCellGI(xgrad) << endls;
  
  // Test 2 : CG
  std::cout << "[main] Begining of test CG" << std::endl;
  std::cout << "E :" << ComplexEdgeCG(xgrad,xcurl) << endls;
  std::cout << "T :" << ComplexCellCG(xgrad,xcurl) << endls;
  
  // Test 2 : DC
  std::cout << "[main] Begining of test DC" << std::endl;
  std::cout << "T :" << ComplexCellDC(xcurl,xnabla) << endls;
  return 0;
}
  
double ComplexEdgeGI(const XGradStokes &xgrad) {
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xgrad.mesh().n_edges());
  Eigen::VectorXd Iv = xgrad.interpolate(sOne,dOne);
  std::function<void(size_t,size_t)> compute_local_ops = [&xgrad,&Iv,&local_ops](size_t start, size_t end)->void {
    for (size_t iE = start; iE < end; iE++) {
      local_ops[iE] = (xgrad.edgeOperators(iE).gradient*xgrad.restrictEdge(iE,Iv)).cwiseAbs().maxCoeff();
    } 
  };
  parallel_for(xgrad.mesh().n_edges(),compute_local_ops,use_threads);
  if (local_ops.maxCoeff() > threshold) nb_errors++;
  return local_ops.maxCoeff();
}
double ComplexFaceGI(const XGradStokes &xgrad) {
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xgrad.mesh().n_faces());
  Eigen::VectorXd Iv = xgrad.interpolate(sOne,dOne);
  std::function<void(size_t,size_t)> compute_local_ops = [&xgrad,&Iv,&local_ops](size_t start, size_t end)->void {
    for (size_t iF = start; iF < end; iF++) {
      local_ops[iF] = (xgrad.buildGradientComponentsFace(iF)*xgrad.restrictFace(iF,Iv)).cwiseAbs().maxCoeff();
    } 
  };
  parallel_for(xgrad.mesh().n_faces(),compute_local_ops,use_threads);
  if (local_ops.maxCoeff() > threshold) nb_errors++;
  return local_ops.maxCoeff();
}
double ComplexCellGI(const XGradStokes &xgrad) {
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xgrad.mesh().n_cells());
  Eigen::VectorXd Iv = xgrad.interpolate(sOne,dOne);
  std::function<void(size_t,size_t)> compute_local_ops = [&xgrad,&Iv,&local_ops](size_t start, size_t end)->void {
    for (size_t iT = start; iT < end; iT++) {
      local_ops[iT] = (xgrad.buildGradientComponentsCell(iT)*xgrad.restrictCell(iT,Iv)).cwiseAbs().maxCoeff();
    } 
  };
  parallel_for(xgrad.mesh().n_cells(),compute_local_ops,use_threads);
  if (local_ops.maxCoeff() > threshold) nb_errors++;
  return local_ops.maxCoeff();
}

double ComplexEdgeCG(const XGradStokes &xgrad, const XCurlStokes &xcurl){
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xcurl.mesh().n_edges());
  std::function<void(size_t,size_t)> compute_local_ops = [&xgrad,&xcurl,&local_ops](size_t start, size_t end)->void {
    for (size_t iE = start; iE < end; iE++) {
      local_ops[iE] = (xcurl.edgeOperators(iE).curl*xgrad.edgeOperators(iE).gradient).cwiseAbs().maxCoeff();
    } 
  };
  parallel_for(xgrad.mesh().n_edges(),compute_local_ops,use_threads);
  if (local_ops.maxCoeff() > threshold) nb_errors++;
  return local_ops.maxCoeff();
}
double ComplexCellCG(const XGradStokes &xgrad, const XCurlStokes &xcurl){
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  std::function<void(size_t,size_t)> compute_local_ops = [&xgrad,&xcurl,&local_ops](size_t start, size_t end)->void {
    for (size_t iT = start; iT < end; iT++) {
      local_ops[iT] = (xcurl.buildCurlComponentsCell(iT)*xgrad.buildGradientComponentsCell(iT)).cwiseAbs().maxCoeff();
    } 
  };
  parallel_for(xcurl.mesh().n_cells(),compute_local_ops,use_threads);
  if (local_ops.maxCoeff() > threshold) nb_errors++;
  return local_ops.maxCoeff();
}

double ComplexCellDC(const XCurlStokes &xcurl, const XNablaStokes & xnabla){
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  std::function<void(size_t,size_t)> compute_local_ops = [&xcurl,&xnabla,&local_ops](size_t start, size_t end)->void {
    for (size_t iT = start; iT < end; iT++) {
      local_ops[iT] = (xnabla.cellOperators(iT).divergence*xcurl.buildCurlComponentsCell(iT)).cwiseAbs().maxCoeff();
    } 
  };
  parallel_for(xcurl.mesh().n_cells(),compute_local_ops,use_threads);
  if (local_ops.maxCoeff() > threshold) nb_errors++;
  return local_ops.maxCoeff();
}

