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
template<typename T> double TestStNa(const XNablaStokes &xnabla,T &v,bool = true);
template<typename T> double TestStVl(const XVLStokes &xvl,T &v,bool = true);

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
XVLStokes::FunctionType FormalFunctionMat(T &v) {
  return [&v](const VectorRd &x)->Eigen::Matrix3d {
    Eigen::Matrix3d rv;
    rv(0,0) = v[0].evaluate(x,0,0,0);
    rv(0,1) = v[1].evaluate(x,0,0,0);
    rv(0,2) = v[2].evaluate(x,0,0,0);
    rv(1,0) = v[3].evaluate(x,0,0,0);
    rv(1,1) = v[4].evaluate(x,0,0,0);
    rv(1,2) = v[5].evaluate(x,0,0,0);
    rv(2,0) = v[6].evaluate(x,0,0,0);
    rv(2,1) = v[7].evaluate(x,0,0,0);
    rv(2,2) = v[8].evaluate(x,0,0,0);
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


  // Create test functions
  std::vector<PolyTest<degree>> Pkx{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 1>> Pkpox{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 2>> Pkp2x{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree + 3>> Pkp3x{Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<TrigTest<degree>> Ttrigx{Initialization::Random,Initialization::Random,Initialization::Random};
  // Create test matrix valued functions
  std::vector<PolyTest<degree>> PkxM{Initialization::Random,Initialization::Random,Initialization::Random,
                                     Initialization::Random,Initialization::Random,Initialization::Random,
                                     Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree+1>> PkpoxM{Initialization::Random,Initialization::Random,Initialization::Random,
                                       Initialization::Random,Initialization::Random,Initialization::Random,
                                       Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<PolyTest<degree+2>> Pkp2xM{Initialization::Random,Initialization::Random,Initialization::Random,
                                       Initialization::Random,Initialization::Random,Initialization::Random,
                                       Initialization::Random,Initialization::Random,Initialization::Random};
  std::vector<TrigTest<degree>> TtrigxM{Initialization::Random,Initialization::Random,Initialization::Random,
                                     Initialization::Random,Initialization::Random,Initialization::Random,
                                     Initialization::Random,Initialization::Random,Initialization::Random};
  
  // Test 
  // Machine epsilon at 10-16, we return the squared norm
  std::cout << "[main] Begining of test StNa" << std::endl;
  std::cout << "We expected zero up to k+1" << std::endl;
  std::cout << "Error for Pk :"<< TestStNa(xnabla, Pkx) << endls;
  std::cout << "Error for Pkpo :"<< TestStNa(xnabla, Pkpox) << endls;
  std::cout << "Error for Pkp2 :"<< TestStNa(xnabla, Pkp2x,false) << endls;
  std::cout << "Error for Pkp3 :"<< TestStNa(xnabla, Pkp3x,false) << endls;
  std::cout << "Error for Ttrig :"<< TestStNa(xnabla, Ttrigx,false) << endls;

  std::cout << "[main] Begining of test StVl" << std::endl;
  std::cout << "We expected zero up to k" << std::endl;
  std::cout << "Error for Pk :"<< TestStVl(xvl, PkxM) << endls;
  std::cout << "Error for Pkpo :"<< TestStVl(xvl, PkpoxM,false) << endls;
  std::cout << "Error for Pkp2 :"<< TestStVl(xvl, Pkp2xM,false) << endls;
  std::cout << "Error for Ttrig :"<< TestStVl(xvl, TtrigxM,false) << endls;

  return 0;
}

template<typename Type> double TestStNa(const XNablaStokes &xnabla, Type &v,bool expect_zero) {
  // Compute interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(FormalFunction(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  // 
  parallel_for(xnabla.mesh().n_cells(),
    [&xnabla, &Iv, &err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        Eigen::VectorXd IvT = xnabla.restrictCell(iT,Iv);
        // Stabilization
        Eigen::MatrixXd L2 = xnabla.computeL2Product(iT,1.,Eigen::MatrixXd::Zero(xnabla.cellBases(0).Polyk3po->dimension(),xnabla.cellBases(0).Polyk3po->dimension())); // remove the cell component with zero
        err(iT) = std::abs(IvT.transpose()*L2*IvT);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

template<typename Type> double TestStVl(const XVLStokes &xvl, Type &v,bool expect_zero) {
  // Compute interpolate
  Eigen::VectorXd Iv = xvl.interpolate(FormalFunctionMat(v));
  Eigen::VectorXd err = Eigen::VectorXd::Zero(xvl.mesh().n_cells());
  // 
  parallel_for(xvl.mesh().n_cells(),
    [&xvl, &Iv, &err](size_t start, size_t end)->void
    {
      for (size_t iT = start; iT < end; iT++){
        const Cell & T = *xvl.mesh().cell(iT);
        Eigen::VectorXd IvT = xvl.restrictCell(iT,Iv);
        // Id on Edges and faces
        std::vector<Eigen::MatrixXd> potentialOp(T.n_edges()+T.n_faces()+1);
        for (size_t iE = 0; iE < T.n_edges(); iE++) {
          potentialOp[iE] = xvl.extendOperator(T,*T.edge(iE),Eigen::MatrixXd::Identity(xvl.edgeBases(0).Polyk3p2->dimension(),xvl.edgeBases(0).Polyk3p2->dimension()));
        }
        for (size_t iF = 0; iF < T.n_faces(); iF++) {
          size_t dimF = xvl.numLocalDofsFace();
          size_t dimFE = xvl.dimensionFace(T.face(iF)->global_index());
          Eigen::MatrixXd locF = Eigen::MatrixXd::Zero(dimF,dimFE);
          locF.rightCols(dimF) = Eigen::MatrixXd::Identity(dimF,dimF);
          potentialOp[T.n_edges() + iF] = xvl.extendOperator(T,*T.face(iF),locF);
        }
        potentialOp[T.n_edges() + T.n_faces()] = Eigen::MatrixXd::Zero(xvl.numLocalDofsCell(),xvl.dimensionCell(iT));
        potentialOp[T.n_edges() + T.n_faces()].rightCols(xvl.numLocalDofsCell()) = Eigen::MatrixXd::Identity(xvl.numLocalDofsCell(),xvl.numLocalDofsCell());
        Eigen::MatrixXd L2 = xvl.computeL2Product_with_Ops(iT,potentialOp,potentialOp,1.,Eigen::MatrixXd::Zero(xvl.numLocalDofsCell(),xvl.numLocalDofsCell()),IntegralWeight(1.)); 
        err(iT) = std::abs(IvT.transpose()*L2*IvT);
      }
    },use_threads);
  double rv = err.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

