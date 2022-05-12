// Tests for the exact complex property.
// Compute nullspace of operators

#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xgradstokes.hpp>
#include <xcurlstokes.hpp>
#include <xnablastokes.hpp>
#include <xvlstokes.hpp>
#include "testfunction.hpp"

//#include <Eigen/SPQRSupport>

#include <parallel_for.hpp>

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore3D;

const std::string mesh_file = "./meshes/" "Cubic-Cells/RF_fmt/gcube_2x2x2";
//const std::string mesh_file = "./meshes/" "Prysmatic-Cells-1/RF_fmt/gdual_5x5x5";
XGradStokes::FunctionType sOne = [](const VectorRd &x)->double {return 1.;};
XGradStokes::FunctionGradType dOne = [](const VectorRd &x)->VectorRd {return VectorRd::Zero();};

constexpr bool use_threads = true;
// Foward declare
Eigen::SparseMatrix<double> assemble_system_grad(const XGradStokes &xgrad,const XCurlStokes & xcurl); 
Eigen::SparseMatrix<double> assemble_system_curl(const XGradStokes &xgrad,const XCurlStokes & xcurl, const XNablaStokes & xnabla);
Eigen::SparseMatrix<double> assemble_system_div(const XCurlStokes & xcurl, const XNablaStokes & xnabla, const XSLStokes & xsl);

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
  std::cout << std::endl << "[main] Test with degree 4" << std::endl;
  validate_potential<4>();
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

  // Create discrete space XVLStokes
  XSLStokes xsl(stokes_core);
  std::cout << "[main] XVLStokes constructed" << std::endl;

  // Test 1 : Ker G
  Eigen::SparseMatrix<double> SystemGrad = assemble_system_grad(xgrad,xcurl);
  std::cout << "[main] Grad system assembled" << std::endl;
  SystemGrad.makeCompressed();
  //Eigen::SPQR<Eigen::SparseMatrix<double>> solverGrad;
  Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<Eigen::SparseMatrix<double>::StorageIndex>> solverGrad;
  solverGrad.compute(SystemGrad);
  std::cout << "[main] solver Grad system : " << solverGrad.info() << std::endl;
  std::cout << "Kernel of dimension 1 expected" << std::endl;
  std::cout << "Dimension : " << xgrad.dimension() << " rank : " << solverGrad.rank() << std::endl;

  // Test 2 : Ker C
  Eigen::SparseMatrix<double> SystemCurl = assemble_system_curl(xgrad,xcurl,xnabla);
  std::cout << "[main] Curl system assembled" << std::endl;
  SystemCurl.makeCompressed();
  //Eigen::SPQR<Eigen::SparseMatrix<double>> solverCurl;
  Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<Eigen::SparseMatrix<double>::StorageIndex>> solverCurl;
  solverCurl.compute(SystemCurl);
  std::cout << "[main] solver Curl system : " << solverCurl.info() << std::endl;
  std::cout << "Kernel of dimension 0 expected" << std::endl;
  std::cout << "Dimension : " << xgrad.dimension() + xcurl.dimension() << " rank : " << solverCurl.rank() << std::endl;

  // Test 3 : Ker D
  Eigen::SparseMatrix<double> SystemDiv = assemble_system_div(xcurl,xnabla,xsl);
  std::cout << "[main] Div system assembled" << std::endl;
  SystemDiv.makeCompressed();
  //Eigen::SPQR<Eigen::SparseMatrix<double>> solverDiv;
  Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<Eigen::SparseMatrix<double>::StorageIndex>> solverDiv;
  solverDiv.compute(SystemDiv);
  std::cout << "[main] solver Div system : " << solverDiv.info() << std::endl;
  std::cout << "Kernel of dimension 0 expected" << std::endl;
  std::cout << "Dimension : " << xcurl.dimension() + xnabla.dimension() << " rank : " << solverDiv.rank() << std::endl;

  return 0;
}

// We use the conponents-wise norm on Hcurl, hence the matrix is block-diagonal
// Construct (Gp,Gq)
Eigen::SparseMatrix<double> assemble_system_grad(const XGradStokes &xgrad,const XCurlStokes & xcurl) 
{

  std::function<void(size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)> batch_local_assembly = [&xgrad,&xcurl](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)->void {
    for (size_t iT = start; iT < end; iT++) {
      // Compute uGT
      Eigen::MatrixXd uGT = xgrad.buildGradientComponentsCell(iT);
      const Cell & T = *xgrad.mesh().cell(iT);
      std::vector<size_t> dofmap_xgrad = xgrad.globalDOFIndices(T);
      size_t Loc_offset_xcurl = 0; 
      size_t dim_V_curl = xcurl.numLocalDofsVertex();
      size_t dim_E_curl = xcurl.numLocalDofsEdge();
      size_t dim_F_curl = xcurl.numLocalDofsFace();
      size_t dim_T_curl = xcurl.numLocalDofsCell();
      // Itterate over each vertex
      for (size_t iV = 0; iV < T.n_vertices();iV++) {
        double scale = T.diam()*T.diam()*T.diam();
        Eigen::MatrixXd LocOp = uGT.middleRows(Loc_offset_xcurl,dim_V_curl);
        Loc_offset_xcurl += dim_V_curl;
        Eigen::MatrixXd ALoc = scale*LocOp.transpose()*xcurl.computeL2opnVertex(T.diam())*LocOp;
        // update triplet
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < xgrad.dimensionCell(iT);j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dofmap_xgrad[j],ALoc(i,j));
          }
        }
      }
      // Itterate over each edge
      for (size_t iE = 0; iE < T.n_edges();iE++) {
        double scale = T.diam()*T.diam();
        Eigen::MatrixXd LocOp = uGT.middleRows(Loc_offset_xcurl,dim_E_curl);
        Loc_offset_xcurl += dim_E_curl;
        Eigen::MatrixXd ALoc = scale*LocOp.transpose()*xcurl.computeL2opnEdge(T.edge(iE)->global_index())*LocOp;
        // update triplet
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < xgrad.dimensionCell(iT);j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dofmap_xgrad[j],ALoc(i,j));
          }
        }
      }
      // Itterate over each face
      for (size_t iF = 0; iF < T.n_faces();iF++) {
        double scale = T.diam();
        Eigen::MatrixXd LocOp = uGT.middleRows(Loc_offset_xcurl,dim_F_curl);
        Loc_offset_xcurl += dim_F_curl;
        Eigen::MatrixXd ALoc = scale*LocOp.transpose()*xcurl.computeL2opnFace(T.face(iF)->global_index())*LocOp;
        // update triplet
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < xgrad.dimensionCell(iT);j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dofmap_xgrad[j],ALoc(i,j));
          }
        }
      }
      {
        // Cell unknowns
        Eigen::MatrixXd LocOp = uGT.middleRows(Loc_offset_xcurl,dim_T_curl);
        Loc_offset_xcurl += dim_T_curl;
        Eigen::MatrixXd ALoc = LocOp.transpose()*xcurl.computeL2opnCell(iT)*LocOp;
        // update triplet
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < xgrad.dimensionCell(iT);j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dofmap_xgrad[j],ALoc(i,j));
          }
        }
      }
    }
  };

  return parallel_assembly_system(xcurl.mesh().n_cells(),xgrad.dimension(),batch_local_assembly,false).first;
}

// We use the conponents-wise norm on Hcurl, hence the matrix is block-diagonal
// Construct (Cu,Cv) + (Gp,v) + (u,Gq) + (p,q)
Eigen::SparseMatrix<double> assemble_system_curl(const XGradStokes &xgrad,const XCurlStokes & xcurl, const XNablaStokes & xnabla) 
{

  std::function<void(size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)> batch_local_assembly = [&xgrad,&xcurl,&xnabla](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)->void {
    for (size_t iT = start; iT < end; iT++) {
      //---------------------------------------------------------------------------------------------------
      // Off diagonal part
      // dofs : Xgrad,Xcurl
      // Compute uGT
      Eigen::MatrixXd uGT = xgrad.buildGradientComponentsCell(iT);
      const Cell & T = *xgrad.mesh().cell(iT);
      std::vector<size_t> dofmap_xgrad = xgrad.globalDOFIndices(T);
      size_t dim_xgrad = xgrad.dimension();
      std::vector<size_t> dofmap_xcurl = xcurl.globalDOFIndices(T);
      size_t Loc_offset_xcurl = 0; 
      size_t dim_V_curl = xcurl.numLocalDofsVertex();
      size_t dim_E_curl = xcurl.numLocalDofsEdge();
      size_t dim_F_curl = xcurl.numLocalDofsFace();
      size_t dim_T_curl = xcurl.numLocalDofsCell();
      // Itterate over each vertex
      for (size_t iV = 0; iV < T.n_vertices();iV++) {
        double scale = T.diam()*T.diam()*T.diam();
        Eigen::MatrixXd LocOp = uGT.middleRows(Loc_offset_xcurl,dim_V_curl);
        Eigen::MatrixXd ALoc = scale*LocOp.transpose()*xcurl.computeL2opnVertex(T.diam());
        // update triplet
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < dim_V_curl;j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dim_xgrad+dofmap_xcurl[Loc_offset_xcurl+j],ALoc(i,j)); // (Gp,v)
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dim_xgrad+dofmap_xcurl[Loc_offset_xcurl+j],dofmap_xgrad[i],ALoc(i,j)); // (u,Gq)
          }
        }
        Loc_offset_xcurl += dim_V_curl;
      }
      // Itterate over each edge
      for (size_t iE = 0; iE < T.n_edges();iE++) {
        double scale = T.diam()*T.diam();
        Eigen::MatrixXd LocOp = uGT.middleRows(Loc_offset_xcurl,dim_E_curl);
        Eigen::MatrixXd ALoc = scale*LocOp.transpose()*xcurl.computeL2opnEdge(T.edge(iE)->global_index());
        // update triplet
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < dim_E_curl;j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dim_xgrad+dofmap_xcurl[Loc_offset_xcurl+j],ALoc(i,j)); // (Gp,v)
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dim_xgrad+dofmap_xcurl[Loc_offset_xcurl+j],dofmap_xgrad[i],ALoc(i,j)); // (u,Gq)
          }
        }
        Loc_offset_xcurl += dim_E_curl;
      }
      // Itterate over each face
      for (size_t iF = 0; iF < T.n_faces();iF++) {
        double scale = T.diam();
        Eigen::MatrixXd LocOp = uGT.middleRows(Loc_offset_xcurl,dim_F_curl);
        Eigen::MatrixXd ALoc = scale*LocOp.transpose()*xcurl.computeL2opnFace(T.face(iF)->global_index());
        // update triplet
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < dim_F_curl;j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dim_xgrad+dofmap_xcurl[Loc_offset_xcurl+j],ALoc(i,j)); // (Gp,v)
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dim_xgrad+dofmap_xcurl[Loc_offset_xcurl+j],dofmap_xgrad[i],ALoc(i,j)); // (u,Gq)
          }
        }
        Loc_offset_xcurl += dim_F_curl;
      }
      {
        // Cell unknowns
        Eigen::MatrixXd LocOp = uGT.middleRows(Loc_offset_xcurl,dim_T_curl);
        Eigen::MatrixXd ALoc = LocOp.transpose()*xcurl.computeL2opnCell(iT);
        // update triplet
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < dim_T_curl;j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dim_xgrad+dofmap_xcurl[Loc_offset_xcurl+j],ALoc(i,j)); // (Gp,v)
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dim_xgrad+dofmap_xcurl[Loc_offset_xcurl+j],dofmap_xgrad[i],ALoc(i,j)); // (u,Gq)
          }
        }
        Loc_offset_xcurl += dim_T_curl;
      }
      //---------------------------------------------------------------------------------------------------
      // (p,q)
      {
        Eigen::MatrixXd ALoc = xgrad.computeL2Product(iT);
        for (size_t i = 0; i < xgrad.dimensionCell(iT);i++) {
          for (size_t j = 0; j < xgrad.dimensionCell(iT);j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xgrad[i],dofmap_xgrad[i],ALoc(i,j)); // (p,q)
          }
        }
      }
      //---------------------------------------------------------------------------------------------------
      // (Cu,Cv)
      {
        Eigen::MatrixXd LocOp = xcurl.buildCurlComponentsCell(iT);
        Eigen::MatrixXd ALoc = LocOp.transpose()*xnabla.computeL2Product(iT)*LocOp;
        for (size_t i = 0; i < xcurl.dimensionCell(iT);i++) {
          for (size_t j = 0; j < xcurl.dimensionCell(iT);j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dim_xgrad+dofmap_xcurl[i],dim_xgrad+dofmap_xcurl[j],ALoc(i,j)); // (Cu,Cv)
          }
        }
      }
    }// for iT
  };

  return parallel_assembly_system(xcurl.mesh().n_cells(),xgrad.dimension()+xcurl.dimension(),batch_local_assembly,true).first;
}
  
// We use the conponents-wise norm on Hcurl, hence the matrix is block-diagonal
// Construct (Du,Dv) + (Cp,v) + (u,Cq) + (p,q)
Eigen::SparseMatrix<double> assemble_system_div(const XCurlStokes & xcurl, const XNablaStokes & xnabla, const XSLStokes & xsl) 
{

  std::function<void(size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)> batch_local_assembly = [&xcurl,&xnabla,&xsl](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)->void {
    for (size_t iT = start; iT < end; iT++) {
      //---------------------------------------------------------------------------------------------------
      // (p,q) in Xcurl
      // dofs : Xcurl,Xnabla
      // Compute uGT
      const Cell & T = *xcurl.mesh().cell(iT);
      std::vector<size_t> dofmap_xcurl = xcurl.globalDOFIndices(T);
      size_t dim_xcurl = xcurl.dimension();
      std::vector<size_t> dofmap_xnabla = xnabla.globalDOFIndices(T);
      size_t Loc_offset_xcurl = 0; 
      size_t dim_V_curl = xcurl.numLocalDofsVertex();
      size_t dim_E_curl = xcurl.numLocalDofsEdge();
      size_t dim_F_curl = xcurl.numLocalDofsFace();
      size_t dim_T_curl = xcurl.numLocalDofsCell();
      // Itterate over each vertex
      for (size_t iV = 0; iV < T.n_vertices();iV++) {
        double scale = T.diam()*T.diam()*T.diam();
        Eigen::MatrixXd ALoc = scale*xcurl.computeL2opnVertex(T.diam());
        // update triplet
        for (size_t i = 0; i < dim_V_curl;i++) {
          for (size_t j = 0; j < dim_V_curl;j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xcurl[Loc_offset_xcurl+i],dofmap_xcurl[Loc_offset_xcurl+j],ALoc(i,j)); 
          }
        }
        Loc_offset_xcurl += dim_V_curl;
      }
      // Itterate over each edge
      for (size_t iE = 0; iE < T.n_edges();iE++) {
        double scale = T.diam()*T.diam();
        Eigen::MatrixXd ALoc = scale*xcurl.computeL2opnEdge(T.edge(iE)->global_index());
        // update triplet
        for (size_t i = 0; i < dim_E_curl;i++) {
          for (size_t j = 0; j < dim_E_curl;j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xcurl[Loc_offset_xcurl+i],dofmap_xcurl[Loc_offset_xcurl+j],ALoc(i,j)); 
          }
        }
        Loc_offset_xcurl += dim_E_curl;
      }
      // Itterate over each face
      for (size_t iF = 0; iF < T.n_faces();iF++) {
        double scale = T.diam();
        Eigen::MatrixXd ALoc = scale*xcurl.computeL2opnFace(T.face(iF)->global_index());
        // update triplet
        for (size_t i = 0; i < dim_F_curl;i++) {
          for (size_t j = 0; j < dim_F_curl;j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xcurl[Loc_offset_xcurl+i],dofmap_xcurl[Loc_offset_xcurl+j],ALoc(i,j)); 
          }
        }
        Loc_offset_xcurl += dim_F_curl;
      }
      {
        // Cell unknowns
        Eigen::MatrixXd ALoc = xcurl.computeL2opnCell(iT);
        // update triplet
        for (size_t i = 0; i < dim_T_curl;i++) {
          for (size_t j = 0; j < dim_T_curl;j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xcurl[Loc_offset_xcurl+i],dofmap_xcurl[Loc_offset_xcurl+j],ALoc(i,j)); 
          }
        }
        Loc_offset_xcurl += dim_T_curl;
      }
      //---------------------------------------------------------------------------------------------------
      // (Cp,q) + (p,Cq)
      {
        Eigen::MatrixXd ALoc = xcurl.buildCurlComponentsCell(iT).transpose()*xnabla.computeL2Product(iT);
        for (size_t i = 0; i < xcurl.dimensionCell(iT);i++) {
          for (size_t j = 0; j < xnabla.dimensionCell(iT);j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dofmap_xcurl[i],dim_xcurl+dofmap_xnabla[j],ALoc(i,j)); // (Cp,q)
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dim_xcurl+dofmap_xnabla[j],dofmap_xcurl[i],ALoc(i,j)); // (p,Cq)
          }
        }
      }
      //---------------------------------------------------------------------------------------------------
      // (Cu,Cv)
      {
        Eigen::MatrixXd LocOp = xnabla.cellOperators(iT).divergence;
        Eigen::MatrixXd ALoc = LocOp.transpose()*xsl.compute_Gram_Cell(iT)*LocOp;
        for (size_t i = 0; i < xnabla.dimensionCell(iT);i++) {
          for (size_t j = 0; j < xnabla.dimensionCell(iT);j++) {
            if (std::abs(ALoc(i,j))>1.e-16) 
            triplets->emplace_back(dim_xcurl+dofmap_xnabla[i],dim_xcurl+dofmap_xnabla[j],ALoc(i,j)); // (Cu,Cv)
          }
        }
      }
    }// for iT
  };

  return parallel_assembly_system(xcurl.mesh().n_cells(),xcurl.dimension()+xnabla.dimension(),batch_local_assembly,true).first;
}

