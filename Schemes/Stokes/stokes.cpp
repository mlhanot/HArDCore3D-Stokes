
#define CHECK_SOLVE

// Memory problems
#undef WITH_UMFPACK
#define ITTERATIVE
//#define ITTERATIVE_LU

#ifndef SOLVE_WITH_PETSC_MPI
#include "stokes.hpp"
#else // SOLVE_WITH_PETSC_MPI
#include "stokes_petsc.hpp"
#endif

#define ACCELERATE_CONVRATE

#include <mesh_builder.hpp>
#include <vtu_writer.hpp>

#include <boost/program_options.hpp>

#include <iomanip>
#include <filesystem>

#include <signal.h>
// Defined to salvage already computed values sending ctrl-c
// Made to interupt the solve, do NOT interrupt the multithreaded part
void signal_callback(int signum){throw std::runtime_error("Interrupted by user");}

using namespace HArDCore3D;

#include "stokes_structs.hpp"

const std::string outdir(STRINGIFY(ROOT_OUTDIR) "/Stokes_corr/");

/// Test parameters
size_t constexpr max_systemsize = 2300000;
bool constexpr write_sols = true;
int SOLN = 3;
int TESTCASE = 4;

// TESTCASE gives the mesh sequences
// SOLN gives the exact solution
std::vector<std::filesystem::path> mesh_files;

Sol_struct exactsol;

//std::function<bool(const VectorRd &)> zNZero = [](const VectorRd &x)->bool {return (x(2) > 1e-7);};
std::function<bool(const VectorRd &)> zNZero = [](const VectorRd &x)->bool {return (x(2) > 1e-7) || (x(1) < 1e-7) || (x(1) > 1. - 1e-7) || (x(0) < 1e-7) || (x(0) > 1. - 1e-7);};
std::function<bool(const VectorRd &)> zZero = [](const VectorRd &x)->bool {return (x(2) < 1e-7);};

template<size_t > void validate_Stokes();

int main(int argc, const char* argv[])
{
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Display this help message")
    ("mesh,m", boost::program_options::value<int>()->default_value(TESTCASE), "The mesh sequence number")
    ("degree,k", boost::program_options::value<int>()->default_value(-1), "The polynomial degree, -1 for all")
    ("solution,s", boost::program_options::value<int>()->default_value(SOLN), "The solution number");
  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout<< desc << std::endl;
    return 0;
  }
  // Print received options to log
  std::cout<<"[main] Solving for solution "<<vm["solution"].as<int>()<<" on mesh seq. "<<vm["mesh"].as<int>()<<std::endl;
  // select mesh 
  TESTCASE = vm["mesh"].as<int>();
  switch(TESTCASE) {
    case (0):
      mesh_files = mesh_files0;
      break;
    case (1):
      mesh_files = mesh_files1;
      break;
    case (2):
      mesh_files = mesh_files2;
      break;
    case (3):
      mesh_files = mesh_files3;
      break;
    case (4):
      mesh_files = mesh_files4;
      break;
    case (5):
      mesh_files = mesh_files5;
      break;
    default:
      std::cerr<<"Mesh sequence "<<TESTCASE<<" not implemented"<<std::endl;
      return 1;
  }
  // select sol
  SOLN = vm["solution"].as<int>();
  switch(SOLN) {
    case (0):
      exactsol = Sol_struct0<1>();
      break;
    case (1):
      exactsol = Sol_struct1<1>();
      break;
    case (2):
      exactsol = Sol_struct2<1>();
      break;
    case (3):
      exactsol = Sol_struct3<1>();
      break;
    case (4):
      exactsol = Sol_struct4<1>();
      break;
    default:
      std::cerr<<"Solution "<<SOLN<<" not implemented"<<std::endl;
      return 1;
  }

  //for (int i = 0; i < _NSIG;i++) {signal(i,signal_callback);} // MPI intercept SIGINT and send something else
  signal(SIGCONT,signal_callback); // Make sure not to register SIGABRT, lest is cause infinite reccursion
  if (vm["degree"].as<int>() < 0 || vm["degree"].as<int>() == 0) {
    std::cout << std::endl << "\033[31m[main] Test with degree 0\033[0m" << std::endl; 
    validate_Stokes<0>();
  }
  if (vm["degree"].as<int>() < 0 || vm["degree"].as<int>() == 1) {
    std::cout << std::endl << "\033[31m[main] Test with degree 1\033[0m" << std::endl;
    validate_Stokes<1>();
  }
  if (vm["degree"].as<int>() < 0 || vm["degree"].as<int>() == 2) {
    std::cout << std::endl << "\033[31m[main] Test with degree 2\033[0m" << std::endl;
    validate_Stokes<2>();
  }
  if (vm["degree"].as<int>() < 0 || vm["degree"].as<int>() == 3) {
    std::cout << std::endl << "\033[31m[main] Test with degree 3\033[0m" << std::endl;
    validate_Stokes<3>();
  }
  PETscSolver_MPI_Finalize();
  return 0;
}

template<size_t degree> 
void validate_Stokes() {
  std::error_code errc;

  std::vector<double> meshsize;
  std::vector<double> meshregularity1;
  std::vector<double> meshregularity2;
  std::vector<double> errorP;
  std::vector<double> normP;
  std::vector<double> errorU;
  std::vector<double> normU;
  std::vector<double> errorT;
  std::vector<double> normT;
  std::vector<std::string> stepnames;
  Eigen::MatrixXd time_hist = Eigen::MatrixXd::Zero(mesh_files.size(),8);

 // Iterate over meshes
  for (size_t i = 0; i < mesh_files.size(); i++) {
    // Build the mesh
    std::filesystem::path mesh_file = mesh_files[i];
    MeshBuilder builder = MeshBuilder(mesh_file);
    std::unique_ptr<Mesh> mesh_ptr = builder.build_the_mesh();
    std::cout << "[main] Mesh size                 " << mesh_ptr->h_max() << std::endl;
    std::cout << "[main] Mesh regularity           " << mesh_ptr->regularity()[0]<<", "<<mesh_ptr->regularity()[1] << std::endl;
    // Store the size of the mesh
    meshsize.emplace_back(mesh_ptr->h_max());
    meshregularity1.emplace_back(mesh_ptr->regularity()[0]);
    meshregularity2.emplace_back(mesh_ptr->regularity()[1]);

    // Create core 
    std::unique_ptr<StokesProblem> stokes_ptr;
    // Check if data already exists
    std::filesystem::path datafile = outdir + std::string("/raw_internal/") + std::to_string(TESTCASE) + std::string("_") + mesh_file.filename().string() + std::string("_k") + std::to_string(degree) + std::string(".xnabla");
    if (std::filesystem::exists(datafile)) {
      stokes_ptr.reset(new StokesProblem(*mesh_ptr,degree,datafile.string()));
    } else {
      std::filesystem::create_directories(datafile.parent_path());
      stokes_ptr.reset(new StokesProblem(*mesh_ptr,degree));
      stokes_ptr->xnabla().Write_internal(datafile.string());
    } 

    StokesProblem &stokes = *stokes_ptr;
    std::cout << "[main] StokesProblem constructed" << std::endl;

    // Setup boundary conditions
    if (SOLN <= 2) {
      stokes.setup_Harmonics(Harmonics_premade::Pressure);
      stokes.setup_Dirichlet_everywhere();
      stokes.interpolate_boundary_value(exactsol.u_exact);
    } else if (SOLN == 3 || SOLN == 4) {
      stokes.setup_Harmonics(Harmonics_premade::None);
      stokes.set_Dirichlet_boundary(zNZero);
      stokes.interpolate_boundary_value(exactsol.u_exact);
    } else {
      throw std::runtime_error("Solution not implemented");
    }
    
    std::cerr<<"[main] Systemsize: "<<stokes.systemEffectiveDim()<<std::endl;
    if (stokes.systemEffectiveDim() > max_systemsize) {
      std::cout<<"[main] System size too large: "<<stokes.systemEffectiveDim()<<" > "<<max_systemsize<<" Skipping"<<std::endl;
      break;
    }
    
    // Create problem and solve
    try {
      stokes.assemble_system(exactsol.f);
    } catch (...) {
      std::cout<<"[main] Assembly failed, writting available solutions to files"<<std::endl;
      break;
    }
    if (SOLN == 4) {
      stokes.set_neumann(zZero,exactsol.u_N);
    }

    // Interpolate exact solution
    Eigen::VectorXd Iu = stokes.xnabla().interpolate(exactsol.u_exact);
    Eigen::VectorXd Ip = stokes.xsl().interpolate(exactsol.p_exact);
    // Store their discrete norms
    double NormE = Norm_H1p(stokes.xnabla(),stokes.xvl(),Iu);
    normU.emplace_back(NormE);
    normT.emplace_back(NormE);
    NormE = Norm_L2s(stokes.xsl(),Ip);
    normP.emplace_back(NormE);
    normT.back() += NormE;

    Eigen::VectorXd uhunk;
    try {
      stokes.compute();
      #ifdef ACCELERATE_CONVRATE
      Eigen::VectorXd solvec = Eigen::VectorXd::Zero(stokes.systemTotalDim());
      solvec.head(stokes.xnabla().dimension()) = Iu;
      solvec.tail(stokes.xsl().dimension()) = Ip;
      uhunk = stokes.solve_with_guess(solvec);
      #else
      uhunk = stokes.solve();
      #endif
    } catch (...) {
      std::cout<<"[main] Solver failed, writting available solutions to files"<<std::endl;
      break;
    }
    std::cout << "[main] System solved" << std::endl;
    // Reinsert Dirichlet BC into the vector, unnecessary copy when there is no Dirichlet BC
    Eigen::VectorXd uh = stokes.reinsertDirichlet(uhunk);

    // Store the difference
    Iu -= uh.segment(0,stokes.xnabla().dimension());
    Ip -= uh.segment(stokes.xnabla().dimension(),stokes.xsl().dimension());

    // Write solutions
    if (write_sols) {
      VtuWriter writer(mesh_ptr.get());
      std::cout << "[main] Writing solutions..." << std::flush;
      Eigen::VectorXd uh_vert = get_vertices_values(stokes.xnabla(),uh.head(stokes.xnabla().dimension()));
      Eigen::VectorXd ph_vert = get_vertices_values(stokes.xsl(),uh.segment(stokes.xnabla().dimension(),stokes.xsl().dimension()));
      Eigen::VectorXd u_vert = evaluate_vertices_values(exactsol.u_exact,stokes.stokescore());
      Eigen::VectorXd p_vert = evaluate_vertices_values(exactsol.p_exact,stokes.stokescore());
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<3> > uhx(uh_vert.data(), uh_vert.size()/3);
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<3> > uhy(uh_vert.data() + 1, uh_vert.size()/3);
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<3> > uhz(uh_vert.data() + 2, uh_vert.size()/3);
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<3> > ux(u_vert.data(), u_vert.size()/3);
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<3> > uy(u_vert.data() + 1, u_vert.size()/3);
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<3> > uz(u_vert.data() + 2, u_vert.size()/3);

      // export_path
      std::string meshname = mesh_file.parent_path().parent_path().filename().string() + std::string("_") + mesh_file.filename().string(); // meshes/Cubic-Cells/RF_fmt/gcube_2x2x2 -> Cubic-Cells_gcube_2x2x2
      std::filesystem::path outfile = outdir + std::string("vtu/") + std::to_string(TESTCASE) + std::string("/sol_") + std::to_string(SOLN) + std::string("/");
      // write uh
      std::filesystem::create_directories(outfile,errc);
      if (errc) {
        std::cout << "Warning : Could not create folder :"<<outfile<<std::endl;
      } else {
        writer.write_to_vtu(outfile.string() + meshname + std::string("_uhx_") + std::to_string(degree) + std::string(".vtu"),uhx);
        writer.write_to_vtu(outfile.string() + meshname + std::string("_uhy_") + std::to_string(degree) + std::string(".vtu"),uhy);
        writer.write_to_vtu(outfile.string() + meshname + std::string("_uhz_") + std::to_string(degree) + std::string(".vtu"),uhz);
        writer.write_to_vtu(outfile.string() + meshname + std::string("_ph_") + std::to_string(degree) + std::string(".vtu"),ph_vert);
        writer.write_to_vtu(outfile.string() + meshname + std::string("_ux_") + std::to_string(degree) + std::string(".vtu"),ux);
        writer.write_to_vtu(outfile.string() + meshname + std::string("_uy_") + std::to_string(degree) + std::string(".vtu"),uy);
        writer.write_to_vtu(outfile.string() + meshname + std::string("_uz_") + std::to_string(degree) + std::string(".vtu"),uz);
        writer.write_to_vtu(outfile.string() + meshname + std::string("_p_") + std::to_string(degree) + std::string(".vtu"),p_vert);
        std::cout <<" Done" << std::endl;
      }
    }
    double Errnorm = Norm_H1p(stokes.xnabla(),stokes.xvl(),Iu);
    errorU.emplace_back(Errnorm);
    errorT.emplace_back(Errnorm);
    Errnorm = Norm_L2s(stokes.xsl(),Ip);
    errorP.emplace_back(Errnorm);
    errorT.back() += Errnorm;

    // Store times
    time_hist(i,0) = stokes.m_timer_hist.wtimes[0];
    time_hist(i,1) = stokes.m_timer_hist.ptimes[0];
    time_hist(i,2) = stokes.m_timer_hist.wtimes[1];
    time_hist(i,3) = stokes.m_timer_hist.ptimes[1];
    time_hist(i,4) = stokes.m_timer_hist.wtimes[2];
    time_hist(i,5) = stokes.m_timer_hist.ptimes[2];
    time_hist(i,6) = stokes.m_timer_hist.wtimes[3];
    time_hist(i,7) = stokes.m_timer_hist.ptimes[3];
    stepnames = stokes.m_timer_hist.ntimes; // We hardcode the 4 times in printing
  } // end for meshes
  
  std::cout << "Absolute   ErrorU   ErrorP   ErrorT" << std::endl;
  for (size_t i = 0; i < errorT.size();i++) {std::cout<<"          "<<FORMATD(2)<<errorU[i]<<errorP[i]<<errorT[i]<<std::endl;}
  std::cout << "Rate" << std::endl;
  for (size_t i = 1; i < errorT.size();i++) {std::cout<<"          "<<FORMATD(2)<<compute_rate(errorU,meshsize,i)<<compute_rate(errorP,meshsize,i)<<compute_rate(errorT,meshsize,i)<<std::endl;}

  // Remplace absolute by relative
  for (size_t i = 0; i < errorT.size();i++) {errorU[i] /= normU[i];errorP[i] /= normP[i];errorT[i] /= normT[i];}
  std::cout << "Relative   ErrorU   ErrorP   ErrorT" << std::endl;
  for (size_t i = 0; i < errorT.size();i++) {std::cout<<"          "<<FORMATD(2)<<errorU[i]<<errorP[i]<<errorT[i]<<std::endl;}
  std::cout << "Rate" << std::endl;
  for (size_t i = 1; i < errorT.size();i++) {std::cout<<"          "<<FORMATD(2)<<compute_rate(errorU,meshsize,i)<<compute_rate(errorP,meshsize,i)<<compute_rate(errorT,meshsize,i)<<std::endl;}
  // output data in files
  std::fstream fs;
  std::filesystem::path outfile = outdir + std::string("/meshtype_") + std::to_string(TESTCASE) + std::string("/sol_") + std::to_string(SOLN) + std::string("_k") + std::to_string(degree) + std::string(".dat");
  std::filesystem::create_directories(outfile.parent_path(),errc);
  if (errc) {
    std::cout << "Warning : Could not create folder :"<<outfile.parent_path()<<std::endl;
  }

  fs.open(outfile, std::ios::out);
  fs << "meshsize \terr"<<std::endl;
  for (size_t i = 0; i < errorT.size();i++){
    fs << meshsize[i] << "\t" << errorT[i] << std::endl;
  }
  fs.close();

  // Store detailed error
  outfile = outdir + std::string("/meshtype_") + std::to_string(TESTCASE) + std::string("/sol_") + std::to_string(SOLN) + std::string("_k") + std::to_string(degree) + std::string("_detailed.dat");
  fs.open(outfile, std::ios::out);
  fs << "meshsize \terrU \terrP \tnormU \tnormP"<<std::endl;
  for (size_t i = 0; i < errorT.size();i++){
    fs << meshsize[i] << "\t" << errorU[i] << "\t" << errorP[i] << "\t" << normU[i] << "\t" << normP[i] << std::endl;
  }
  fs.close();

   // Store times
  outfile = outdir + std::string("/meshtype_") + std::to_string(TESTCASE) + std::string("/sol_") + std::to_string(SOLN) + std::string("_k") + std::to_string(degree) + std::string("_timing.dat");
  fs.open(outfile, std::ios::out);
  if (stepnames.size() > 3)
    fs << stepnames[0] <<"\t\t"<<stepnames[1]<<"\t\t"<<stepnames[2]<<"\t\t"<<stepnames[3]<<std::endl;
  fs << "wall: \tcpu: \t wall: \tcpu: \twall \tcpu: \twall: \tcpu:"<<std::endl;
  for (size_t i = 0; i < errorT.size();i++){
    fs << time_hist(i,0) << "\t";
    fs << time_hist(i,1) << "\t";
    fs << time_hist(i,2) << "\t"; 
    fs << time_hist(i,3) << "\t";
    fs << time_hist(i,4) << "\t"; 
    fs << time_hist(i,5) << "\t"; 
    fs << time_hist(i,6) << "\t"; 
    fs << time_hist(i,7) << std::endl;
  }
  fs.close();

  // Store mesh regularity
  outfile = outdir + std::string("/meshtype_") + std::to_string(TESTCASE) + std::string("/meshreg.dat") + std::to_string(degree); // prevent overwrite
  fs.open(outfile, std::ios::out);
  fs << "meshreg1 \tmeshreg2"<<std::endl;
  for (size_t i = 0; i < errorT.size();i++){
    fs << meshregularity1[i] << "\t" << meshregularity2[i] << std::endl;
  }
  fs.close();

} // end validate_Stokes 

