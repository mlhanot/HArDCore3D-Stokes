#ifndef STOKES_STRUCTS_HPP
#define STOKES_STRUCTS_HPP

class formatted_output {
    private:
      int width;
      std::ostream& stream_obj;
    public:
      formatted_output(std::ostream& obj, int w): width(w), stream_obj(obj) {}
      template<typename T>
      formatted_output& operator<<(const T& output) {
        stream_obj << std::setw(width) << output;

        return *this;}

      formatted_output& operator<<(std::ostream& (*func)(std::ostream&)) {
        func(stream_obj);
        return *this;}
  };
#define FORMATD(W)                                                      \
  ""; formatted_output(std::cout,W+8) << std::setiosflags(std::ios_base::left | std::ios_base::scientific) << std::setprecision(W) << std::setfill(' ')

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)

inline double compute_rate(const std::vector<double> &a, const std::vector<double> &h, size_t i) {
  return (std::log(a[i]) - std::log(a[i-1]))/(std::log(h[i]) - std::log(h[i-1]));
}

const std::vector<std::filesystem::path> mesh_files0 = {
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Cubic-Cells/" "RF_fmt/" "gcube_2x2x2",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Cubic-Cells/" "RF_fmt/" "gcube_4x4x4",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Cubic-Cells/" "RF_fmt/" "gcube_8x8x8",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Cubic-Cells/" "RF_fmt/" "gcube_16x16x16",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Cubic-Cells/" "RF_fmt/" "gcube_32x32x32"};/**/

const std::vector<std::filesystem::path> mesh_files1 = {
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Prysmatic-Cells-1/" "RF_fmt/" "gdual_5x5x5",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Prysmatic-Cells-1/" "RF_fmt/" "gdual_10x10x10",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Prysmatic-Cells-1/" "RF_fmt/" "gdual_15x15x15",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Prysmatic-Cells-1/" "RF_fmt/" "gdual_20x20x20",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Prysmatic-Cells-1/" "RF_fmt/" "gdual_25x25x25"};/**/

const std::vector<std::filesystem::path> mesh_files2 = {
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Random-Hexahedra/" "RF_fmt/" "gcube.1",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Random-Hexahedra/" "RF_fmt/" "gcube.2",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Random-Hexahedra/" "RF_fmt/" "gcube.3",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Random-Hexahedra/" "RF_fmt/" "gcube.4"};/**/

const std::vector<std::filesystem::path> mesh_files3 = {
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Tetgen-Cube-0/" "RF_fmt/" "cube.1",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Tetgen-Cube-0/" "RF_fmt/" "cube.2",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Tetgen-Cube-0/" "RF_fmt/" "cube.3",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Tetgen-Cube-0/" "RF_fmt/" "cube.4",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Tetgen-Cube-0/" "RF_fmt/" "cube.5",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Tetgen-Cube-0/" "RF_fmt/" "cube.6"};/**/

const std::vector<std::filesystem::path> mesh_files4 = {
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-small-2/" "RF_fmt/" "voro.2",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-small-2/" "RF_fmt/" "voro.3",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-small-2/" "RF_fmt/" "voro.4",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-small-2/" "RF_fmt/" "voro.5",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-small-2/" "RF_fmt/" "voro.6",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-small-2/" "RF_fmt/" "voro.7",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-small-2/" "RF_fmt/" "voro.8"};/**/

const std::vector<std::filesystem::path> mesh_files5 = {
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-Tets-1/" "RF_fmt/" "voro.1",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-Tets-1/" "RF_fmt/" "voro.2",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-Tets-1/" "RF_fmt/" "voro.3",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-Tets-1/" "RF_fmt/" "voro.4",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-Tets-1/" "RF_fmt/" "voro.5",
                                             STRINGIFY(ROOT_DIR) "/meshes/" "Voro-Tets-1/" "RF_fmt/" "voro.6"};/**/


struct Sol_struct {
  std::function<VectorRd(const VectorRd &)> u_exact;
  std::function<double(const VectorRd &)> p_exact;
  std::function<VectorRd(const VectorRd &)> f;
  std::function<VectorRd(const VectorRd &)> u_N;
};

template<size_t k_int> // Polynomial
struct Sol_struct0 : public Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  Sol_struct0(): 
    Sol_struct([](const VectorRd &x)->VectorRd {
      VectorRd rv;
      rv << 
      x(1)*x(1) + x(0) + x(2),
      x(0)*x(0) - x(1),
      x(0)*x(1) + 0.2;
      return rv;
    },
    [](const VectorRd &x)->double {
      return x(2) - x(1);
    },
    [](const VectorRd &x)->VectorRd {
      VectorRd rv;
      rv << 
        0. -2.,
        -1. - 2.,
        1. - 0.;
      return rv;
    }) {};
};
template<size_t k_int> // Dirichlet, p poly
struct Sol_struct1 : public Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  Sol_struct1() :
    Sol_struct([](const VectorRd &x)->VectorRd {
      VectorRd rv;
      rv << 
  k*std::pow(x(0), 2)*std::pow(-x(0) + 1., 2)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1)),
  x(0)*(-x(0) + 1.)*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2))*std::cos(k*x(2))
  - (2.*std::pow(x(0), 3) - 3.*std::pow(x(0), 2) + x(0))*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2)),
  -x(0)*(-x(0) + 1.)*std::sin(k*x(1))*std::pow(std::sin(k*x(2)), 2)*std::cos(k*x(1));
      return rv;
    },
    [](const VectorRd &x)->double {
      return x(0) - x(1) ;
    },
    [](const VectorRd &x)->VectorRd {
      VectorRd rv;
      rv << 
  5.*std::pow(k, 3)*std::pow(x(0), 2)*std::pow(-x(0) + 1., 2)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1))
  - 2.*k*std::pow(x(0), 2)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1))
  - 4.*k*x(0)*(2.*x(0) - 2.)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1))
  - 2.*k*std::pow(-x(0) + 1., 2)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1)) + 1.,
  6.*std::pow(k, 2)*x(0)*(-x(0) + 1.)*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2))*std::cos(k*x(2))
  - 2.*std::pow(k, 2)*x(0)*(-x(0) + 1.)*std::sin(k*x(2))*std::pow(std::cos(k*x(1)), 2)*std::cos(k*x(2))
  + (12.*x(0) - 6.)*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2)) - 3.*std::pow(k, 2)*(2.*std::pow(x(0), 3)
  - 3.*std::pow(x(0), 2) + x(0))*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2))
  + 2.*std::pow(k, 2)*(2.*std::pow(x(0), 3) - 3.*std::pow(x(0), 2)
  + x(0))*std::sin(k*x(2))*std::pow(std::cos(k*x(1)), 2)
  + 2.*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2))*std::cos(k*x(2)) - 1.,
  -6.*std::pow(k, 2)*x(0)*(-x(0) + 1.)*std::sin(k*x(1))*std::pow(std::sin(k*x(2)), 2)*std::cos(k*x(1))
  + 2.*std::pow(k, 2)*x(0)*(-x(0) + 1.)*std::sin(k*x(1))*std::cos(k*x(1))*std::pow(std::cos(k*x(2)), 2)
  - 2.*std::sin(k*x(1))*std::pow(std::sin(k*x(2)), 2)*std::cos(k*x(1));
      return rv;
    }) {};
};
template<size_t k_int> // Dirichlet, p trig
struct Sol_struct2 : public Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  Sol_struct2() :
    Sol_struct([](const VectorRd &x)->VectorRd {
      VectorRd rv;
      rv << 
  k*std::pow(x(0), 2)*std::pow(-x(0) + 1., 2)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1)),
  x(0)*(-x(0) + 1.)*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2))*std::cos(k*x(2))
  - (2.*std::pow(x(0), 3) - 3.*std::pow(x(0), 2) + x(0))*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2)),
  -x(0)*(-x(0) + 1.)*std::sin(k*x(1))*std::pow(std::sin(k*x(2)), 2)*std::cos(k*x(1));
      return rv;
    },
    [](const VectorRd &x)->double {
      return std::sin(k*x(0)*0.5) - x(1)*x(2)*8./double(k);
    },
    [](const VectorRd &x)->VectorRd {
      VectorRd rv;
      rv << 
  5.*std::pow(k, 3)*std::pow(x(0), 2)*std::pow(-x(0) + 1., 2)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1))
  - 2.*k*std::pow(x(0), 2)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1))
  - 4.*k*x(0)*(2.*x(0) - 2.)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1))
  - 2.*k*std::pow(-x(0) + 1., 2)*std::sin(k*x(1))*std::sin(k*x(2))*std::cos(k*x(1))
  + k*std::cos(k*x(0)*0.5)*0.5, // pressure
  6.*std::pow(k, 2)*x(0)*(-x(0) + 1.)*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2))*std::cos(k*x(2))
  - 2.*std::pow(k, 2)*x(0)*(-x(0) + 1.)*std::sin(k*x(2))*std::pow(std::cos(k*x(1)), 2)*std::cos(k*x(2))
  + (12.*x(0) - 6.)*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2)) - 3.*std::pow(k, 2)*(2.*std::pow(x(0), 3)
  - 3.*std::pow(x(0), 2) + x(0))*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2))
  + 2.*std::pow(k, 2)*(2.*std::pow(x(0), 3) - 3.*std::pow(x(0), 2)
  + x(0))*std::sin(k*x(2))*std::pow(std::cos(k*x(1)), 2)
  + 2.*std::pow(std::sin(k*x(1)), 2)*std::sin(k*x(2))*std::cos(k*x(2))
  - 8.*x(2)/k, // pressure
  -6.*std::pow(k, 2)*x(0)*(-x(0) + 1.)*std::sin(k*x(1))*std::pow(std::sin(k*x(2)), 2)*std::cos(k*x(1))
  + 2.*std::pow(k, 2)*x(0)*(-x(0) + 1.)*std::sin(k*x(1))*std::cos(k*x(1))*std::pow(std::cos(k*x(2)), 2)
  - 2.*std::sin(k*x(1))*std::pow(std::sin(k*x(2)), 2)*std::cos(k*x(1))
  - 8.*x(1)/k;
      return rv;
    }) {};
};
template<size_t k_int> // mixte, homogenous Neumann on z = 0
struct Sol_struct3 : public Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  Sol_struct3(): 
    Sol_struct([](const VectorRd &x)->VectorRd {
      VectorRd rv;
      double expG = std::exp((x(0) - 0.5)*(x(0) - 0.5)/2. + (x(1) - 0.5)*(x(1) - 0.5)/2.);
      rv << 
      x(2)*x(2) * expG,
      x(2)*x(2) * std::sin(k*x(0)/2.)*std::cos(k*x(1)/2.),
      x(2)*x(2)*x(2) * ((1. - 2.*x(0))*expG +  k*std::sin(k*x(0)/2.)*std::sin(k*x(1)/2.))/6.; 
      return rv;
    },
    [](const VectorRd &x)->double {
      return x(2)*x(2);
    },
    [](const VectorRd &x)->VectorRd {
      VectorRd rv;
      double expG = std::exp((x(0) - 0.5)*(x(0) - 0.5)/2. + (x(1) - 0.5)*(x(1) - 0.5)/2.);
      rv << 
      -(x(2)*x(2)*((x(0) - 0.5)*(x(0) - 0.5) + (x(1) - 0.5)*(x(1) - 0.5) + 2.) + 2.)*expG,
      -(2. - k*k*x(2)*x(2)/2.) * std::sin(k*x(0)/2.)*std::cos(k*x(1)/2.),
      2.*x(2)
        + x(2)*x(2)*x(2)*((2.*x(0) - 1.0)*(x(1) - 0.5)*(2.*x(0) - 1.0)*(x(1) - 0.5)*expG 
        + (2*x(0) - 1.0)*expG + k*k*k*std::sin(k*x(0)/2.)*std::sin(k*x(1)/2.)/4.)/6. 
        + x(2)*x(2)*x(2)*((x(0) - 0.5)*(x(0) - 0.5)*(2.*x(0) - 1.0)*expG 
        + 4.*(x(0) - 0.5)*expG + (2.*x(0) - 1.0)*expG 
        + k*k*k*std::sin(k*x(0)/2.)*std::sin(k*x(1)/2.)/4.)/6. 
        + x(2)*((2*x(0) - 1.0)*expG - k*std::sin(k*x(0)/2.)*std::sin(k*x(1)/2.));
      return rv;
    }) {};
};
template<size_t k_int> // mixte,  Neumann on z = 0
struct Sol_struct4 : public Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  Sol_struct4():
    Sol_struct([](const VectorRd &x)->VectorRd {
      VectorRd rv;
      double expG = std::exp((x(0) - 0.5)*(x(0) - 0.5)/2. + (x(1) - 0.5)*(x(1) - 0.5)/2.);
      rv << 
      expG,
      std::sin(k*x(0)/2.)*std::cos(k*x(1)/2.),
      -x(2)*((2.*x(0) - 1.)*expG -  k*std::sin(k*x(0)/2.)*std::sin(k*x(1)/2.))/2.; 
      return rv;
    },
    [](const VectorRd &x)->double {
      return x(2)*x(2);
    },
    [](const VectorRd &x)->VectorRd {
      VectorRd rv;
      double expG = std::exp((x(0) - 0.5)*(x(0) - 0.5)/2. + (x(1) - 0.5)*(x(1) - 0.5)/2.);
      rv << 
      -((x(0) - 0.5)*(x(0) - 0.5) + (x(1) - 0.5)*(x(1) - 0.5) + 2.)*expG,
      k*k*std::sin(k*x(0)/2.)*std::cos(k*x(1)/2.)/2,
      2.*x(2)
        - x(2)*((-32.*x(0) + 16.0)*expG
        + (-8.*x(0) + 4.)*(x(0) - 0.5)*(x(0) - 0.5)*expG
        + (-8.*x(0) + 4.)*(x(1) - 0.5)*(x(1) - 0.5)*expG
        - 2.*k*k*k*std::sin(k*x(0)/2.)*sin(k*x(1)/2.))/8.;
      return rv;
    },
    [](const VectorRd &x)->VectorRd {
      VectorRd rv;
      double expG = std::exp((x(0) - 0.5)*(x(0) - 0.5)/2. + (x(1) - 0.5)*(x(1) - 0.5)/2.);
      rv << 0.,
      0.,
      (x(0) - 0.5)*expG - k*std::sin(k*x(0)/2.)*std::sin(k*x(1)/2.)/2.;
      return rv;
    }) {};
};

#endif // STOKES_STRUCTS_HPP
