#ifndef TESTFUNCTION_HPP
#define TESTFUNCTION_HPP

#include <basis.hpp>
#include <random>
#include <boost/math/differentiation/autodiff.hpp>

// Workarround for https://github.com/boostorg/math/issues/445
template <typename X> X powfx(const X &x, size_t n) 
{X rv = 1; for(size_t i = 0; i < n; i++) {rv *= x;} return rv;}

using namespace boost::math::differentiation;

namespace HArDCore3D {
  
  static std::mt19937 gen(6);

  enum Initialization {
    Zero,
    One,
    Default,
    Random
  };

  [[maybe_unused]] static void fill_random_vector (Eigen::VectorXd &inout) {
    std::uniform_real_distribution<> dis(-10.,10.);
    for (long i = 0; i < inout.size();i++) {
      inout[i] = dis(gen);
    }
    return; 
  }

  // Create a global polynomial of total degree k and provide its derivatives
  template <size_t k>
  class PolyTest {
    public:
      PolyTest(Initialization initmode = Initialization::Zero,double _scale = 1.) : scale(_scale) {
        coefficients.resize(nb_monomial,0.);
        switch(initmode) {
          case (Initialization::Random) :
            {
              std::uniform_real_distribution<> dis(1.,3.);
              for (size_t i = 0; i < coefficients.size();i++) {
                coefficients[i] = dis(gen);
              }
            }
            break;
          case (Initialization::One):
            {
              for (size_t i = 0; i < coefficients.size();i++) {
                coefficients[i] = 1.;
              }
            }
          case (Initialization::Default):
          case (Initialization::Zero):
           ; 
        }
        powers = MonomialPowers<Cell>::complete(k);
      }

      double evaluate(const VectorRd &x, size_t diffx = 0, size_t diffy = 0, size_t diffz = 0) const {
        auto const variables = make_ftuple<double, k+2, k+2, k+2>(x(0),x(1),x(2));
        auto const& X = std::get<0>(variables);
        auto const& Y = std::get<1>(variables);
        auto const& Z = std::get<2>(variables);
        auto const v = f(X,Y,Z);
        return v.derivative(diffx,diffy,diffz);
      }

      std::vector<double> coefficients;
      double scale;
      template <typename X, typename Y, typename Z> // promote from boost
        promote<X,Y,Z> f(const X &x,const Y &y, const Z &z) const {
          promote<X,Y,Z> rv = 0;
          for (size_t i = 0;i < powers.size(); i++) {
            rv += coefficients[i]*powfx(x,powers[i](0))*powfx(y,powers[i](1))*powfx(z,powers[i](2));
          }
          return scale*rv;
        }

      std::string expr_string() const {
        std::stringstream output;
        output << "P(X,Y,Z) =";
        for (size_t i = 0; i < powers.size(); i++) {
          output << " + " << coefficients[i] << "X^" << powers[i](0) << "Y^" << powers[i](1) << "Z^" << powers[i](2);
        }
        return output.str();
      }
    private:
      const static size_t nb_monomial = (k + 1) * (k + 2) *(k + 3)/ 6;
      std::vector<VectorZd> powers;
  };

  // Create a global exponential function and provide its derivatives
  template <size_t k>
  class TrigTest {
    public:
      TrigTest(Initialization initmode = Initialization::Zero,double _scale = 1.) : scale(_scale) {
        for (size_t ix = 1; ix < 2;ix++) {
          for (size_t iy = 1; iy < 2;iy++) {
            for (size_t jx = 1; jx < 2;jx++) {
              for (size_t jy = 1; jy < 2;jy++) {
                for (size_t kx = 1; kx < 2;kx++) {
                  for (size_t ky = 1; ky < 2;ky++) {
                    for (size_t iz = 1; iz < 2;iz++) {
                      for (size_t jz = 1; jz < 2;jz++) {
                        for (size_t kz = 1; kz < 2;kz++) {
                            Eigen::VectorXi v(9);
                            v << ix, iy, jx, jy, kx, ky, iz, jz, kz;
                            powers.push_back(v);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        nb_elements = powers.size();
        coefficients.resize(nb_elements);
        switch(initmode) {
          case (Initialization::Random) :
            {
              std::uniform_real_distribution<> dis(1.,2.);
              for (size_t i = 0; i < coefficients.size(); i++) {
                coefficients[i] = Eigen::Vector3d(dis(gen),dis(gen),dis(gen));
              }
            }
            break;
          case (Initialization::Default):
          case (Initialization::Zero):
          default:
            for (size_t i = 0; i < coefficients.size(); i++) {
              coefficients[i] = Eigen::Vector3d::Zero();
            }
        }
      }

      double evaluate(const VectorRd &x, size_t diffx = 0, size_t diffy = 0, size_t diffz = 0) const {
        auto const variables = make_ftuple<double, k+2, k+2, k+2>(x(0),x(1),x(2));
        auto const& X = std::get<0>(variables);
        auto const& Y = std::get<1>(variables);
        auto const& Z = std::get<2>(variables);
        auto const v = f(X,Y,Z);
        return v.derivative(diffx,diffy,diffz);
      }

      std::vector<Eigen::Vector3d> coefficients;
      template <typename X, typename Y, typename Z> // promote from boost
        promote<X,Y,Z> f(const X &x,const Y &y,const Z &z) const {
          promote<X,Y,Z> rv = 0.;
          for (size_t i = 0;i < powers.size(); i++) {
            rv += coefficients[i](0)*cos(coefficients[i](1)*powfx(x,powers[i](0)))*
                  coefficients[i](0)*cos(coefficients[i](1)*powfx(y,powers[i](1))); 
                  coefficients[i](0)*cos(coefficients[i](1)*powfx(z,powers[i](6))); 
            rv += coefficients[i](0)*sin(coefficients[i](1)*powfx(x,powers[i](2)))*
                  coefficients[i](0)*sin(coefficients[i](1)*powfx(y,powers[i](3)));
                  coefficients[i](0)*sin(coefficients[i](1)*powfx(z,powers[i](7)));
            rv += coefficients[i](0)*exp(coefficients[i](1)*powfx(x,powers[i](4)))*
                  coefficients[i](0)*exp(coefficients[i](1)*powfx(y,powers[i](5))); 
                  coefficients[i](0)*exp(coefficients[i](1)*powfx(z,powers[i](8))); 
          }
          return scale*rv;
        }
      double scale;
      std::string expr_string() const {return "Not Implemented";}
    private:
      size_t nb_elements;
      std::vector<Eigen::VectorXi> powers;
  };

  [[maybe_unused]] static int m_switch = 0;

  [[maybe_unused]] static class Errandswitch {
    public:
    Errandswitch operator++(int) {Errandswitch temp(*this); m_errcount++; m_switch++; return temp;}
    operator int() const {return m_errcount;}
    private:
      int m_errcount = 0;
    } nb_errors;

  [[maybe_unused]] static struct endls_s {} endls;
    std::ostream& operator<<(std::ostream& out, endls_s) {
    if (m_switch) {
      m_switch = 0;
      return out << "\033[1;31m Unexpected\033[0m" << std::endl;
    } else {
      return out << std::endl;
    }
  }

  static constexpr double threshold = 1e-9;

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

} // end of namespace
#endif

