#include <iostream>

#include <basis.hpp>

namespace HArDCore3D
{

  //------------------------------------------------------------------------------
  // Scalar monomial basis on a cell
  //------------------------------------------------------------------------------

  MonomialScalarBasisCell::MonomialScalarBasisCell(const Cell &T, size_t degree)
      : m_T(T),
        m_degree(degree),
        m_xT(T.center_mass()),
        m_hT(T.diam()),
        m_powers(MonomialPowers<Cell>::complete(degree))
  {
    // do nothing
  }

  MonomialScalarBasisCell::FunctionValue MonomialScalarBasisCell::function(size_t i, const VectorRd &x) const
  {
    VectorRd y = _coordinate_transform(x);
    const VectorZd &powers = m_powers[i];
    return std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * std::pow(y(2), powers(2));
  }

  MonomialScalarBasisCell::GradientValue MonomialScalarBasisCell::gradient(size_t i, const VectorRd &x) const
  {
    VectorRd y = _coordinate_transform(x);
    const VectorZd &powers = m_powers[i];
    VectorRd grad = VectorRd::Zero();
    grad(0) = (powers(0) == 0 ? 0. : powers(0) * std::pow(y(0), powers(0) - 1) * std::pow(y(1), powers(1)) * std::pow(y(2), powers(2)));
    grad(1) = (powers(1) == 0 ? 0. : std::pow(y(0), powers(0)) * powers(1) * std::pow(y(1), powers(1) - 1) * std::pow(y(2), powers(2)));
    grad(2) = (powers(2) == 0 ? 0. : std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * powers(2) * std::pow(y(2), powers(2) - 1));
    return grad / m_hT;
  }


  //------------------------------------------------------------------------------
  // Scalar monomial basis on a face
  //------------------------------------------------------------------------------

  MonomialScalarBasisFace::MonomialScalarBasisFace(const Face &F, size_t degree)
      : m_degree(degree),
        m_xF(F.center_mass()),
        m_hF(F.diam()),
        m_nF(F.normal()),
        m_jacobian(Eigen::Matrix<double, 2, dimspace>::Zero()),
        m_powers(MonomialPowers<Face>::complete(degree))
  {
    // Compute change of variables
    //m_jacobian.row(0) = F.edge(0)->tangent();
    //m_jacobian.row(1) = F.edge_normal(0);
    std::vector<VectorRd> tmp = F.face_tangentbasis();
    m_jacobian.row(0) = tmp[0];
    m_jacobian.row(1) = tmp[1];
    m_jacobian /= m_hF;
  }

  MonomialScalarBasisFace::FunctionValue MonomialScalarBasisFace::function(size_t i, const VectorRd &x) const
  {
    Eigen::Vector2d y = _coordinate_transform(x);
    const Eigen::Vector2i &powers = m_powers[i];
    return std::pow(y(0), powers(0)) * std::pow(y(1), powers(1));
  }

  MonomialScalarBasisFace::GradientValue MonomialScalarBasisFace::gradient(size_t i, const VectorRd &x) const
  {
    Eigen::Vector2d y = _coordinate_transform(x);
    const Eigen::Vector2i &powers = m_powers[i];
    Eigen::Vector2d grad = Eigen::Vector2d::Zero();
    grad(0) = (powers(0) == 0 ? 0. : powers(0) * std::pow(y(0), powers(0) - 1) * std::pow(y(1), powers(1)));
    grad(1) = (powers(1) == 0 ? 0. : std::pow(y(0), powers(0)) * powers(1) * std::pow(y(1), powers(1) - 1));
    return m_jacobian.transpose() * grad;
  }

  MonomialScalarBasisFace::GradientValue MonomialScalarBasisFace::curl(size_t i, const VectorRd &x) const
  {
    return gradient(i, x).cross(m_nF);
  }

  //------------------------------------------------------------------------------
  // Scalar monomial basis on an edge
  //------------------------------------------------------------------------------

  MonomialScalarBasisEdge::MonomialScalarBasisEdge(const Edge &E, size_t degree)
      : m_degree(degree),
        m_xE(E.center_mass()),
        m_hE(E.diam()),
        m_tE(E.tangent())
  {
    // Do nothing
  }

  MonomialScalarBasisEdge::FunctionValue MonomialScalarBasisEdge::function(size_t i, const VectorRd &x) const
  {
    return std::pow(_coordinate_transform(x), i);
  }

  MonomialScalarBasisEdge::GradientValue MonomialScalarBasisEdge::gradient(size_t i, const VectorRd &x) const
  {
    return (i == 0 ? 0. : i * std::pow(_coordinate_transform(x), i - 1) / m_hE) * m_tE;
  }
  
  //------------------------------------------------------------------------------
  // Basis for R^{c,k}(T)
  //------------------------------------------------------------------------------

  RolyComplBasisCell::RolyComplBasisCell(const Cell &T, size_t degree)
      : m_degree(degree),
        m_xT(T.center_mass()),
        m_hT(T.diam())
  {
    // Monomial powers for P^{k-1}(T)
    if (m_degree >= 1){
      m_powers = MonomialPowers<Cell>::complete(m_degree-1);
    }else{
      std::cout << "Attempting to construct RckT with degree 0, stopping" << std::endl;
      exit(1);
    }
  }

  RolyComplBasisCell::FunctionValue RolyComplBasisCell::function(size_t i, const VectorRd &x) const
  {
    VectorRd y = _coordinate_transform(x);
    const VectorZd &powers = m_powers[i];
    return std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * std::pow(y(2), powers(2)) * y;
  }
  
  RolyComplBasisCell::DivergenceValue RolyComplBasisCell::divergence(size_t i, const VectorRd &x) const
  {
    VectorRd y = _coordinate_transform(x);
    const VectorZd &powers = m_powers[i];
    return (powers(0)+powers(1)+powers(2)+3) * std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * std::pow(y(2), powers(2)) / m_hT;
  }


  //------------------------------------------------------------------------------
  // Basis for G^{c,k}(T)
  //------------------------------------------------------------------------------

  GolyComplBasisCell::GolyComplBasisCell(const Cell &T, size_t degree)
      : m_degree(degree),
        m_xT(T.center_mass()),
        m_hT(T.diam())
  {
    // Monomial powers for P^{k-1}(T)
    if (degree >= 1){
      m_dimPkmo3D = PolynomialSpaceDimension<Cell>::Poly(m_degree-1);
      m_dimPkmo2D = PolynomialSpaceDimension<Face>::Poly(m_degree-1);
      std::vector<VectorZd> powers3D = MonomialPowers<Cell>::complete(m_degree-1);
      std::vector<Eigen::Vector2i> powers2D = MonomialPowers<Face>::complete(m_degree-1);
      m_powers.resize(2 * m_dimPkmo3D + m_dimPkmo2D);
      for (size_t i = 0; i < m_dimPkmo3D; i++){
        m_powers[i] = powers3D[i];
        m_powers[i + m_dimPkmo3D] = powers3D[i];
      }
      size_t offset = 2 * m_dimPkmo3D;
      for (size_t i = 0; i < m_dimPkmo2D; i++){
        m_powers[offset + i] = VectorZd::Zero();
        m_powers[offset + i].head(2) = powers2D[i];
      }
    }else{
      std::cout << "Attempting to construct GckT with degree 0, stopping" << std::endl;
      exit(1);
    }
  }

  VectorRd GolyComplBasisCell::direction_value(size_t i, const VectorRd &x) const
  {
    assert(i < m_powers.size());
    if (i < m_dimPkmo3D){
      return VectorRd(0., x(2), -x(1));
    } else if (i < 2 * m_dimPkmo3D){
      return VectorRd(-x(2), 0., x(0));
    } else {
      return VectorRd(x(1), -x(0), 0.);
    }
  }

  VectorRd GolyComplBasisCell::direction_curl(size_t i, const VectorRd &x) const
  {
    assert(i < m_powers.size());
    if (i < m_dimPkmo3D){
      return VectorRd(-2., 0., 0.);
    } else if (i < 2 * m_dimPkmo3D){
      return VectorRd(0. , -2., 0.);
    } else {
      return VectorRd(0., 0., -2.);
    }
  }

  GolyComplBasisCell::FunctionValue GolyComplBasisCell::function(size_t i, const VectorRd &x) const
  {
    assert(i < m_powers.size());
    VectorRd y = _coordinate_transform(x);
    const VectorZd &powers = m_powers[i];
    return std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * std::pow(y(2), powers(2)) * direction_value(i, y);
  }
  
  GolyComplBasisCell::CurlValue GolyComplBasisCell::curl(size_t i, const VectorRd &x) const
  {
    VectorRd y = _coordinate_transform(x);
    const VectorZd &powers = m_powers[i];
    
    // value of the scalar factor in the basis function
    double scal = std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * std::pow(y(2), powers(2));
    
    // gradient of the scalar factor in the basis function
    VectorRd G = VectorRd::Zero(); 
    G(0) = (powers(0) == 0 ? 0. : powers(0) * std::pow(y(0), powers(0) - 1) * std::pow(y(1), powers(1)) * std::pow(y(2), powers(2)));
    G(1) = (powers(1) == 0 ? 0. : std::pow(y(0), powers(0)) * powers(1) * std::pow(y(1), powers(1) - 1) * std::pow(y(2), powers(2)));
    G(2) = (powers(2) == 0 ? 0. : std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * powers(2) * std::pow(y(2), powers(2) - 1));

    return ( G.cross(direction_value(i, y)) + scal * direction_curl(i, y) ) / m_hT;
  }


  //------------------------------------------------------------------------------
  // Basis for R^{c,k}(F)
  //------------------------------------------------------------------------------

  RolyComplBasisFace::RolyComplBasisFace(const Face &F, size_t degree)
      : m_degree(degree),
        m_xF(F.center_mass()),
        m_hF(F.diam()),
        m_jacobian(Eigen::Matrix<double, 2, dimspace>::Zero())
  {
    // Compute monomial powers for P^{k-1}(F)
    if (degree>=1){
      m_powers = MonomialPowers<Face>::complete(degree-1);
    }else{
      std::cout << "Attempting to construct RckF with degree 0 (possibly while constructing GckF), stopping" << std::endl;
      exit(1);
    }
  
    // Compute change of variable
    //m_jacobian.row(0) = F.edge(0)->tangent();
    //m_jacobian.row(1) = F.edge_normal(0);
    std::vector<VectorRd> tmp = F.face_tangentbasis();
    m_jacobian.row(0) = tmp[0];
    m_jacobian.row(1) = tmp[1];
    m_jacobian /= m_hF;
  }

  RolyComplBasisFace::FunctionValue RolyComplBasisFace::function(size_t i, const VectorRd &x) const
  {
    Eigen::Vector2d y = _coordinate_transform(x);
    const Eigen::Vector2i &powers = m_powers[i];
    return std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * (x-m_xF)/m_hF;
  }

  RolyComplBasisFace::DivergenceValue RolyComplBasisFace::divergence(size_t i, const VectorRd &x) const
  {
    Eigen::Vector2d y = _coordinate_transform(x);
    const Eigen::Vector2i &powers = m_powers[i];
    return (powers(0)+powers(1)+2) * std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) / m_hF;
  }


  //------------------------------------------------------------------------------
  // Basis for G^{c,k}(F)
  //------------------------------------------------------------------------------

  GolyComplBasisFace::GolyComplBasisFace(const Face &F, size_t degree)
      : m_degree(degree),
        m_nF(F.normal())
  {
    m_Rck_basis.reset(new RolyComplBasisFace(F, degree));
  }

  GolyComplBasisFace::FunctionValue GolyComplBasisFace::function(size_t i, const VectorRd &x) const
  {
    // The basis of Gck(F) is a simple rotation of the basis of Rck
    return (m_Rck_basis->function(i, x)).cross(m_nF);
  }

  //------------------------------------------------------------------------------
  // Basis for Rb^{c,k}(F)
  //------------------------------------------------------------------------------

  RolybComplBasisFace::RolybComplBasisFace(const Face &F, size_t degree)
      : m_degree(degree),
        m_xF(F.center_mass()),
        m_nF(F.normal()),
        m_hF(F.diam()),
        m_jacobian(Eigen::Matrix<double, 2, dimspace>::Zero())
  {
    // Compute monomial powers for P^{k-1}(F)
    if (degree>1){
      m_powers = MonomialPowers<Face>::complete(degree-2);
    }else{
      std::cout << "Attempting to construct RbckF with degree < 2, stopping" << std::endl;
      exit(1);
    }
  
    // Compute change of variable
    std::vector<VectorRd> tmp = F.face_tangentbasis();
    m_jacobian.row(0) = tmp[0];
    m_jacobian.row(1) = tmp[1];
    m_jacobian /= m_hF;
  }

  // [-xyQ & - yyQ// xxQ & xyQ] = Q [-y//x] [x & y] = Q rot(pi/2, nF)*[x // y] * [x//y]^T
  RolybComplBasisFace::FunctionValue RolybComplBasisFace::function(size_t i, const VectorRd &x) const
  {
    Eigen::Vector2d y = _coordinate_transform(x);
    const Eigen::Vector2i &powers = m_powers[i];
    return std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * 
            (Eigen::AngleAxisd(0.5*Math::PI,m_nF)*(x - m_xF)) *(x-m_xF).transpose()/(m_hF*m_hF);
  }

  RolybComplBasisFace::DivergenceValue RolybComplBasisFace::divergence(size_t i, const VectorRd &x) const
  {
    Eigen::Vector2d y = _coordinate_transform(x);
    const Eigen::Vector2i &powers = m_powers[i];
    return (Eigen::AngleAxisd(0.5*Math::PI,m_nF)*(x - m_xF))*
        (powers(0)+powers(1)+3) * std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) / (m_hF*m_hF);
  }

  //------------------------------------------------------------------------------
  // Basis for Rb^{k}(F)
  //------------------------------------------------------------------------------

  RolybBasisFace::RolybBasisFace(const Face &F, size_t degree)
      : m_degree(degree),
        m_xF(F.center_mass()),
        m_nF(F.normal()),
        m_hF(F.diam()),
        m_jacobian(Eigen::Matrix<double, 2, dimspace>::Zero())
  {
    // Compute monomial powers for P^{k}(F)
    if (degree>=1){
      m_powers = MonomialPowers<Face>::complete(degree);
    }else{
      std::cout << "Attempting to construct RbkF with degree 0, stopping" << std::endl;
      exit(1);
    }
  
    // Compute change of variable
    std::vector<VectorRd> tmp = F.face_tangentbasis();
    m_jacobian.row(0) = tmp[0];
    m_jacobian.row(1) = tmp[1];
    m_jacobian /= m_hF;
  }

  // (div^-1)_{i,j} = h/(2 + i + j) 1_{i,j}; (grad)_{i,j} -> 1/h [i 1_{i-1,j} // j 1_{i,j-1}]
  // (div^-1 grad)_{i,j} = 1/(2 + i + j - 1) [X i 1_{i-1,j} // X j 1_{i,j-1}]
  RolybBasisFace::FunctionValue RolybBasisFace::function(size_t i, const VectorRd &x) const
  {
    i++; // shift i, discard constant
    Eigen::Vector2d y = _coordinate_transform(x);
    const Eigen::Vector2i &powers = m_powers[i];
    Eigen::Matrix3d rv = Eigen::Matrix3d::Zero();
    // We assume that m_powers[i] = [0,0] iff i = 0
    if (powers(0) == 0) { // Avoid 0*x^-1 
      rv = powers(1)* std::pow(y(1),powers(1)-1) * m_jacobian.row(1).transpose() * (x - m_xF).transpose();
        // j x^0 y^{j-1} tF2 \otimes (X - X_F)/hF 
    } else if (powers(1) == 0) {
      rv = powers(0)* std::pow(y(0),powers(0)-1) * m_jacobian.row(0).transpose() * (x - m_xF).transpose();
    } else {
      rv = powers(0)* std::pow(y(0),powers(0)-1) * std::pow(y(1),powers(1))
            * m_jacobian.row(0).transpose() * (x - m_xF).transpose();
      rv += powers(1)* std::pow(y(1),powers(1)-1) * std::pow(y(0),powers(0))
            * m_jacobian.row(1).transpose() * (x - m_xF).transpose();
    } 
    return rv/(1. + powers(0) + powers(1));
  }

  RolybBasisFace::DivergenceValue RolybBasisFace::divergence(size_t i, const VectorRd &x) const
  {
    i++;
    Eigen::Vector2d y = _coordinate_transform(x);
    const Eigen::Vector2i &powers = m_powers[i];
    Eigen::Vector3d rv = Eigen::Vector3d::Zero();
    if (powers(0) == 0) {
      rv = powers(1) * std::pow(y(1),powers(1) - 1) * m_jacobian.row(1).transpose();
    } else if (powers(1) == 0) {  
      rv = powers(0) * std::pow(y(0),powers(0) - 1) * m_jacobian.row(0).transpose();
    } else {
      rv = powers(1) * std::pow(y(1),powers(1) - 1) * std::pow(y(0),powers(0)) * m_jacobian.row(1).transpose();
      rv += powers(0) * std::pow(y(0),powers(0) - 1) * std::pow(y(1),powers(1)) * m_jacobian.row(0).transpose();
    }
    return rv; // jacobian = ntF /hF
  }

  //------------------------------------------------------------------------------
  // Basis for Rb^{k}(T)
  //------------------------------------------------------------------------------

  RolybBasisCell::RolybBasisCell(const Cell &T, size_t degree)
      : m_degree(degree),
        m_xT(T.center_mass()),
        m_hT(T.diam())
  {
    // Compute monomial powers for P^{k}(T)
    if (degree>=1){
      m_powers = MonomialPowers<Cell>::complete(degree);
    }else{
      std::cout << "Attempting to construct RbkT with degree 0, stopping" << std::endl;
      exit(1);
    }
  }

  // (div^-1)_{i,j,k} = h/(3 + i + j + k) 1_{i,j,k}; (grad)_{i,j,k} -> 1/h [i 1_{i-1,j,k} // ...]
  // (div^-1 grad)_{i,j} = 1/(3 + i + j + k - 1) [X i 1_{i-1,j,k} // ...]
  RolybBasisCell::FunctionValue RolybBasisCell::function(size_t i, const VectorRd &x) const
  {
    i++; // shift i, discard constant
    VectorRd y = _coordinate_transform(x);
    const VectorZd &powers = m_powers[i];
    Eigen::Matrix3d rv = Eigen::Matrix3d::Zero();
    if (powers(0) > 0) {
      rv.row(0) = powers(0)* std::pow(y(0),powers(0)-1) * std::pow(y(1),powers(1)) * std::pow(y(2),powers(2)) * y;
    }
    if (powers(1) > 0) {
      rv.row(1) = powers(1)* std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)-1) * std::pow(y(2),powers(2)) * y;
    }
    if (powers(2) > 0) {
      rv.row(2) = powers(2)* std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)) * std::pow(y(2),powers(2)-1) * y;
    }
    return rv/(2. + powers(0) + powers(1) + powers(2));
  }

  RolybBasisCell::DivergenceValue RolybBasisCell::divergence(size_t i, const VectorRd &x) const
  {
    i++; // shift i, discard constant
    VectorRd y = _coordinate_transform(x);
    const VectorZd &powers = m_powers[i];
    Eigen::Vector3d rv = Eigen::Vector3d::Zero();
    if (powers(0) > 0) {
      rv(0) = powers(0)* std::pow(y(0),powers(0)-1) * std::pow(y(1),powers(1)) * std::pow(y(2),powers(2));
    }
    if (powers(1) > 0) {
      rv(1) = powers(1)* std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)-1) * std::pow(y(2),powers(2));
    }
    if (powers(2) > 0) {
      rv(2) = powers(2)* std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)) * std::pow(y(2),powers(2)-1);
    }
    return rv/m_hT;
  }

  //------------------------------------------------------------------------------
  // Basis for Rb^{c,k}(T)
  //------------------------------------------------------------------------------

  RolybComplBasisCell::RolybComplBasisCell(const Cell &T, size_t degree)
      : m_degree(degree),
        m_xT(T.center_mass()),
        m_hT(T.diam()),
        dim_Pkm2_F(PolynomialSpaceDimension<Face>::Poly(degree-2)),
        dim_Pkm3_T(PolynomialSpaceDimension<Cell>::Poly(degree-3))
  {
    // Compute monomial powers for P^{k-1}(T)
    if (degree<2){
      std::cout << "Attempting to construct RbckT with degree < 2, stopping" << std::endl;
      exit(1);
    }else{
      m_powers_L = MonomialPowers<Face>::complete(degree-2); // Y, Z
      m_powers_B = MonomialPowers<Face>::complete(degree-2); // X, Z
      m_powers_G = MonomialPowers<Face>::complete(degree-2); // X, Y
      if (degree> 2) {
        m_powers_C1 = MonomialPowers<Cell>::complete(degree-3); // C1
        m_powers_C2 = MonomialPowers<Cell>::complete(degree-3); // C2
      }
    }
  }

  RolybComplBasisCell::FunctionValue RolybComplBasisCell::function(size_t i, const VectorRd &x) const
  {
    VectorRd y = _coordinate_transform(x);
    size_t il = i,j;
    map_powers(il,j);
    MatrixRd rv = MatrixRd::Zero();
    switch (j) {
      case(0) : // Lambda, P2 = z Lambda, P3 = - y Lambda
        {
        const Eigen::Vector2i &powers = m_powers_L[il];
        double Lambda = std::pow(y(1),powers(0)) * std::pow(y(2),powers(1)); // Y^i Z^j
        rv.row(1) = y * y(2) * Lambda; // P2 = z Lambda
        rv.row(2) = y * (-y(1)) * Lambda; // P3 = - y Lambda
        }
        break;
      case(1): 
        {
        const Eigen::Vector2i &powers = m_powers_B[il];
        double Beta = std::pow(y(0),powers(0)) * std::pow(y(2),powers(1)); // X^i Z^j
        rv.row(0) = y * (-y(2)) * Beta; // P1 = - z Beta
        rv.row(2) = y * y(0) * Beta; // P3 = x Beta
        }
        break;
      case(2): 
        {
        const Eigen::Vector2i &powers = m_powers_G[il];
        double Gamma = std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)); // X^i Y^j
        rv.row(0) = y * y(1) * Gamma; // P1 = y Gamma
        rv.row(1) = y * (-y(0)) * Gamma; // P2 = -x Gamma
        }
        break;
      case(3): // C1; C3 = -C1 - C2
        {
        const Eigen::Vector3i &powers = m_powers_C1[il];
        double C = std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)) * std::pow(y(2),powers(2));
        rv.row(0) = y * y(1) * y(2) * C; // P1 = y z C1
        rv.row(2) = y * (-y(0)) * y(1) * C; // P3 = x y C3 = - x y C1
        }
        break;
      case(4): // C2; C3 = -C1 - C2
        {
        const Eigen::Vector3i &powers = m_powers_C2[il];
        double C = std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)) * std::pow(y(2),powers(2));
        rv.row(1) = y * y(0) * y(2) * C; // P2 = x z C1
        rv.row(2) = y * (-y(0)) * y(1) * C; // P3 = x y C3 = - x y C1
        }
        break;
    }
    return rv;
  }

  RolybComplBasisCell::DivergenceValue RolybComplBasisCell::divergence(size_t i, const VectorRd &x) const
  {
    VectorRd y = _coordinate_transform(x);
    size_t il = i,j;
    map_powers(il,j);
    VectorRd rv = VectorRd::Zero();
    switch (j) {
      case(0) : // Lambda, P2 = z Lambda, P3 = - y Lambda
        {
        const Eigen::Vector2i &powers = m_powers_L[il];
        double Lambda = std::pow(y(1),powers(0)) * std::pow(y(2),powers(1)); // Y^i Z^j
        Lambda *= 4. + powers(0) + powers(1); // Lambda -> Lambda'
        rv(1) = y(2) * Lambda; // z Lambda'
        rv(2) = -y(1) * Lambda; // - y Lambda'
        }
        break;
      case(1): 
        {
        const Eigen::Vector2i &powers = m_powers_B[il];
        double Beta = std::pow(y(0),powers(0)) * std::pow(y(2),powers(1)); // X^i Z^j
        Beta *= 4. + powers(0) + powers(1); // Beta -> Beta'
        rv(0) = -y(2) * Beta; // - z Beta
        rv(2) = y(0) * Beta; // x Beta
        }
        break;
      case(2): 
        {
        const Eigen::Vector2i &powers = m_powers_G[il];
        double Gamma = std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)); // X^i Y^j
        Gamma *= 4. + powers(0) + powers(1); // Gamma -> Gamma'
        rv(0) = y(1) * Gamma; // y Gamma
        rv(1) = -y(0) * Gamma; // -x Gamma
        }
        break;
      case(3): // C1; C3 = -C1 - C2
        {
        const Eigen::Vector3i &powers = m_powers_C1[il];
        double C = std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)) * std::pow(y(2),powers(2));
        C *= 5. + powers(0) + powers(1) + powers(2);
        rv(0) = y(1) * y(2) * C; // y z C1
        rv(2) = -y(0) * y(1) * C; // x y C3 = - x y C1
        }
        break;
      case(4): // C2; C3 = -C1 - C2
        {
        const Eigen::Vector3i &powers = m_powers_C2[il];
        double C = std::pow(y(0),powers(0)) * std::pow(y(1),powers(1)) * std::pow(y(2),powers(2));
        C *= 5. + powers(0) + powers(1) + powers(2);
        rv(1) = y(0) * y(2) * C; // x z C1
        rv(2) = -y(0) * y(1) * C; // x y C3 = - x y C1
        }
        break;
    }
    return rv/m_hT;
  }

  //------------------------------------------------------------------------------
  // A common notion of scalar product for scalars and vectors
  //------------------------------------------------------------------------------

  double scalar_product(const double &x, const double &y)
  {
    return x * y;
  }

  double scalar_product(const double &x, const Eigen::Matrix<double, 1, 1> &y)
  {
    return x * y(0);
  }
  
  double scalar_product(const VectorRd &x, const VectorRd &y)
  {
    return x.dot(y);
  }
  
  double scalar_product(const MatrixRd &x, const MatrixRd &y)
  {
    return x.cwiseProduct(y).sum();
  }

  boost::multi_array<VectorRd, 2> matrix_vector_product(
    const boost::multi_array<MatrixRd, 2> &basis_quad,
    const VectorRd &v)
  {
    boost::multi_array<VectorRd, 2> basis_dot_v_quad(boost::extents[basis_quad.shape()[0]][basis_quad.shape()[1]]);
    std::transform(basis_quad.origin(),basis_quad.origin() + basis_quad.num_elements(),
    basis_dot_v_quad.origin(), [&v](const MatrixRd &x) -> VectorRd { return x*v; });
    return basis_dot_v_quad;
  }

  boost::multi_array<MatrixRd, 2> tensor_tangent_product(
    const boost::multi_array<MatrixRd, 2> &basis_quad,
    const Eigen::Matrix<double,3,1> &v)
  {
    boost::multi_array<MatrixRd, 2> basis_dot_v_quad(boost::extents[basis_quad.shape()[0]][basis_quad.shape()[1]]);
    std::transform(basis_quad.origin(),basis_quad.origin() + basis_quad.num_elements(),
    basis_dot_v_quad.origin(), [&v](const MatrixRd &x) -> MatrixRd { return x - (x*v)*v.transpose(); });
    return basis_dot_v_quad;
  }

  boost::multi_array<double, 2> eval_trace_quad(const boost::multi_array<MatrixRd, 2> &basis_quad)
  {
    boost::multi_array<double, 2> basis_trace_quad(boost::extents[basis_quad.shape()[0]][basis_quad.shape()[1]]);
    std::transform(basis_quad.origin(),basis_quad.origin() + basis_quad.num_elements(),
    basis_trace_quad.origin(), [](const MatrixRd &x) -> double {return x.trace();});
    return basis_trace_quad;
  }

  boost::multi_array<VectorRd, 2>
  vector_product(
      const boost::multi_array<VectorRd, 2> &basis_quad,
      const VectorRd &v)
  {
    boost::multi_array<VectorRd, 2> basis_cross_v_quad(boost::extents[basis_quad.shape()[0]][basis_quad.shape()[1]]);
    std::transform(basis_quad.origin(), basis_quad.origin() + basis_quad.num_elements(),
                   basis_cross_v_quad.origin(), [&v](const VectorRd &x) -> VectorRd { return x.cross(v); });
    return basis_cross_v_quad;
  }

  //------------------------------------------------------------------------------
  //      Gram matrices
  //------------------------------------------------------------------------------

  // Vector3d for B1, tensorialised double for B2
  Eigen::MatrixXd compute_gram_matrix(const boost::multi_array<VectorRd, 2> &B1,
                                      const boost::multi_array<double, 2> &B2,
                                      const QuadratureRule &qr)
  {
    // Check that the basis evaluation and quadrature rule are coherent
    assert(qr.size() == B1.shape()[1] && qr.size() == B2.shape()[1]);

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(B1.shape()[0], dimspace * B2.shape()[0]);
    for (size_t i = 0; i < B1.shape()[0]; i++)
    {
      for (size_t k = 0; k < dimspace; k++)
      {
        VectorRd ek = VectorRd::Zero();
        ek(k) = 1.;
        for (size_t j = 0; j < B2.shape()[0]; j++)
        {
          for (size_t iqn = 0; iqn < qr.size(); iqn++)
          {
            M(i, k * B2.shape()[0] + j) += qr[iqn].w * B1[i][iqn].dot(ek) * B2[j][iqn];
          } // for iqn
        }   // for j
      }     // for k
    }       // for i
    return M;
  }

  // Gramm matrix for double-valued B1, B2
  Eigen::MatrixXd compute_gram_matrix(const boost::multi_array<double, 2> &B1,
                                      const boost::multi_array<double, 2> &B2,
                                      const QuadratureRule &qr,
                                      const size_t nrows,
                                      const size_t ncols,
                                      const std::string sym)
  {
    // Check that the basis evaluation and quadrature rule are coherent
    assert(qr.size() == B1.shape()[1] && qr.size() == B2.shape()[1]);
    // Check that we don't ask for more members of family than available
    assert(nrows <= B1.shape()[0] && ncols <= B2.shape()[0]);

    // Re-cast quadrature weights into ArrayXd to facilitate computations
    Eigen::ArrayXd qr_weights = Eigen::ArrayXd::Zero(qr.size());
    for (size_t iqn = 0; iqn < qr.size(); iqn++)
    {
      qr_weights(iqn) = qr[iqn].w;
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nrows, ncols);
    for (size_t i = 0; i < nrows; i++)
    {
      size_t jcut = 0;
      if (sym == "sym")
        jcut = i;
      for (size_t j = 0; j < jcut; j++)
      {
        M(i, j) = M(j, i);
      }
      for (size_t j = jcut; j < ncols; j++)
      {
        std::vector<double> tmp(B1.shape()[1]);
        // Extract values at quadrature nodes for elements i of B1 and j of B2
        boost::multi_array<double, 1> B1i = B1[boost::indices[i][boost::multi_array_types::index_range(0, B1.shape()[1])]];
        boost::multi_array<double, 1> B2j = B2[boost::indices[j][boost::multi_array_types::index_range(0, B1.shape()[1])]];
        double *p1 = &B1i[0];
        double *p2 = &B2j[0];
        Eigen::ArrayXd B1i_as_array = Eigen::Map<Eigen::ArrayXd, Eigen::Unaligned>(p1, B1i.shape()[0]);
        Eigen::ArrayXd B2j_as_array = Eigen::Map<Eigen::ArrayXd, Eigen::Unaligned>(p2, B2j.shape()[0]);

        // Multiply by quadrature weights and sum (using .sum() of ArrayXd makes this step faster than a loop
        M(i, j) = (qr_weights * B1i_as_array * B2j_as_array).sum();
      } // for j
    }   // for i
    return M;
  }

  // Gram matrix for double-valued complete family. Do not make this inline, this slows down calculations.
  Eigen::MatrixXd compute_gram_matrix(const boost::multi_array<double, 2> &B1,
                                      const boost::multi_array<double, 2> &B2,
                                      const QuadratureRule &qr,
                                      const std::string sym)
  {
    return compute_gram_matrix(B1, B2, qr, B1.shape()[0], B2.shape()[0], sym);
  }

  // Gram matrix for Vector3d-valued B1 and B2
  Eigen::MatrixXd compute_gram_matrix(const boost::multi_array<VectorRd, 2> &B1,
                                      const boost::multi_array<VectorRd, 2> &B2,
                                      const QuadratureRule &qr,
                                      const size_t nrows,
                                      const size_t ncols,
                                      const std::string sym)
  {
    // Check that the basis evaluation and quadrature rule are coherent
    assert(qr.size() == B1.shape()[1] && qr.size() == B2.shape()[1]);
    // Check that we don't ask for more members of family than available
    assert(nrows <= B1.shape()[0] && ncols <= B2.shape()[0]);

    // Re-cast quadrature weights into ArrayXd to make computations faster
    Eigen::ArrayXd qr_weights = Eigen::ArrayXd::Zero(qr.size());
    for (size_t iqn = 0; iqn < qr.size(); iqn++)
    {
      qr_weights(iqn) = qr[iqn].w;
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nrows, ncols);
    for (size_t i = 0; i < nrows; i++)
    {
      size_t jcut = 0;
      if (sym == "sym")
        jcut = i;
      for (size_t j = 0; j < jcut; j++)
      {
        M(i, j) = M(j, i);
      }
      for (size_t j = jcut; j < ncols; j++)
      {
        // Array of scalar products
        Eigen::ArrayXd B1i_dot_B2j = Eigen::ArrayXd::Zero(qr.size());
        for (size_t iqn = 0; iqn < qr.size(); iqn++)
        {
          B1i_dot_B2j(iqn) = B1[i][iqn].dot(B2[j][iqn]);
        }

        // Multiply component-wise by weights and sum
        M(i, j) = (qr_weights * B1i_dot_B2j).sum();
      } // for j
    }   // for i
    return M;
  }

  // Gram matrix for Vector3d-valued complete family. Do not make this inline, this slows down actual calculations.
  Eigen::MatrixXd compute_gram_matrix(const boost::multi_array<VectorRd, 2> &B1,
                                      const boost::multi_array<VectorRd, 2> &B2,
                                      const QuadratureRule &qr,
                                      const std::string sym)
  {
    return compute_gram_matrix(B1, B2, qr, B1.shape()[0], B2.shape()[0], sym);
  }

  Eigen::MatrixXd compute_weighted_gram_matrix(
      const FType<VectorRd> &f,
      const BasisQuad<VectorRd> &B1,
      const BasisQuad<double> &B2,
      const QuadratureRule &qr,
      size_t n_rows,
      size_t n_cols)
  {
    // If default, set n_rows and n_cols to size of families
    if (n_rows == 0 && n_cols == 0)
    {
      n_rows = B1.shape()[0];
      n_cols = B2.shape()[0];
    }

    // Number of quadrature nodes
    const size_t num_quads = qr.size();
    // Check number of quadrature nodes is compatible with B1 and B2
    assert(num_quads == B1.shape()[1] && num_quads == B2.shape()[1]);
    // Check that we don't ask for more members of family than available
    assert(n_rows <= B1.shape()[0] && n_cols <= B2.shape()[0]);

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_rows, n_cols);
    for (size_t iqn = 0; iqn < num_quads; iqn++)
    {
      double qr_weight = qr[iqn].w;
      VectorRd f_on_qr = f(qr[iqn].vector());
      for (size_t i = 0; i < n_rows; i++)
      {
        double f_dot_B1 = f_on_qr.dot(B1[i][iqn]);
        for (size_t j = 0; j < n_cols; j++)
        {
          M(i, j) += qr_weight * f_dot_B1 * B2[j][iqn];
        }
      }
    }
    return M;
  }

  Eigen::MatrixXd compute_weighted_gram_matrix(
      const FType<VectorRd> &f,
      const BasisQuad<double> &B1,
      const BasisQuad<VectorRd> &B2,
      const QuadratureRule &qr,
      size_t n_rows,
      size_t n_cols)
  {
    return compute_weighted_gram_matrix(f, B2, B1, qr, n_cols, n_rows).transpose();
  }

} // end of namespace HArDCore3D
