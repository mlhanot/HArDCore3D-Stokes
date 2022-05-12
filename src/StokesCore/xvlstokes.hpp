#ifndef XVLSTOKES_HPP
#define XVLSTOKES_HPP

#include <globaldofspace.hpp>
#include <integralweight.hpp>
#include "stokescore.hpp"
#include "xnablastokes.hpp"

namespace HArDCore3D
{
  /*!
   *	\addtogroup StokesCore
   * @{
   */

  /// Discrete L^2(vector) space: L2 product and global interpolator
  /** On each face, the DOFs correspond to the polynomial bases on the face provided by m_stokes_core.*/
  /** Dofs V : 0
      Dofs E : 3*|P^{k+2}| W_E [x y z]
      Dofs F : |~P^{k+1}| W_F [tensor(3D)]
      Dofs T : |RTb^{k+1}| W_T [tensor]
  */

  class XVLStokes : public GlobalDOFSpace
  {
  public:
    typedef std::function<Eigen::Matrix3d(const Eigen::Vector3d &)> FunctionType;

    /// Constructor
    XVLStokes(const StokesCore & stokes_core, bool use_threads = true, std::ostream & output = std::cout);

    /// Return the mesh
    const Mesh & mesh() const
    {
      return m_stokes_core.mesh();
    }
    
    /// Return the polynomial degree
    const size_t & degree() const
    {
      return m_stokes_core.degree();
    }
    
    /// Interpolator of a continuous function
    Eigen::VectorXd interpolate(
          const FunctionType & W, ///< The function to interpolate
          const int doe_cell = -1, ///< The optional degre of cell quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
          const int doe_face = -1, ///< The optional degre of face quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
          const int doe_edge = -1 ///< The optional degre of edge quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
                ) const;

    /// Return cell bases for the face of index iT
    inline const StokesCore::CellBases & cellBases(size_t iT) const
    {
      return m_stokes_core.cellBases(iT);
    }

    /// Return cell bases for cell T
    inline const StokesCore::CellBases & cellBases(const Cell & T) const
    {
      return m_stokes_core.cellBases(T.global_index());
    }
    
    /// Return face bases for the face of index iF
    inline const StokesCore::FaceBases & faceBases(size_t iF) const
    {
      return m_stokes_core.faceBases(iF);
    }

    /// Return cell bases for face F
    inline const StokesCore::FaceBases & faceBases(const Face & F) const
    {
      return m_stokes_core.faceBases(F.global_index());
    }

    /// Return edge bases for the edge of index iE
    inline const StokesCore::EdgeBases & edgeBases(size_t iE) const
    {
      return m_stokes_core.edgeBases(iE);
    }

    /// Return cell bases for edge E
    inline const StokesCore::EdgeBases & edgeBases(const Edge & E) const
    {
      return m_stokes_core.edgeBases(E.global_index());
    }
    
    // L2 product
    Eigen::MatrixXd compute_Gram_Edge(size_t iE) const; // Polyk3p2
    Eigen::MatrixXd compute_Gram_Face(size_t iF) const; // tildePolykpo
    Eigen::MatrixXd compute_Gram_Cell(size_t iT) const; // RTb

    /// Compute the matrix of the (weighted) L2-product for the cell of index iT, precomposed with nabla.
    // The mass matrix of RTb^k+1(T) is the most expensive mass matrix in the calculation of this norm, which
    // is why there's the option of passing it as parameter if it's been already pre-computed when the norm is called.
    Eigen::MatrixXd computeL2Product_GG(
                                     const size_t iT, ///< index of the cell
                                     const XNablaStokes & xnabla, // XNabla instance to get nabla operators
                                     const double & penalty_factor = 1., ///< pre-factor for stabilisation term
                                     const Eigen::MatrixXd & mass_RTbkpo_T = Eigen::MatrixXd::Zero(1,1), ///< if pre-computed, the mass matrix of (RTb^k+1(T)); if none is pre-computed, passing Eigen::MatrixXd::Zero(1,1) will force the calculation
                                     const IntegralWeight & weight = IntegralWeight(1.) ///< weight function in the L2 product, defaults to 1
                                     ) const;

    /// Compute the matrix of the L2 product, applying leftOp and rightOp to the variables. Probably not directly called, mostly invoked through the wrapper computeL2Product_GG
    Eigen::MatrixXd computeL2Product_with_Ops(
                                     const size_t iT, ///< index of the cell
                                     const std::vector<Eigen::MatrixXd> & leftOp, ///< edge, face and element operators to apply on the left
                                     const std::vector<Eigen::MatrixXd> & rightOp, ///< edge, face and element operators to apply on the right
                                     const double & penalty_factor, ///< pre-factor for stabilisation term
                                     const Eigen::MatrixXd & w_mass_RTbkpo_T, ///< mass matrix of RTb^k+1(T) weighted by weight
                                     const IntegralWeight & weight ///< weight function in the L2 product
                                     ) const;

  private:

    const StokesCore & m_stokes_core;
    bool m_use_threads;
    std::ostream & m_output;

  };

  /*!
   *	\addtogroup StokesCore
   * @{
   */

  /// Discrete L^2(scalar) space: L2 product and global interpolator
  /** On each face, the DOFs correspond to the polynomial bases on the face provided by m_stokes_core.*/
  /** Dofs V : 0
      Dofs E : 0
      Dofs F : 0
      Dofs T : |P^k| q_T [scalar]
  */

  class XSLStokes : public GlobalDOFSpace
  {
  public:
    typedef std::function<double(const Eigen::Vector3d &)> FunctionType;

    /// Constructor
    XSLStokes(const StokesCore & stokes_core, bool use_threads = true, std::ostream & output = std::cout);

    /// Return the mesh
    const Mesh & mesh() const
    {
      return m_stokes_core.mesh();
    }
    
    /// Return the polynomial degree
    const size_t & degree() const
    {
      return m_stokes_core.degree();
    }
    
    /// Interpolator of a continuous function
    Eigen::VectorXd interpolate(
          const FunctionType & v, ///< The function to interpolate
          const int doe_cell = -1 ///< The optional degre of cell quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
                ) const;


    /// Return cell bases for the face of index iT
    inline const StokesCore::CellBases & cellBases(size_t iT) const
    {
      return m_stokes_core.cellBases(iT);
    }

    /// Return cell bases for cell T
    inline const StokesCore::CellBases & cellBases(const Cell & T) const
    {
      return m_stokes_core.cellBases(T.global_index());
    }
    
    /// Return face bases for the face of index iF
    inline const StokesCore::FaceBases & faceBases(size_t iF) const
    {
      return m_stokes_core.faceBases(iF);
    }

    /// Return cell bases for face F
    inline const StokesCore::FaceBases & faceBases(const Face & F) const
    {
      return m_stokes_core.faceBases(F.global_index());
    }

    /// Return edge bases for the edge of index iE
    inline const StokesCore::EdgeBases & edgeBases(size_t iE) const
    {
      return m_stokes_core.edgeBases(iE);
    }

    /// Return cell bases for edge E
    inline const StokesCore::EdgeBases & edgeBases(const Edge & E) const
    {
      return m_stokes_core.edgeBases(E.global_index());
    }
    
    // L2 product
    Eigen::MatrixXd compute_Gram_Cell(size_t iT) const; // Polyk

    // Evaluate the scalar function, inefficient
    double evaluatePotential(size_t iT, const Eigen::VectorXd & uh, const VectorRd & x) const;

  private:

    const StokesCore & m_stokes_core;
    bool m_use_threads;
    std::ostream & m_output;

  };
} // end of namespace HArDCore3D
#endif
