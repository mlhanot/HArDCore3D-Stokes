#ifndef XVLSTOKES_HPP
#define XVLSTOKES_HPP

#include <globaldofspace.hpp>
#include <integralweight.hpp>
#include "ddrcore.hpp"

namespace HArDCore3D
{

  /*!
   *	\addtogroup DDRCore
   * @{
   */

  /// Discrete L^2(scalar) space: L2 product and global interpolator
  /** On each face, the DOFs correspond to the polynomial bases on the face provided by m_ddr_core.*/
  /** Dofs V : 0
      Dofs E : 0
      Dofs F : 0
      Dofs T : |P^k| q_T [scalar]
  */

  class XSL : public GlobalDOFSpace
  {
  public:
    typedef std::function<double(const Eigen::Vector3d &)> FunctionType;

    /// Constructor
    XSL(const DDRCore & ddr_core, bool use_threads = true, std::ostream & output = std::cout);

    /// Return the mesh
    const Mesh & mesh() const
    {
      return m_ddr_core.mesh();
    }
    
    /// Return the polynomial degree
    const size_t & degree() const
    {
      return m_ddr_core.degree();
    }
    
    /// Interpolator of a continuous function
    Eigen::VectorXd interpolate(
          const FunctionType & v, ///< The function to interpolate
          const int doe_cell = -1 ///< The optional degre of cell quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
                ) const;


    /// Return cell bases for the face of index iT
    inline const DDRCore::CellBases & cellBases(size_t iT) const
    {
      return m_ddr_core.cellBases(iT);
    }

    /// Return cell bases for cell T
    inline const DDRCore::CellBases & cellBases(const Cell & T) const
    {
      return m_ddr_core.cellBases(T.global_index());
    }
    
    /// Return face bases for the face of index iF
    inline const DDRCore::FaceBases & faceBases(size_t iF) const
    {
      return m_ddr_core.faceBases(iF);
    }

    /// Return cell bases for face F
    inline const DDRCore::FaceBases & faceBases(const Face & F) const
    {
      return m_ddr_core.faceBases(F.global_index());
    }

    /// Return edge bases for the edge of index iE
    inline const DDRCore::EdgeBases & edgeBases(size_t iE) const
    {
      return m_ddr_core.edgeBases(iE);
    }

    /// Return cell bases for edge E
    inline const DDRCore::EdgeBases & edgeBases(const Edge & E) const
    {
      return m_ddr_core.edgeBases(E.global_index());
    }
    
    // L2 product
    Eigen::MatrixXd compute_Gram_Cell(size_t iT) const; // Polyk

    // Evaluate the scalar function, inefficient
    double evaluatePotential(size_t iT, const Eigen::VectorXd & uh, const VectorRd & x) const;

  private:

    const DDRCore & m_ddr_core;
    bool m_use_threads;
    std::ostream & m_output;

  };
} // end of namespace HArDCore3D
#endif
