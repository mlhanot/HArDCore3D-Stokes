#ifndef XGRADSTOKES_HPP
#define XGRADSTOKES_HPP

#include <globaldofspace.hpp>
#include <integralweight.hpp>

#include "stokescore.hpp"

namespace HArDCore3D
{
  /*!
   *  \addtogroup StokesCore
   * @{
   */

  /// Discrete H1 space: local operators, L2 product and global interpolator
  /** On each edge/face/element, the DOFs (if any) correspond to the polynomial bases on the edge/face/element provided by m_stokes_core */
  /** Dofs V : 3 G_{q,V} [x y z]
               1 q(V) [scalar]
      Dofs E : |P^{k-1}| q_E [scalar]
               |P^{k}| G_{q,E} [nE1]
               |P^{k}| G_{q,E} [nE2]
      Dofs F : |P^{k-1}| q_F [scalar]
               |P^{k-1}| G_{q,F} [scalar]
      Dofs T : |P^{k-1}| q_T [scalar]
  */
  class XGradStokes : public GlobalDOFSpace
  {
  public:
    typedef std::function<double(const Eigen::Vector3d &)> FunctionType;
    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> FunctionGradType;

    /// A structure to store local operators (gradient and potential)
    struct LocalOperators
    {
      LocalOperators(
                     const Eigen::MatrixXd & _gradient, ///< Gradient operator
                     const Eigen::MatrixXd & _gradient_perp, ///< Gradient^perp for faces
                     const Eigen::MatrixXd & _potential ///< Potential operator
                     )
        : gradient(_gradient),
          gradient_perp(_gradient_perp),
          potential(_potential)
      {
        // Do nothing
      }
      
      Eigen::MatrixXd gradient;
      Eigen::MatrixXd gradient_perp;
      Eigen::MatrixXd potential;
    };
    
    /// Constructor
    XGradStokes(const StokesCore & ddr_core, bool use_threads = true, std::ostream & output = std::cout);

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
          const FunctionType & q, ///< The function to interpolate
          const FunctionGradType & Gq, ///< The gradient of the function to interpolate
          const int doe_cell = -1, ///< The optional degre of cell quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
          const int doe_face = -1, ///< The optional degre of face quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
          const int doe_edge = -1 ///< The optional degre of edge quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
          ) const;

    /// Return edge operators for the edge of index iE
    inline const LocalOperators & edgeOperators(size_t iE) const
    {
      return *m_edge_operators[iE];
    }

    /// Return edge operators for edge E
    inline const LocalOperators & edgeOperators(const Edge & E) const
    {
      return *m_edge_operators[E.global_index()];
    }
    
    /// Return face operators for the face of index iF
    inline const LocalOperators & faceOperators(size_t iF) const
    {
      return *m_face_operators[iF];
    }

    /// Return face operators for face F
    inline const LocalOperators & faceOperators(const Face & F) const
    {
      return *m_face_operators[F.global_index()];
    }

    /// Return cell operators for the cell of index iT
    inline const LocalOperators & cellOperators(size_t iT) const
    {
      return *m_cell_operators[iT];
    }

    /// Return cell operators for cell T
    inline const LocalOperators & cellOperators(const Cell & T) const
    {
      return *m_cell_operators[T.global_index()];
    }

    /// Return cell bases for the cell of index iT
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

    /// Return edge bases for edge E
    inline const StokesCore::EdgeBases & edgeBases(const Edge & E) const
    {
      return m_stokes_core.edgeBases(E.global_index());
    }

    /// Compute the matrix of the (weighted) L2-product for the cell of index iT. The stabilisation here is based on cell and face potentials.
    // The mass matrix of P^{k+1}(T) is the most expensive mass matrix in the calculation of this norm, which
    // is why there's the option of passing it as parameter if it's been already pre-computed when the norm is called.
    // As there lack a definition for the additional terms, we simply add them as ||G||^2
    Eigen::MatrixXd computeL2Product(
                                     const size_t iT, ///< index of the cell
                                     const double & penalty_factor = 1., ///< pre-factor for stabilisation term
                                     const Eigen::MatrixXd & mass_Pkpo_T = Eigen::MatrixXd::Zero(1,1), ///< if pre-computed, the mass matrix of P^{k+1}(T); if none is pre-computed, passing Eigen::MatrixXd::Zero(1,1) will force the calculation
                                     const IntegralWeight & weight = IntegralWeight(1.) ///< weight function in the L2 product, defaults to 1
                                     ) const;

    // Build the components of the gradient operator (probably less useful)
    Eigen::MatrixXd buildGradientComponentsFace(size_t iF) const;
    /// Build the components of the gradient operator (probably not useful in practice to implement schemes)
    Eigen::MatrixXd buildGradientComponentsCell(size_t iT) const;
    
  private:    
    LocalOperators _compute_edge_gradient_potential(size_t iE);
    LocalOperators _compute_face_gradient_potential(size_t iF);
    LocalOperators _compute_cell_gradient_potential(size_t iT);    

    const StokesCore & m_stokes_core;
    bool m_use_threads;
    std::ostream & m_output;

    // Containers for local operators
    std::vector<std::unique_ptr<LocalOperators> > m_edge_operators;
    std::vector<std::unique_ptr<LocalOperators> > m_face_operators;
    std::vector<std::unique_ptr<LocalOperators> > m_cell_operators;
  };

} // end of namespace HArDCore3D

#endif
