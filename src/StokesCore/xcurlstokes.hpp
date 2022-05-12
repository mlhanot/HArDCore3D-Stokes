#ifndef XCURLSTOKES_HPP
#define XCURLSTOKES_HPP

#include <globaldofspace.hpp>
#include <integralweight.hpp>
#include "stokescore.hpp"
#include "xgradstokes.hpp"

namespace HArDCore3D
{
  /*!
   *  \addtogroup StokesCore
   * @{
   */

  /// Discrete Hcurl space: local operators, L2 product and global interpolator
    /** On each edge, the DOFs correspond to the polynomial bases on the edge provided by m_stokes_core. */ 
  /** Dofs V : 3 R_{v,V} [x y z]
               3 v_E [x y z]
      Dofs E : |P^{k}| v_E [tE]
               |P^{k}| v_E [nE1]
               |P^{k}| v_E [nE2]
               |P^{k+1}| R_{v,E} [x]
               |P^{k+1}| R_{v,E} [y]
               |P^{k+1}| R_{v,E} [z]
      Dofs F : |R^{k-1}| v_{R,F} [vector]
               |R^{c,k}| v_{R,F}^c [vector]
               |P^{k-1}| v_F [scalar]
               |G^{k}| R_{v,G,F} [vector]
               |G^{c,k}| R_{v,G,F}^c [vector]
      Dofs T : |R^{k-1}| v_{R,T} [vector]
               |R^{c,k}| v_{R,T}^c [vector]
  */
  class XCurlStokes : public GlobalDOFSpace
  {
  public:
    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> FunctionType;
    typedef std::function<Eigen::Vector3d(const Eigen::Vector3d &)> FunctionCurlType;
    typedef std::function<Eigen::Matrix3d(const Eigen::Vector3d &)> FunctionGradType;

    /// A structure to store the local operators (curl and potential)
    struct LocalOperators
    {
      LocalOperators(
                     const Eigen::MatrixXd & _curl,     ///< Curl operator
                     const Eigen::MatrixXd & _curl_vec, ///< Vector Curl operator for faces
                     const Eigen::MatrixXd & _potential ///< Potential operator
                     )
        : curl(_curl),
          curl_vec(_curl_vec),
          potential(_potential)
      {
        // Do nothing
      }

      Eigen::MatrixXd curl;
      Eigen::MatrixXd curl_vec;
      Eigen::MatrixXd potential;
    };

    /// Constructor
    XCurlStokes(const StokesCore & stokes_core, bool use_threads = true, std::ostream & output = std::cout);

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
          const FunctionCurlType & Cv, ///< The function to interpolate
          const FunctionGradType & Gv, ///< The function to interpolate
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
      return * m_cell_operators[T.global_index()];  
    }

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

    /// Return edge bases for edge E
    inline const StokesCore::EdgeBases & edgeBases(const Edge & E) const
    {
      return m_stokes_core.edgeBases(E.global_index());
    }

    /// Compute the matrix of the (weighted) L2-product for the cell of index iT. The stabilisation here is based on cell and face potentials.
    // The mass matrix of P^k(T)^3 is the most expensive mass matrix in the calculation of this norm, which
    // is why there's the option of passing it as parameter if it's been already pre-computed when the norm is called.
    // Warning: Does not include dofs v_F, R_{v,G,F}, R_{v,G,F}^c, R_{v,V}, R_{v,E}
    // TODO include them
    Eigen::MatrixXd computeL2Product(
                                     const size_t iT, ///< index of the cell
                                     const double & penalty_factor = 1., ///< pre-factor for stabilisation term
                                     const Eigen::MatrixXd & mass_Pk3_T = Eigen::MatrixXd::Zero(1,1), ///< if pre-computed, the mass matrix of (P^k(T))^3; if none is pre-computed, passing Eigen::MatrixXd::Zero(1,1) will force the calculation
                                     const IntegralWeight & weight = IntegralWeight(1.) ///< weight function in the L2 product, defaults to 1
                                     ) const;

    /// Compute the matrix of the L2 product, applying leftOp and rightOp to the variables. Probably not directly called, mostly invoked through the wrappers computeL2Product and computeL2ProductGradient
    Eigen::MatrixXd computeL2Product_with_Ops(
                                     const size_t iT, ///< index of the cell
                                     const std::vector<Eigen::MatrixXd> & leftOp, ///< edge, face and element operators to apply on the left
                                     const std::vector<Eigen::MatrixXd> & rightOp, ///< edge, face and element operators to apply on the right
                                     const double & penalty_factor, ///< pre-factor for stabilisation term
                                     const Eigen::MatrixXd & w_mass_Pk3_T, ///< mass matrix of (P^k(T))^3 weighted by weight
                                     const IntegralWeight & weight ///< weight function in the L2 product
                                     ) const;
                                

    // Componentwise norms
    // L2 on cell/face does not includes components from faces/edges
    Eigen::MatrixXd computeL2opnVertex(double h) const; ///< scaling
    Eigen::MatrixXd computeL2opnEdge(size_t iE) const; ///< index of the edge
    Eigen::MatrixXd computeL2opnuEdge(size_t iE) const; ///< index of the edge, includes vertices
    Eigen::MatrixXd computeL2opnFace(size_t iF) const; ///< index of the face
    Eigen::MatrixXd computeL2opnCell(size_t iT) const; ///< index of the cell

    // Compute the projection and Rdofs on a cell
    Eigen::MatrixXd buildCurlComponentsCell(size_t iT) const;

  private:
    LocalOperators _compute_edge_curl_potential(size_t iE);
    LocalOperators _compute_face_curl_potential(size_t iF);
    LocalOperators _compute_cell_curl_potential(size_t iT);
    
    const StokesCore & m_stokes_core;
    bool m_use_threads;
    std::ostream & m_output;

    // Containers for local operators
    std::vector<std::unique_ptr<LocalOperators> > m_cell_operators;
    std::vector<std::unique_ptr<LocalOperators> > m_face_operators;
    std::vector<std::unique_ptr<LocalOperators> > m_edge_operators;
    
  };

} // end of namespace HArDCore3D
#endif
