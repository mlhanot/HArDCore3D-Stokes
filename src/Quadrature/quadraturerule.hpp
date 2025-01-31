// Core data structures and methods required to implement the discrete de Rham sequence in 3D
//
// Provides:
//  - Generation of quadrature rules over cells and faces of the mesh
//
// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)
// Author: Jérome Droniou (jerome.droniou@monash.edu)
//

/*
 *
 *      This library was developed around HHO methods, although some parts of it have a more
 * general purpose. If you use this code or part of it in a scientific publication, 
 * please mention the following book as a reference for the underlying principles
 * of HHO schemes:
 *
 * The Hybrid High-Order Method for Polytopal Meshes: Design, Analysis, and Applications. 
 *  D. A. Di Pietro and J. Droniou. Modeling, Simulation and Applications, vol. 19. 
 *  Springer International Publishing, 2020, xxxi + 525p. doi: 10.1007/978-3-030-37203-3. 
 *  url: https://hal.archives-ouvertes.fr/hal-02151813.
 *
 */

#ifndef QUADRATURERULE_HPP
#define QUADRATURERULE_HPP

#include <vector>

#include <mesh.hpp>


/*!	
*	@defgroup Quadratures 
* @brief Classes providing quadratures rules on cells, faces and edges,
*   as well as cost-effective ways to compute Gram matrices of polynomial bases.
*/

namespace HArDCore3D {

/*!
*	@addtogroup Quadratures
* @{
*/

  /// Description of one node and one weight from a quadrature rule
  struct QuadratureNode
  {
    double x, y, z, w;
    QuadratureNode(double x, double y, double z, double w)
      : x(x), y(y), z(z), w(w)
    {
      // Do nothing
    }

    /// Returns the quadrature point as an Eigen vector
    inline Eigen::Vector3d vector() const {
      return Eigen::Vector3d(x,y,z);
    }
  };

  typedef std::vector<QuadratureNode> QuadratureRule;

  /// Generate quadrature rule on mesh element
  QuadratureRule generate_quadrature_rule(
					  const Cell & T,                ///< Reference to the mesh cell
					  const int doe,              ///< Degree of exactness
					  const bool force_split = false ///< TRUE if we want the quadrature nodes to be computed by forcing the splitting of the cell into pyramids based on the center of mass and the faces, and then cutting these pyramids into tetrahedrons (otherwise, tetrahedrons are not split) 
					  );  /**< @returns list of quadrature nodes and weights */

  /// Generate quadrature rule on mesh face
  QuadratureRule generate_quadrature_rule(
					  const Face & F,                ///< Reference to the mesh face
					  const int doe               ///< Degree of exactness
					  );  /**< @returns list of quadrature nodes and weights */

  /// Generate quadrature rule on mesh edge
  QuadratureRule generate_quadrature_rule(
					  const Edge & E,                ///< Reference to the mesh edge
					  const int doe               ///< Degree of exactness
					  );  /**< @returns list of quadrature nodes and weights */

}  // end of namespace HArDCore3D
#endif
