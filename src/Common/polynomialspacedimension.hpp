// Core data structures and methods required to implement the discrete de Rham sequence in 3D
//
// Provides:
//  - Dimension of ull and partial polynomial spaces on the element, faces, and edges
//
// Author: Daniele Di Pietro (daniele.di-pietro@umontpellier.fr)
//

/*
 *
 * This library was developed around HHO methods, although some parts of it have a more
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

/*
 * The DDR sequence has been designed in
 *
 *  Fully discrete polynomial de Rham sequences of arbitrary degree on polygons and polyhedra.
 *   D. A. Di Pietro, J. Droniou, and F. Rapetti, 33p, 2019. url: https://arxiv.org/abs/1911.03616.
 *
 * If you use this code in a scientific publication, please mention the above article.
 *
 */
#ifndef POLYNOMIALSPACEDIMENSION_HPP
#define POLYNOMIALSPACEDIMENSION_HPP

#include <mesh.hpp>

namespace HArDCore3D
{

  /*!	
 * @defgroup Common 
 * @brief Various general functions and classes
 */

  /*!
   *	\addtogroup Common
   * @{
   */


  /// Basis dimensions for various polynomial spaces on edges/faces/elements (when relevant): Pk, Gk, Rk and complements.
  template<typename GeometricSupport>
  struct PolynomialSpaceDimension
  {
    // Only specializations are relevant
  };

  template<>
  struct PolynomialSpaceDimension<Edge>
  {
    /// Dimension of Pk(E)
    static size_t Poly(int k)
    {
      return (k >= 0 ? k + 1 : 0);
    }
  };

  template<>
  struct PolynomialSpaceDimension<Face>
  {
    /// Dimension of Pk(F)
    static size_t Poly(int k)
    {
      return (k >= 0 ? (k + 1) * (k + 2) / 2 : 0);
    }
    /// Dimension of Gk(F)
    static size_t Goly(int k)
    {
      return (k >= 0 ? PolynomialSpaceDimension<Face>::Poly(k + 1) - 1 : 0);
    }
    /// Dimension of Gck(F)
    static size_t GolyCompl(int k)
    {
      return 2 * PolynomialSpaceDimension<Face>::Poly(k) - PolynomialSpaceDimension<Face>::Goly(k);
    }
    /// Dimension of Rk(F)
    static size_t Roly(int k)
    {
      return (k >= 0 ? PolynomialSpaceDimension<Face>::Poly(k + 1) - 1 : 0);
    }
    /// Dimension of Rck(F)
    static size_t RolyCompl(int k)
    {
      return 2 * PolynomialSpaceDimension<Face>::Poly(k) - PolynomialSpaceDimension<Face>::Roly(k);
    }

    /// Dimension of Rb(F)
    static size_t Rolyb(int k)
    {
      return PolynomialSpaceDimension<Face>::Poly(k) - 1;
    }

    /// Dimension of Rbc(F)
    static size_t RolybCompl(int k)
    {
      return PolynomialSpaceDimension<Face>::Poly(k-2);
    }

    // Dimension of RTb(F)
    static size_t RTb(int k)
    {
      return PolynomialSpaceDimension<Face>::RolybCompl(k) + PolynomialSpaceDimension<Face>::Rolyb(k-1) + 2*PolynomialSpaceDimension<Face>::Roly(k - 1);
    }
      
    // Dimension of tildeP(F)
    static size_t tildePoly(int k)
    {
      return 6*PolynomialSpaceDimension<Face>::Poly(k);
    }
  };
  
  template<>
  struct PolynomialSpaceDimension<Cell>
  {
    /// Dimension of Pk(T)
    static size_t Poly(int k)
    {
      return (k >= 0 ? (k+1)*(k+2)*(k+3)/6 : 0);
    }
    /// Dimension of Gk(T)
    static size_t Goly(int k)
    {
      return (k >= 0 ? PolynomialSpaceDimension<Cell>::Poly(k + 1) - 1 : 0);
    }
    /// Dimension of Gck(T)
    static size_t GolyCompl(int k)
    {
      return 3 * PolynomialSpaceDimension<Cell>::Poly(k) - PolynomialSpaceDimension<Cell>::Goly(k);
    }
    /// Dimension of Rk(T)
    static size_t Roly(int k)
    {
      return (k >= 0 ? PolynomialSpaceDimension<Cell>::GolyCompl(k + 1) : 0);
    }
    /// Dimension of Rck(T)
    static size_t RolyCompl(int k)
    {
      return 3 * PolynomialSpaceDimension<Cell>::Poly(k) - PolynomialSpaceDimension<Cell>::Roly(k);
    }

    /// Dimension of Rb(T)
    static size_t Rolyb(int k)
    {
      return (k >= 1 ? PolynomialSpaceDimension<Cell>::Poly(k) - 1 : 0);
    }

    /// Dimension of Rbc(T)
    static size_t RolybCompl(int k)
    {
      if (k > 2) {
        return 3 * PolynomialSpaceDimension<Face>::Poly(k-2) + 2 * PolynomialSpaceDimension<Cell>::Poly(k-3);
      } else if (k > 1) {
        return 3 * PolynomialSpaceDimension<Face>::Poly(k-2);
      } else {
        return 0;
      }
    }

    /// Dimension of RTb(T)
    static size_t RTb(int k)
    {
      return PolynomialSpaceDimension<Cell>::RolybCompl(k) + PolynomialSpaceDimension<Cell>::Rolyb(k-1) + 
              3 * PolynomialSpaceDimension<Cell>::Roly(k-1);
    }
  };

  //@}
  
} // namespace HArDCore3D
#endif
