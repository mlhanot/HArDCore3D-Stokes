// Core data structures and methods required to implement the discrete Stokes sequence in 3D
//
// Provides:
//  - Full and partial polynomial spaces on the element, faces, and edges
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

/*
 * The Stokes sequence has been designed in
 *
 *  XXX
 *
 * If you use this code in a scientific publication, please mention the above article.
 *
 */
 

#ifndef STOKESCORE_HPP
#define STOKESCORE_HPP

#include <memory>
#include <iostream>

#include <basis.hpp>
#include <polynomialspacedimension.hpp>

/*!	
 * @defgroup StokesCore 
 * @brief Classes providing tools to the Discrete De Rham sequence
 */


namespace HArDCore3D
{

  /*!
   *	\addtogroup StokesCore
   * @{
   */


  //------------------------------------------------------------------------------

  /// Construct all polynomial spaces for the DDR sequence
  class StokesCore
  {
  public:
    // Types for element bases
    typedef Family<MonomialScalarBasisCell> PolyBasisCellType;
    typedef TensorizedVectorFamily<PolyBasisCellType, 3> Poly3BasisCellType;
    typedef Family<GradientBasis<ShiftedBasis<MonomialScalarBasisCell> > > GolyBasisCellType;
    typedef Family<GolyComplBasisCell> GolyComplBasisCellType;
    typedef Family<CurlBasis<GolyComplBasisCell>> RolyBasisCellType;
    typedef Family<RolyComplBasisCell> RolyComplBasisCellType;
    //families for RTb
    typedef TensorizedMatrixFamily<RolyBasisCellType, 3> Roly3BasisCellType;
    typedef Family<RolybComplBasisCell> RolybComplBasisCellType;
    typedef Family<RolybBasisCell> RolybBasisCellType;
    typedef SumFamily<SumFamily<RolybComplBasisCellType,RolybBasisCellType>,Roly3BasisCellType> RTbBasisCellType; // (Rbc^k + Rb^k) + R3^k
    // Nabla potential
    typedef TensorizedMatrixFamily<RolyComplBasisCellType, 3> RolyCompl3BasisCellType;

    // Types for face bases
    typedef Family<MonomialScalarBasisFace> PolyBasisFaceType;
    typedef TangentFamily<PolyBasisFaceType> Poly2BasisFaceType;
    typedef Family<CurlBasis<ShiftedBasis<MonomialScalarBasisFace>>> RolyBasisFaceType;
    typedef Family<RolyComplBasisFace> RolyComplBasisFaceType;
    typedef Family<GradientBasis<ShiftedBasis<MonomialScalarBasisFace>>> GolyBasisFaceType;
    typedef Family<GolyComplBasisFace> GolyComplBasisFaceType;
    // families for Ptilde
    typedef bPolyBasisFace<PolyBasisFaceType> bPolyBasisFaceType;
    typedef PolyT2BasisFace<RolyBasisFaceType> Roly2BasisFaceType; // (R^k)^2 \subset P^{3x3}
    typedef Family<RolybComplBasisFace> RolybComplBasisFaceType;
    typedef Family<RolybBasisFace> RolybBasisFaceType;
    typedef SumFamily<bPolyBasisFaceType,SumFamily<SumFamily<RolybComplBasisFaceType,RolybBasisFaceType>,Roly2BasisFaceType>> tildePolyBasisFaceType; // bP^k + ((Rbc^k + Rb^k) + R2^k) 
    typedef SumFamily<RolyBasisFaceType,RolyComplBasisFaceType> RTBasisFaceType; // Defined for convenience, never used
    // Nabla potential
    typedef TensorizedVectorFamily<PolyBasisFaceType, 3> Poly3BasisFaceType;
    typedef TensorizedMatrixFamily<RolyComplBasisFaceType, 3> RolyCompl3BasisFaceType;

    // Type for edge basis
    typedef Family<MonomialScalarBasisEdge> PolyBasisEdgeType;
    typedef TensorizedVectorFamily<PolyBasisEdgeType,3> Poly3BasisEdgeType;

    /// Structure to store element bases
    /** 'Poly': basis of polynomial space; 'Goly': gradient basis; 'Roly': curl basis.\n
        'k', 'kmo' (k-1) and 'kpo' (k+1) determines the degree.\n
        'Compl' for the complement of the corresponding 'Goly' or 'Roly' in the 'Poly' space */
    struct CellBases
    {
      /// Geometric support
      typedef Cell GeometricSupport;

      std::unique_ptr<PolyBasisCellType> Polykpo;
      std::unique_ptr<PolyBasisCellType> Polyk;
      std::unique_ptr<PolyBasisCellType> Polykmo;
      std::unique_ptr<Poly3BasisCellType> Polyk3po; // Nabla potential
      std::unique_ptr<Poly3BasisCellType> Polyk3;
      std::unique_ptr<GolyBasisCellType> Golykmo;
      std::unique_ptr<GolyComplBasisCellType> GolyComplk;
      std::unique_ptr<GolyComplBasisCellType> GolyComplkpo;
      std::unique_ptr<RolyBasisCellType>  Rolykmo;
      std::unique_ptr<RolyComplBasisCellType> RolyComplk;
      std::unique_ptr<RolyComplBasisCellType> RolyComplkp2;
      // RTb
      std::unique_ptr<Roly3BasisCellType> Roly3k;
      std::unique_ptr<RolybComplBasisCellType> RolybComplkpo;
      std::unique_ptr<RolybBasisCellType> Rolybk;
      std::unique_ptr<RTbBasisCellType> RTbkpo;
      std::unique_ptr<RolyCompl3BasisCellType> RolyComplk3p2; // Nabla potential
    };

    /// Structure to store face bases
    /** See CellBases for details */
    struct FaceBases
    {
      /// Geometric support
      typedef Face GeometricSupport;

      std::unique_ptr<PolyBasisFaceType> Polykpo;
      std::unique_ptr<PolyBasisFaceType> Polyk;
      std::unique_ptr<PolyBasisFaceType> Polykmo;
      std::unique_ptr<Poly2BasisFaceType> Polyk2;
      std::unique_ptr<RolyBasisFaceType> Rolykmo;
      std::unique_ptr<RolyComplBasisFaceType> RolyComplk;
      std::unique_ptr<RolyComplBasisFaceType> RolyComplkp2;
      std::unique_ptr<GolyBasisFaceType> Golyk;
      std::unique_ptr<GolyComplBasisFaceType> GolyComplk;
      // tildeP
      std::unique_ptr<bPolyBasisFaceType> bPolykpo;
      std::unique_ptr<Roly2BasisFaceType> Rolyk2po;
      std::unique_ptr<RolybComplBasisFaceType> RolybComplkpo;
      std::unique_ptr<RolybBasisFaceType> Rolybkpo;
      std::unique_ptr<tildePolyBasisFaceType> tildePolykpo;
      // Potential Nabla
      std::unique_ptr<Poly3BasisFaceType> Polyk3p2;
      std::unique_ptr<RolyCompl3BasisFaceType> RolyComplk3p3;
    };

    /// Structure to store edge bases
    /** See CellBases for details */
    struct EdgeBases
    {
      /// Geometric support
      typedef Edge GeometricSupport;

      std::unique_ptr<PolyBasisEdgeType> Polykp3;
      std::unique_ptr<PolyBasisEdgeType> Polykp2;
      std::unique_ptr<PolyBasisEdgeType> Polykpo;
      std::unique_ptr<PolyBasisEdgeType> Polyk;
      std::unique_ptr<PolyBasisEdgeType> Polykmo;
      std::unique_ptr<Poly3BasisEdgeType> Polyk3p3;
      std::unique_ptr<Poly3BasisEdgeType> Polyk3p2;
      std::unique_ptr<Poly3BasisEdgeType> Polyk3po;
    };    
    
    /// Constructor
    StokesCore(const Mesh & mesh, size_t K, bool use_threads = true, std::ostream & output = std::cout);
    
    /// Return a const reference to the mesh
    const Mesh & mesh() const
    {
      return m_mesh;
    }

    /// Return the polynomial degree
    const size_t & degree() const
    {
      return m_K;
    }
    
    /// Return cell bases for element iT
    inline const CellBases & cellBases(size_t iT) const
    {
      // Make sure that the basis has been created
      assert( m_cell_bases[iT] );
      return *m_cell_bases[iT].get();
    }

    /// Return face bases for face iF
    inline const FaceBases & faceBases(size_t iF) const
    {
      // Make sure that the basis has been created
      assert( m_face_bases[iF] );
      return *m_face_bases[iF].get();
    }

    /// Return edge bases for edge iE
    inline const EdgeBases & edgeBases(size_t iE) const
    {
      // Make sure that the basis has been created
      assert( m_edge_bases[iE] );
      return *m_edge_bases[iE].get();
    }

  private:
    /// Compute the bases on an element T
    CellBases _construct_cell_bases(size_t iT);

    /// Compute the bases on a face F
    FaceBases _construct_face_bases(size_t iF);

    /// Compute the bases on an edge E
    EdgeBases _construct_edge_bases(size_t iE);
    
    // Pointer to the mesh
    const Mesh & m_mesh;
    // Degree
    const size_t m_K;
    // Output stream
    std::ostream & m_output;
    
    // Cell bases
    std::vector<std::unique_ptr<CellBases> > m_cell_bases;
    // Face bases
    std::vector<std::unique_ptr<FaceBases> > m_face_bases;
    // Edge bases
    std::vector<std::unique_ptr<EdgeBases> > m_edge_bases;
        
  };
  
} // end of namespace HArDCore3D

#ifdef DEBUG
template<typename T> inline void DEBUGPRINTMAT(T M,std::string name = "N/A") {std::cout<<"Matrix "<<name<<" : Rows "<<M.rows()<<", Cols "<<M.cols()<<std::endl;}
#define DBPMAT(x) DEBUGPRINTMAT(x,#x)
template<typename T> inline void DEBUGPRINTMATCONTENT(T M) {std::cout.precision(3); std::cout<< M << std::endl;}
#define DBPMC(x) DBPMAT(x); DEBUGPRINTMATCONTENT(x)
template<typename T> inline Eigen::MatrixXd flushM(T M) {return (1.e-10 < M.array().abs()).select(M, 0);}
#endif

#endif // STOKESCORE_HPP
