// Classes and methods to ouput various file formats
//
// Currently provides:
//  - Method to write raw matrices.
//
// Author: Marien Hanot
//

#ifndef _VTU_WRITER_HPP
#define _VTU_WRITER_HPP

#include <cstdio>
#include <string>
#include "mesh.hpp"

#include <Eigen/Dense>


/*!	
*	@defgroup Savestates 
* @brief Classes providing tools to write raw matrices
*/

namespace HArDCore3D {

// The read/write does not check anything for consistency

// ----------------------------------------------------------------------------
//                            Class definition
// ----------------------------------------------------------------------------

class MatWriter {
  public:
    MatWriter (const std::string &filename);
    ~MatWriter();
    int append_mat(const Eigen::MatrixXd &);
  private:
    std::FILE * m_fh;
};

class MatReader {
  public:
    MatReader (const std::string &filename);
    ~MatReader();
    Eigen::MatrixXd read_MatXd();
  private:
    std::FILE * m_fh;
};
}// end of namespace HArDCore3D

#endif //_VTU_WRITER
