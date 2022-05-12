#include "savestates.hpp"

#include <cstdint>

namespace HArDCore3D {
// Class
MatWriter::MatWriter(const std::string &filename) {
  m_fh = std::fopen(filename.c_str(), "wb");
}

MatWriter::~MatWriter() {
  std::fclose(m_fh);
}

int MatWriter::append_mat(const Eigen::MatrixXd &mat) {
  int nw;
  uint64_t size[2];
  size[0] = mat.rows();
  size[1] = mat.cols();
  nw = std::fwrite(&size,sizeof(uint64_t),2,m_fh);
  if (nw != 2) return 1;
  const double *data = mat.data();
  nw = std::fwrite(data,sizeof(double),mat.size(),m_fh);
  return (nw == mat.size()) ? 0 : 1;
}

MatReader::MatReader(const std::string &filename) {
  m_fh = std::fopen(filename.c_str(), "rb");
}

MatReader::~MatReader() {
  std::fclose(m_fh);
}

Eigen::MatrixXd MatReader::read_MatXd() {
  int nr;
  uint64_t size[2];
  Eigen::MatrixXd rv;
  nr = std::fread(&size,sizeof(uint64_t),2,m_fh);
  if (nr != 2) throw std::runtime_error("MatLoad: Wrong data format");
  rv.resize(size[0],size[1]);
  double *data = rv.data();
  nr = std::fread(data,sizeof(double),rv.size(),m_fh);
  if (nr != rv.size()) throw std::runtime_error("MatLoad: Wrong data format");
  return rv;
}
  
} // end of namespace HArDCore3D


