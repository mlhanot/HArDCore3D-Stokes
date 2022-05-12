#include <iostream>
#include <string>
#include <vector>
#include <math.h>

inline double compute_rate(const std::vector<double> &a, const std::vector<double> &h, size_t i) {
  return (std::log(a[i]) - std::log(a[i-1]))/(std::log(h[i]) - std::log(h[i-1]));
}

int main()
{
  std::vector<double> h,e;
  std::string line;
  double htmp,etmp;
  while(std::getline(std::cin,line,'\n')) {
    if (sscanf(line.c_str(),"%lf\t%lf",&htmp,&etmp)) {
      h.emplace_back(htmp);
      e.emplace_back(etmp);
    }
  }
  std::cout<<"Rate: "<<std::endl;
  for (size_t i = 1; i < h.size(); i++) {
    std::cout<<compute_rate(e,h,i)<<std::endl;
  }
  return 0;
}

