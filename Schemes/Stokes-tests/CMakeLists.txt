ADD_EXECUTABLE(PotentialGrad potentialgrad.cpp) 
TARGET_LINK_LIBRARIES(PotentialGrad mesh quadrature stokescore plot)

ADD_EXECUTABLE(PotentialCurl potentialcurl.cpp) 
TARGET_LINK_LIBRARIES(PotentialCurl mesh quadrature stokescore plot)

ADD_EXECUTABLE(PotentialNabla potentialnabla.cpp) 
TARGET_LINK_LIBRARIES(PotentialNabla mesh quadrature stokescore plot)

ADD_EXECUTABLE(LocalComplex localcomplex.cpp) 
TARGET_LINK_LIBRARIES(LocalComplex mesh quadrature stokescore plot)

ADD_EXECUTABLE(ExactComplex exactcomplex.cpp) 
TARGET_LINK_LIBRARIES(ExactComplex mesh quadrature stokescore plot)

ADD_EXECUTABLE(StabConsistency stabconsistency.cpp) 
TARGET_LINK_LIBRARIES(StabConsistency mesh quadrature stokescore plot)

