# Library: stokescore

FILE(GLOB source_list stokescore.* xgradstokes.* xcurlstokes.* xnablastokes.* xvlstokes.*)
ADD_LIBRARY(stokescore ${source_list})
TARGET_LINK_LIBRARIES(stokescore basis mesh quadrature savestates)
