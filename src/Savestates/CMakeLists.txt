# Library: savestates

FILE(GLOB source_list savestates.*)
ADD_LIBRARY(savestates ${source_list})
TARGET_LINK_LIBRARIES(savestates)
