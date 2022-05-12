mpirun -n 1 Schemes/stokes : -n 5 src/Solver/petsc_altmain
#mpirun -n 1 Schemes/stokes -m 0 -s 0 -k 0 : -n 5 src/Solver/petsc_altmain -info -mat-view-info

