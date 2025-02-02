# HArDCore3D-Stokes

Fork of HArDCore3D adding support for the Stokes complex (https://arxiv.org/abs/2112.03125). See the original description below.

This fork also introduce the option to save assembled operators in a file for later use and a rudimentary interface to PETSc. 
The HArDCore library make heavy use of threads parallelism, hence the transition toward the MPI paradigm used in PETSc is far from trivial.
Currently it is achieved by introducing another main (found in src/Solver) acting as a basic state machine answering commands from the main process.
The resolution is done in two steps: first the system is assembled in parallel (multi-threaded) by a single process, the other being idle.
Then all threads are joined, data is spread accross all process and the system is solved in parallel with PETSc over MPI.

This is effectively a Multiple Program Multiple Data execution model. 
Examples are available in Schemes/Stokes.
MPI is enable by setting "WITH_PETSC_MPI" in CMakeList.txt. It will most likely requires to hint the correct installation path of PETSc.

# HArDCore3D
HArD::Core3D (Hybrid Arbitrary Degree::Core 3D) - Library to implement schemes with face and cell polynomial unknowns on 3D generic meshes.

This is the 3D version of the HArD::Core library (https://github.com/jdroniou/HArDCore). The documentation can be found at https://jdroniou.github.io/HArDCore3D-release/

The quadrature (src/Quadratures) module in HArDCore3D is partially based on Marco Manzini's code available at https://github.com/gmanzini-LANL/PDE-Mesh-Manager. A previous version of the mesh builder also used Marco's code, but we have since then developed a specific mesh builder in HArDCore3D.

The purpose of HArD::Core3D is to provide easy-to-use tools to code hybrid schemes, such as the Hybrid High-Order method. The data structure is described using intuitive classes that expose natural functions we use in the mathematical description of the scheme. For example, each mesh element is a member of the class 'Cell', that gives access to its diameter, the list of its faces (themselves members of the class 'Face' that describe the geometrical features of the face), etc. Functions are also provided to compute the key elements usually required to implement hybrid schemes, such as mass matrices of local basis functions, stiffness matrices, etc. The approach adopted is that of a compromise between readibility/usability and efficiency. 

As an example, when creating a mass matrix, the library requires the user to first compute the quadrature nodes and weights, then compute the basis functions at these nodes, and then assemble the mass matrix. This ensures a local control on the required degree of exactness of the quadrature rule, and also that basis functions are not evaluated several times at the same nodes (once computed and stored locally, the values at the quadrature nodes can be re-used several times). Each of these steps is however concentrated in one line of code, so the assembly of the mass matrix described above is actually done in three lines:

```
QuadratureRule quadT = generate_quadrature_rule(T, 2*m_K);<br>
boost::multi_array<double, 2> phiT_quadT = evaluate_quad<Function>::compute(basisT, quadT);<br>
Eigen::MatrixXd MTT = compute_gram_matrix(phiT_quadT, quadT);
```

Note that the `ElementQuad` class offers a convenient way to compute and store the quadrature rules and values of basis functions at the nodes, and makes it easy to pass these data to functions. More details and examples are provided in the documentation.

The implementations in this library follow general principles described in the appendix of the book "*The Hybrid High-Order Method for Polytopal Meshes: Design, Analysis, and Applications*" (D. A. Di Pietro and J. Droniou. 2019, 516p. url: https://hal.archives-ouvertes.fr/hal-02151813). High-order methods with hybrid unknowns have certain specificities which sometimes require fine choices, e.g. of basis functions (hierarchical, orthonormalised or not), etc. We refer to this manuscript for discussions on these specificities. When using the HArDCore library for a scientific publication, please refer to this book. Some modules of the library have been developed for specific scientific articles; the README.txt file in the corresponding directories provide the details of these articles, which you are kindly requested to refer to if you use these modules.


This library was developed with the direct help and indirect advice of several people. Many thanks to them: Daniel Anderson, Lorenzo Botti, Hanz Martin Cheng, Daniele Di Pietro, Daniel Jackson, Marco Manzini, Liam Yemm.

The development of this library was partially supported by Australian Government through the Australian Research Council's Discovery Projects funding scheme (project number DP170100605).
