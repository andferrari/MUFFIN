# easy_muffin

## Small Demos
- [example_class.py](example_class.py): executes centralized version of  `muffin`
- `example_class_mpi.py`: executes & compares the centralized & distributed versions of `muffin` using MPI
```
mpirun --np 3 python example_class_mpi.py
```
- `example_class_mpi_sure.py`: executes & compares the centralized & distributed version of `muffin` + SURE 

## Testing
- `Run_test_mpi2.py`: executes & saves results of the distributed `muffin` with ability to set the arguments from terminal
``` 
mpirun --np 44  python3 Run_tst_mpi2.py -L 256 -N 2000 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -N_dct 0 -data M31_skyline2_30dbmpirun --np 44  python3 Run_tst_mpi2.py -L 256 -N 2000 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -N_dct 0 -data M31_skyline2_30dbmpirun --np 44  python3 Run_tst_mpi2.py -L 256 -N 2000 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -N_dct 0 -data M31_skyline2_30db
```
- `Figures2_Run_tst_mpi2.py`: Loads results saved in `Run_test_mpi2.py` and plots results Figures

## Muffin codes
- `SuperNiceSpectraDeconv.py` : original code with the `muffin` iterative algorithms
- `deconv3d.py` : code for the `muffin` iterative algorithm written using a class framework
- `deconv3D_mpi2.py` : distributed version of `deconv3d.py` using MPI
- `deconv3d_tools.py` : module with fonctions called by `deconv3d.py` and `deconv3D_mpi2.py`
