# easy_muffin

## Verification Demos for centralized and distributed versions of MUFFIN
- [example_class.py](easy_muffin_py/example_class.py): executes centralized version of  `muffin` takes as input the path to the data and the prefix of the data. The data is supposed to be named : prefix_dirty prefix_sky prefix_psf. 
``` 
python3 example_class.py -fol /home/rammanouil/Bureau/easy_muffin/easy_muffin_py/data -nam m31_3d_conv_10db
``` 

- [example_class_mpi_sure.py](easy_muffin_py/example_class_mpi_sure.py): executes & compares the centralized & distributed version of `muffin` + SURE. Similarly to [example_class.py](easy_muffin_py/example_class.py) it takes as input the path to the data folder and the prefix of the data.
```
mpirun --np 6 python3 example_class_mpi_sure.py -fol /home/rammanouil/Bureau/easy_muffin/easy_muffin_py/data -nam m31_3d_conv_10db
```

## Running Tests of MUFFIN on a node 
- [Run_test_mpi.py](easy_muffin_py/Run_test_mpi.py): executes & saves results of the distributed `muffin`. You have to set the algorithm parameters from the terminal. 
``` 
mpirun --np 44  python3 Run_tst_mpi2.py -L 256 -N 2000 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -N_dct 0 -data M31_skyline2_30dbmpirun --np 44  python3 Run_tst_mpi2.py -L 256 -N 2000 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -N_dct 0 -data M31_skyline2_30dbmpirun --np 44  python3 Run_tst_mpi2.py -L 256 -N 2000 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -N_dct 0 -data M31_skyline2_30db
```
- [Figures2_Run_tst_mpi2.py](Run_test_mpi2.py/Figures2_Run_tst_mpi2.py): Loads results saved in `Run_test_mpi2.py` and plots  Figures (SNR, PSNR, MSE, cost, SUGAR, restored image ... )

## Testing MUFFIN on a cluster using SLURM  
- [Run_test_mpi_sigamm.py](easy_muffin_py/Run_test_mpi_sigamm.py): adapted version of [Run_test_mpi2.py](easy_muffin_py/Run_test_mpi2.py) to run on a cluster using SLURM
- [Run_batch_test.slurm](easy_muffin_py/Run_batch_test.slurm): sets the number of nodes, wall time for running [Run_test_mpi_sigamm.py](easy_muffin_py/Run_test_mpi_sigamm.py) on a cluster using SLURM 
```
sbatch Run_batch_test.slurm 
```

## MUFFIN algorithm & related functions 
- [SuperNiceSpectraDeconv.py](easy_muffin_py/SuperNiceSpectraDeconv.py) : original code with the `muffin` iterative algorithms
- [deconv3d.py](easy_muffin_py/deconv3d.py)  : code for the `muffin` iterative algorithm written using a class framework
- [deconv3D_mpi2.py](easy_muffin_py/deconv3D_mpi2.py)  : distributed version of `deconv3d.py` using MPI
- [deconv3d_tools.py](easy_muffin_py/deconv3d_tools.py)  : module with fonctions called by `deconv3d.py` and `deconv3D_mpi2.py`
