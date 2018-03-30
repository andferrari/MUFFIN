# easy_muffin

## Verification Demos for centralized and distributed versions of MUFFIN
- [example_class.py](easy_muffin_py/example_class.py): executes centralized version of  `muffin` takes as input the path to the data and the prefix of the data. The data is supposed to be named : prefix_dirty prefix_sky prefix_psf. 
``` 
python3 example_class.py -fol /home/rammanouil/Bureau/easy_muffin/easy_muffin_py/data -nam m31_3d_conv_10db
``` 

- [example_class_mpi_sure.py](easy_muffin_py/example_class_mpi_sure.py): executes & compares the centralized & distributed version of `muffin` + SURE. Similarly to [example_class.py](easy_muffin_py/example_class.py) it takes as input the path to the data folder and the prefix of the data.
```
mpirun --np 3  python3 run_tst_mpi.py -L 2 -N 4 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -data m31_3d_conv_10db -fol /home/rammanouil/Bureau/easy_muffin/easy_muffin_py/data -pxl_w 1 -bnd_w 1
m31_3d_conv_10db
```

## Running Tests of MUFFIN on a node 
- [run_tst_mpi.py](easy_muffin_py/run_tst_mpi.py): executes & saves results of the distributed `muffin`. You have to set the algorithm parameters from the terminal. 
``` 
mpirun --np 44  python3 Run_tst_mpi2.py -L 256 -N 2000 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -N_dct 0 -data m31_3d_conv_10db -fol /home/rammanouil/Bureau/easy_muffin/easy_muffin_py/data
```
- [plot_figures_Run_tst_mpi.py](easy_muffin_py/plot_figures_Run_tst_mpi.py): Loads results saved in `run_test_mpi.py` and plots  Figures (SNR, PSNR, MSE, cost, SUGAR, ... )
```
python3 plot_figures_Run_tst_mpi.py -res_fol /home/rammanouil/Bureau/easy_muffin/easy_muffin_py/output/1881888
```

## Testing MUFFIN on a cluster using SLURM  
- [run_tst_mpi_sigamm.py](easy_muffin_py/run_tst_mpi_sigamm.py): adapted version of [run_tst_mpi.py](easy_muffin_py/run_tst_mpi.py) to run on a cluster using SLURM
- [run_batch_test.slurm](easy_muffin_py/run_batch_test.slurm): sets the number of nodes, wall time for running [run_test_mpi_sigamm.py](easy_muffin_py/run_test_mpi_sigamm.py) on a cluster using SLURM 
```
sbatch Run_batch_test.slurm 
```

## MUFFIN algorithm & related functions 
- [super_nice_spectra_deconv.py](easy_muffin_py/super_nice_spectra_deconv.py) : original code with the `muffin` iterative algorithms
- [deconv3d.py](easy_muffin_py/deconv3d.py)  : code for the `muffin` iterative algorithm written using a class framework
- [deconv3d_mpi.py](easy_muffin_py/deconv3d_mpi.py)  : distributed version of `deconv3d.py` using MPI
- [deconv3d_tools.py](easy_muffin_py/deconv3d_tools.py)  : module with fonctions called by `deconv3d.py` and `deconv3d_mpi.py`
