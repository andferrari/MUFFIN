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
- [run_tst_mpi.py](easy_muffin_py/run_tst_mpi.py): executes & saves results of the distributed `muffin`. You have to set the algorithm parameters from the terminal. 
``` 
mpirun --np 3  python3 run_tst_mpi.py -L 2 -N 4 -mu_s 0.2 -mu_l 7 -mu_w 10 -stp_s 0.3 -stp_l 10000 -data m31_3d_conv_10db -fol /home/rammanouil/Bureau/easy_muffin/easy_muffin_py/data -pxl_w 1 -bnd_w 1
```
- [plot_figures_Run_tst_mpi.py](easy_muffin_py/plot_figures_Run_tst_mpi.py): Loads results saved in `run_test_mpi.py` and plots  Figures (SNR, PSNR, MSE, cost, SUGAR, ... )
```
python3 plot_figures_Run_tst_mpi.py -res_fol /home/rammanouil/Bureau/easy_muffin/easy_muffin_py/output/1881888
```

## Testing MUFFIN on a cluster using SLURM  
- [run_sigamm.py](easy_muffin_py/run_sigamm.py): adapted version of [run_tst_mpi.py](easy_muffin_py/run_tst_mpi.py) to run on a cluster using SLURM
- [run_sigamm.slurm](easy_muffin_py/run_sigamm.slurm): sets the number of nodes, and all algorithm parameters, and wall time for running [run_sigamm.py](easy_muffin_py/run_sigamm.py) on a cluster using SLURM 
```
 ssh licallo.oca.eu
 cd ../../SCRATCH/rammanouil/easy_muffin/easy_muffin_py/
 arguments=" -s 1 -L 256 -N 5000 -mu_s 0.005 -mu_l 3 -mu_w 10 -stp_s 0 -stp_l 0 -pxl_w 1 -bnd_w 1 -data M31_skyline220db -fol data/data_david -sav 1 -init 0 -fol_init output_sigamm/7844989"; export arguments
 sbatch run_sigamm.slurm
```

## MUFFIN algorithm & related functions 
- [super_nice_spectra_deconv.py](easy_muffin_py/super_nice_spectra_deconv.py) : original code with the `muffin` iterative algorithms
- [deconv3d.py](easy_muffin_py/deconv3d.py)  : code for the `muffin` iterative algorithm written using a class framework
- [deconv3d_mpi.py](easy_muffin_py/deconv3d_mpi.py)  : distributed version of `deconv3d.py` using MPI
- [deconv3d_tools.py](easy_muffin_py/deconv3d_tools.py)  : module with fonctions called by `deconv3d.py` and `deconv3d_mpi.py`
