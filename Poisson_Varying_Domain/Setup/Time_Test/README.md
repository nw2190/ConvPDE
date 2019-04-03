# Performing Time Tests

The time comparisons for the FEniCS, FEniCS (coarse mesh), network, and network (GPU) can be carried out using the following commands:

```console
$ ./GENERATE_DATA.sh

 [ Sampling Functions ]

  Progress:  100.0%


 [ Converting Functions ]
 
  Progress:  100.0%


 [ Generating Meshes ]

  Progress:  100.0%


 [ Preprocessing Data ]
 
  Progress:  100.0%


$ ./RUN_TIME_TESTS.sh

...

 [ AVERAGE TIME RESULTS ]

 Average FEniCS Time:            0.04490 seconds
 Average FEniCS Time (Coarse):   0.03382 seconds

 Average Network Time (CPU):     0.01025 seconds
 Average Network Time (GPU):     0.00173 seconds

```
