#!/bin/bash

module load blas
module load cuda
module load lapack

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/uahclsc0016/FiniteElementMethod/lapack_cuda/build/lib

echo -ne "\n\nWaiting for job to start...\n\n"

echo -ne "==================\n" 
echo -ne "Starting execution\n" 
echo -ne "==================\n\n"

# For performance
for i in 10 50 100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000
do    
    echo -ne "==================\n"
    echo -ne "Starting Test for $i\n" 
    echo -ne "==================\n\n" 
    
    build/bin/lapack_test $i >> CPU_results.txt
done

# For validation COMMENT/UNCOMMENT as needed
# build/bin/lapack_test 5 > CUDA_points.txt

# build/bin/lapack_test 5 > HOST_points.txt

echo -ne "\n==================\n"
echo -ne "Finished execution\n" 
echo -ne "==================\n\n" 
echo "Hit Ctrl + C to exit..."
