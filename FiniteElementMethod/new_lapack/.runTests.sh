#!/bin/bash

echo -ne "\n\nWaiting for job to start...\n\n"

echo -ne "==================\n" 
echo -ne "Starting execution\n" 
echo -ne "==================\n\n"

for i in 10 50 100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000
do    
    echo -ne "==================\n"
    echo -ne "Starting Test for $i\n" 
    echo -ne "==================\n\n" 
    
    ./main $i >> HOST_results.txt
done

echo -ne "\n==================\n"
echo -ne "Finished execution\n" 
echo -ne "==================\n\n" 
echo "Hit Ctrl + C to exit..."
