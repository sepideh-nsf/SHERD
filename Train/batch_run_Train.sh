#!/bin/bash


# a
for i1 in {1..5};
do 
	
	    cas=Train${i1};
	    echo $cas
        mkdir $cas
	    cp -r copy_Train/. $cas

        cd $cas
        sed -i 's,''jobname'','"$cas"',' Job.sh

		sbatch Job.sh
		cd ..
		
		sleep 5s
done;