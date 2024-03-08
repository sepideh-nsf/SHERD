#!/bin/bash


# a
#for i2 in {1..5};
#do 
	#cas1=Train${i2};
	#echo $cas1
	cp -rp ../Train .
    
	cd Train
	
	 for i1 in {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
     do 
	
	 # phi
         for i3 in {0.1,0.2,0.3,0.4,0.5};
         do
	
	         cas=z${i1}_${i3};
	         echo $cas
             mkdir $cas
	         cp -r ../Newfolder/copy/. $cas/.

             cd $cas
              sed -i 's,''jobname'','"$cas"',' Job.sh
	   	      sed -i 's,''SplitVar'','"${i1}"',' Evaluation.py
	 	      sed -i 's,''CompVar'','"${i3}"',' Evaluation.py

		      sbatch Job.sh
		     cd ..
		
		     sleep 0.5s

         done;
	 done;
    
#    cd ../
#done;