#!/bin/bash


# a
#for i2 in {1..5};
#do 
	#cas1=Train${i2};
	#echo $cas1
	cp -rp ../Train/Train1 .
    
	cd Train1
	
	 for i1 in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55};
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