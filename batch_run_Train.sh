#!/bin/bash

for tau in {10,30,50,70};
do 
# a
  for bunch in {50,100,200,500};
  do
    cas5=tb${tau}_${bunch}_2;
	echo $cas5
    mkdir $cas5
	cp -rp backup/. $cas5/.
	
	cd $cas5/Train/
	
    for i1 in {1..5};
    do 
	
	    cas=Train${i1};
	    echo $cas
        mkdir $cas
	    cp -r copy_Train/. $cas

        cd $cas
		
		cas6=k${tau}_${bunch}_${i1}
		echo $cas6
		
        sed -i 's,''jobname'','"$cas6"',' Job.sh
		sed -i 's,''TAUVAR'','"${tau}"',' Our_method_train.py
		sed -i 's,''BUNCHVAR'','"${bunch}"',' Our_method_train.py

		sbatch Job.sh
		cd ..
		
		sleep 3s
    done;
	
	cd ../../
  done;
done;