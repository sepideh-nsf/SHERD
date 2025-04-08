#!/bin/bash


#for bunch in {50,100,200,500};
#do
bunch=500
  for tau in {10,30,50,70};
  do 
# a

    #cas5=tb${tau}_50_2;
    cas5=tb${tau}_${bunch}_2;
	echo $cas5
	rm -r $cas5/Evaluate
	cp -rp backup/Evaluate $cas5/.
	
	#cp -p backup/Evaluate/batch_run_Eval.sh $cas5/Evaluate/.
	#cp -p backup/Evaluate/Job_Eval.sh $cas5/Evaluate/.
	
	cd $cas5/Evaluate/
      sed -i 's,''jobname'','"$cas5"',' Job_Eval.sh
      sbatch Job_Eval.sh
	  sleep 5m
	
	cd ../../
  done;
#done;