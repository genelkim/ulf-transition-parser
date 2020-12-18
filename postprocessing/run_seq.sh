#!/bin/sh
#If sequence to AMR
#python ./amr_format.py --data_dir ./dev_data --amrseq_file ../smatch_2.0.2/dev.thirdrun.amr
#python ./amr_format.py --data_dir ./data_prep/dev_categorized --amrseq_file ../smatch_2.0.2/dev.new.amr
python ./amr_format.py --data_dir ../data_prep/final_data/test --amrseq_file ../../smatch_2.0.2/test.final1.amr
#python ./amr_format.py --data_dir ./data_prep/final_data/dev --amrseq_file ../smatch_2.0.2/dev.final.amr

#rm $INPUT.tmp
