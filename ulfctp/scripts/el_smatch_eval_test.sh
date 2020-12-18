#!/bin/bash

smatch_dir=$1
cp $2 ${smatch_dir}
cp $3 ${smatch_dir}
cd ${smatch_dir}
gold=$(basename $2)
pred=$(basename $3)
out=`python3 evaluate_el_smatch.py --gold_file "$gold" --hypo_file "$pred" --out out.tmp --vout vout.tmp`
out=($out)
rm $gold $pred
echo ${out[0]} ${out[1]} ${out[2]} 

