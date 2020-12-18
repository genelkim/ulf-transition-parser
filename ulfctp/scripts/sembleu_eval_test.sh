#!/bin/bash
#SBATCH --time=1:00:00 --output=eval.out --error=eval.err
#SBATCH --mem=5GB
#SBATCH -c 5

smatch_dir=$1
cp $2 ${smatch_dir}
cp $3 ${smatch_dir}
cd ${smatch_dir}
gold=$(basename $2)
pred=$(basename $3)
out=`python3 sembleu/eval.py "$gold" "$pred"`
out=($out)
rm $gold $pred
echo ${out[4]}