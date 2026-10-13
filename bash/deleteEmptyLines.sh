awk 'NF' $1 > ${1}_temp
mv ${1}_temp $1