#!/bin/bash

echo
echo "this script must be run using 'source'"
echo


ACCOUNT=tra24_astrophd
if [ -n "$1" ]; then
	ACCOUNT=$1
fi

# SE LEONARDO FUNZIONA
PARTITION=boost_usr_prod
# ALTRIMENTI G100
#PARTITION=g100_usr_interactive

SRUN="srun --account $ACCOUNT --partition $PARTITION --time 08:00:00 --nodes 1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --mem=123000 --job-name=astro-phd --pty /bin/bash"
echo $SRUN
eval $SRUN
