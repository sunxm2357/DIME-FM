#!/bin/bash

sbatch cifar10.sh $1
sbatch cifar100.sh $1
sbatch country211.sh $1
sbatch food101.sh $1
sbatch small_datasets1.sh $1
sbatch small_datasets2.sh $1

