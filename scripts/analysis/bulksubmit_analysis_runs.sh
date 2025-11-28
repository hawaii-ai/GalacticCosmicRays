#!/bin/bash

export FILE='/home/linneamw/sadow_koastore/personal/linneamw/research/gcr/GalacticCosmicRays/scripts/analysis/plot_hmc_train_size_all_runs.py'
sbatch --export=ALL analyze.slurm

export FILE='/home/linneamw/sadow_koastore/personal/linneamw/research/gcr/GalacticCosmicRays/scripts/analysis/plot_hmc_train_size_individual_runs.py'
sbatch --export=ALL analyze.slurm

export FILE='/home/linneamw/sadow_koastore/personal/linneamw/research/gcr/GalacticCosmicRays/scripts/analysis/plot_hmc_train_size_one_run_summary.py'
sbatch --export=ALL analyze.slurm

export FILE='/home/linneamw/sadow_koastore/personal/linneamw/research/gcr/GalacticCosmicRays/scripts/analysis/plot_hmc_train_size_ppc.py'
sbatch --export=ALL analyze.slurm