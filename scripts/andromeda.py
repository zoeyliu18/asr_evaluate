import io
for i in range(1, 35):
	with io.open('hupa_second_dates' + str(i) + '.pbs', 'w', encoding = 'utf-8') as f:
		f.write('#!/bin/bash' + '\n')
		f.write("#SBATCH --job-name='hs da" + str(i) + "'" + '\n')
		f.write('#SBATCH --ntasks 1 --cpus-per-task 4' + '\n')
		f.write('#SBATCH --mem=55gb' + '\n')
		f.write('#SBATCH --time=48:00:00' + '\n')
		f.write('#SBATCH --mail-type=BEGIN,END,FAIL.' + '\n')
		f.write('#SBATCH --partition=gpuv100' + '\n')
		f.write('module load slurmExtras/1.0' + '\n')
		f.write('module load anaconda' + '\n')
		f.write('module load cuda10.2' + '\n')
		f.write('module load kaldi' + '\n')
		f.write('module load cudnn7.6-cuda10.2' + '\n')
		f.write('module load pytorch/1.7.0gpu' + '\n')
		f.write('cd /data/liuaal/kaldi/' + '\n')
		f.write('bash hupa_second_dates' + str(i) + '.sh' + '\n')
		f.write('\n')


for i in range(1, 37):
	with io.open('swahili_heldout' + str(i) + '.pbs', 'w', encoding = 'utf-8') as f:
		f.write('#!/bin/tcsh' + '\n')
		f.write('#PBS -l mem=35gb,walltime=24:00:00,advres=gpgpu2' + '\n')
		f.write('#PBS -m abe -M liuaal@bc.edu' + '\n')
		f.write('module load cuda10.0/toolkit/10.0.130' + '\n')
		f.write('module load kaldi/5.5.6gpu'  + '\n')
		f.write('module load intel/2020' + '\n')
		f.write('module load gcc/7.2.0' + '\n')
		f.write('module load gnu_gcc/7.4.0' + '\n')
		f.write('cd /gsfs0/data/liuaal/kaldi/' + '\n')
		f.write('bash swahili_heldout' + str(i) + '.sh' + '\n')
		f.write('\n')
