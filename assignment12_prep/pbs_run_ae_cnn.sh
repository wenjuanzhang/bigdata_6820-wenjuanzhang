#PBS -N ae_cnn
#PBS -l walltime=0:59:00
#PBS -l nodes=1:ppn=10
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2
python -u ae_cnn_emnist.py >& ae_cnn.log

