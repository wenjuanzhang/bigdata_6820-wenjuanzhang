#PBS -N calcviz_vgg16_L1
#PBS -l walltime=0:59:00
#PBS -l nodes=1:ppn=10:gpus=1
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2
python -u calcviz_vgg16.py --layer 1 >& calcviz_vgg16_L1.log

