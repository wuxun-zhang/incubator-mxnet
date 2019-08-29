opt1=$1
opt2=$2
opt3=$3

phy_cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print$4}')
omp_threads=$phy_cpus
echo "Setting OMP_NUM_THREADS to "$omp_threads
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=$omp_threads

bind_core=$[$omp_threads-1]
echo "Binding process to core : 0-"$bind_core

if [[ ${opt1} = 'mkldnn'  ]];then
   mkldnn_option='mkldnn'
   enable_mkldnn=1
else
   mkldnn_option='non-mkldnn'
   enable_mkldnn=0
fi

if [[ ${opt2} = 'int64'  ]];then
   tensor_type='int64'
   enable_int64=1
else
   tensor_type='int32'
   enable_int64=0
fi

if [[ ${opt3} = 'build'  ]];then
   need_build=1
else
   need_build=0
fi

if [ ${need_build} -eq 1 ];then
   cd ../../
   pwd
   make clean
   make USE_MKLDNN=${enable_mkldnn} USE_INT64_TENSOR_SIZE=${enable_int64} USE_BLAS=mkl USE_INTEL_PATH=/opt/intel/ -j
   cd benchmark/opperf
fi


output_dir=./logs/${tensor_type}_${mkldnn_option}
if [ ! -d ${output_dir} ];then
   echo "Creating directory: "${output_dir}
   mkdir -p ${output_dir}
fi

export PYTHONPATH=$(pwd)/../../:$(pwd)/../../python:$PYTHONPATH
echo "PYTHONPATH: "$PYTHONPATH


taskset -c 0-$bind_core python opperf.py -p 'python' -o ./logs/mxnet_operator_benchmarks.json --tensor-size-option ${tensor_type} --mkldnn-option ${mkldnn_option}
