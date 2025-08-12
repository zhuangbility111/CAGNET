#!/bin/bash
# CAGNET Distributed Benchmark Runner for A100 GPUs
# Adapted for benchmark_broad_func_one5d.py and benchmark_broad_func_oned.py

echo "=== CAGNET Distributed Benchmark Runner ==="

# GPUs per node
# NPERNODE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
NPERNODE=8

# total GPUs
NPROC=$NPERNODE
if [ "${NHOSTS}" != "" ]; then
	    NPROC=$(expr $NHOSTS '*' $NPERNODE)
fi

export MASTER_ADDR=$(hostname -i | awk '{print $1}')
export MASTER_PORT=12345

# Data directory path
DATA_DIR="/hs/work0/home/users/chen.zhuang/distributed_spmm/data/binary"

# Available matrices - choose one or modify as needed
# MATRIX_NAME=${MATRIX_NAME:-"arabic-2005"}
MATRIX_NAME=${MATRIX_NAME:-"uk-2002"}
# Other available options: mouse_gene, delaunay_n24, europe_osm, GAP-web, Queen_4147, twitter7, uk-2002

# Benchmark parameters
NUM_FEATURES=${NUM_FEATURES:-64}
NUM_RUNS=${NUM_RUNS:-10}
REPLICATION=${REPLICATION:-2}
BENCHMARK_TYPE=${BENCHMARK_TYPE:-"both"}  # Options: "1d", "1.5d", "both"

echo "Configuration:"
echo "  Total processes: $NPROC"
echo "  Processes per node: $NPERNODE"
echo "  Matrix: $MATRIX_NAME"
echo "  Data directory: $DATA_DIR"
echo "  Features: $NUM_FEATURES"
echo "  Runs: $NUM_RUNS"
echo "  Replication factor: $REPLICATION"
echo "  Benchmark type: $BENCHMARK_TYPE"
echo ""

# Set OpenMPI environment variables for better performance
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_ALLOW_RUN_AS_ROOT=1
export PSM2_CUDA=1
export PSM2_GPUDIRECT=1

# Common MPI parameters
MPI_PARAMS="-x LD_LIBRARY_PATH -x PATH -x PSM2_CUDA -x PSM2_GPUDIRECT -x MASTER_ADDR -x MASTER_PORT \
           -x MODULUS_DISTRIBUTED_INITIALIZATION_METHOD=OPENMPI \
           -n $NPROC -npernode $NPERNODE --bind-to none"

# Run 1D benchmark
if [[ "$BENCHMARK_TYPE" == "1d" || "$BENCHMARK_TYPE" == "both" ]]; then
    echo "Running 1D CAGNET benchmark (benchmark_broad_func_oned.py)..."
    
    # Run sparse-aware version
    echo "  -> Running sparse-aware version..."
    mpirun $MPI_PARAMS \
        python benchmark_broad_func_oned.py \
        --npz-file "$DATA_DIR/$MATRIX_NAME.npz" \
        --num-features $NUM_FEATURES \
        --num-runs $NUM_RUNS \
        --distributed > debug_1d_sparse_aware.log 2>&1
    
    echo "  -> Sparse-aware completed. Check debug_1d_sparse_aware.log for details."
    
    # Run sparse-unaware version
    echo "  -> Running sparse-unaware version..."
    mpirun $MPI_PARAMS \
        python benchmark_broad_func_oned.py \
        --npz-file "$DATA_DIR/$MATRIX_NAME.npz" \
        --num-features $NUM_FEATURES \
        --num-runs $NUM_RUNS \
        --sparse-unaware \
        --distributed > debug_1d_sparse_unaware.log 2>&1
    
    echo "  -> Sparse-unaware completed. Check debug_1d_sparse_unaware.log for details."
    echo ""
fi

# Run 1.5D benchmark  
if [[ "$BENCHMARK_TYPE" == "1.5d" || "$BENCHMARK_TYPE" == "both" ]]; then
    echo "Running 1.5D CAGNET benchmark (benchmark_broad_func_one5d.py)..."
    
    # Run sparse-aware version
    echo "  -> Running sparse-aware version..."
    mpirun $MPI_PARAMS \
        python benchmark_broad_func_one5d.py \
        --npz-file "$DATA_DIR/$MATRIX_NAME.npz" \
        --num-features $NUM_FEATURES \
        --num-runs $NUM_RUNS \
        --replication $REPLICATION \
        --distributed > debug_1_5d_sparse_aware.log 2>&1
    
    echo "  -> Sparse-aware completed. Check debug_1_5d_sparse_aware.log for details."
    
    # Run sparse-unaware version
    echo "  -> Running sparse-unaware version..."
    mpirun $MPI_PARAMS \
        python benchmark_broad_func_one5d.py \
        --npz-file "$DATA_DIR/$MATRIX_NAME.npz" \
        --num-features $NUM_FEATURES \
        --num-runs $NUM_RUNS \
        --replication $REPLICATION \
        --sparse-unaware \
        --distributed > debug_1_5d_sparse_unaware.log 2>&1
    
    echo "  -> Sparse-unaware completed. Check debug_1_5d_sparse_unaware.log for details."
    echo ""
fi

echo "Benchmark completed successfully!"
echo ""
echo "Log files generated:"
if [[ "$BENCHMARK_TYPE" == "1d" || "$BENCHMARK_TYPE" == "both" ]]; then
    echo "  - debug_1d_sparse_aware.log (1D sparse-aware)"
    echo "  - debug_1d_sparse_unaware.log (1D sparse-unaware)"
fi
if [[ "$BENCHMARK_TYPE" == "1.5d" || "$BENCHMARK_TYPE" == "both" ]]; then
    echo "  - debug_1_5d_sparse_aware.log (1.5D sparse-aware)"
    echo "  - debug_1_5d_sparse_unaware.log (1.5D sparse-unaware)"
fi
echo ""
echo "Usage examples:"
echo "# Run both 1D and 1.5D benchmarks with default matrix (arabic-2005):"
echo "./run_distributed_a100.sh"
echo ""
echo "# Run only 1D benchmark with mouse_gene matrix:"
echo "MATRIX_NAME=mouse_gene BENCHMARK_TYPE=1d ./run_distributed_a100.sh"
echo ""
echo "# Run only 1.5D benchmark with different replication factor:"
echo "MATRIX_NAME=twitter7 BENCHMARK_TYPE=1.5d REPLICATION=4 ./run_distributed_a100.sh"
echo ""
echo "# Custom configuration:"
echo "MATRIX_NAME=GAP-web NUM_FEATURES=256 NUM_RUNS=5 REPLICATION=2 ./run_distributed_a100.sh"

