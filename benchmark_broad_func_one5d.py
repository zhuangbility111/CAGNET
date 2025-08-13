#!/usr/bin/env python3
"""
Performance benchmarking script for broad_func_one5d function.
This script extracts and tes    if len(row_data_send) == 0: # not q for either stage
        row_data_send = [torch.cuda.DoubleTensor(device=gcn_instance["device"])]*len(col_procs) the broad_func_one5d function from the CAGNET project.
"""

import argparse
import math
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict
from itertools import accumulate
import scipy.io as spio
import socket

# Import the sparse extension
try:
    from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu
except ImportError:
    print("Warning: sparse_coo_tensor_cpp not available. Please compile the sparse extension first.")
    print("Run: cd sparse-extension && python setup.py build_ext --inplace")
    exit(1)

# Reuse functions from benchmark_broad_func_oned.py to avoid duplication
from benchmark_broad_func_oned import stop_time, split_coo, load_npz_file, setup_distributed, compute_full_reference_spmm, extract_local_partition, verify_distributed_result

# Extracted broad_func_one5d function - exact copy from original
def broad_func_one5d(gcn_instance, graph, ampbyp, inputs):
    """
    Exact copy of broad_func_one5d function from gcn_conv.py
    gcn_instance is a dict containing: rank, size, device, sparse_unaware, timers, epoch, timings,
                                      replication, node_count, col_groups, row_groups
    """
    
    if gcn_instance["sparse_unaware"]:
        # print("in su code", flush=True)
        n_per_proc = math.ceil(float(gcn_instance["node_count"]) / (gcn_instance["size"] / gcn_instance["replication"]))

        z_loc = torch.cuda.DoubleTensor(ampbyp[0].size(0), inputs.size(1), device=gcn_instance["device"]).fill_(0)

        inputs_recv = torch.cuda.DoubleTensor(n_per_proc, inputs.size(1), device=gcn_instance["device"]).fill_(0)

        rank_c = gcn_instance["rank"] // gcn_instance["replication"]
        rank_col = gcn_instance["rank"] % gcn_instance["replication"]

        stages = gcn_instance["size"] // (gcn_instance["replication"] ** 2)
        if rank_col == gcn_instance["replication"] - 1:
            stages = (gcn_instance["size"] // gcn_instance["replication"]) - (gcn_instance["replication"] - 1) * stages
        
        row_count_sum = 0
        for i in range(stages):
            q = (rank_col * (gcn_instance["size"] // (gcn_instance["replication"] ** 2)) + i) * gcn_instance["replication"] + rank_col

            q_c = q // gcn_instance["replication"]

            am_partid = rank_col * (gcn_instance["size"] // gcn_instance["replication"] ** 2) + i

            if q == gcn_instance["rank"]:
                inputs_recv = inputs.clone()
            elif q_c == gcn_instance["size"] // gcn_instance["replication"] - 1:
                inputs_recv = torch.cuda.DoubleTensor(ampbyp[am_partid].size(1), \
                                                        inputs.size(1), \
                                                        device=gcn_instance["device"]).fill_(0)

            row_count_sum += inputs_recv.numel()
            inputs_recv = inputs_recv.contiguous()
            start = time.time()
            dist.broadcast(inputs_recv, src=q, group=gcn_instance["col_groups"][rank_col])
            stop_time(gcn_instance, "broadcast", start, barrier=False)
            start = time.time()
            spmm_gpu(ampbyp[am_partid].indices()[0].int(), ampbyp[am_partid].indices()[1].int(), 
                            ampbyp[am_partid].values(), ampbyp[am_partid].size(0), 
                            ampbyp[am_partid].size(1), inputs_recv, z_loc)
            stop_time(gcn_instance, "spmm_gpu", start, barrier=False)
        z_loc = z_loc.contiguous()
        start = time.time()
        dist.all_reduce(z_loc, op=dist.reduce_op.SUM, group=gcn_instance["row_groups"][rank_c])
        stop_time(gcn_instance, "reduce", start, barrier=False)
        return z_loc

    """ 1.5d sparse-aware implementation (a2a version) """
    # print("in sa code", flush=True)
    
    z_loc = torch.cuda.DoubleTensor(ampbyp[0].size(0), inputs.size(1), device=gcn_instance["device"]).fill_(0)
    rank = gcn_instance["rank"]
    rank_c = gcn_instance["rank"] // gcn_instance["replication"]
    rank_col = gcn_instance["rank"] % gcn_instance["replication"]

    col_procs = list(range(rank_col, gcn_instance["size"], gcn_instance["replication"]))

    stages = gcn_instance["size"] // (gcn_instance["replication"] ** 2)
    if rank_col == gcn_instance["replication"] - 1:
        stages = (gcn_instance["size"] // gcn_instance["replication"]) - (gcn_instance["replication"] - 1) * stages

    row_data_send = []
    row_indices_recv = gcn_instance["row_indices_recv"]
    row_data_recv = [torch.cuda.DoubleTensor(device=gcn_instance["device"])]*len(col_procs)
    row_indices_send = gcn_instance["row_indices_send"]
    
    start = time.time()
    for i in range(stages):
        q = (rank_col * (gcn_instance["size"] // (gcn_instance["replication"] ** 2)) + i) * gcn_instance["replication"] + rank_col
        q_c = q // gcn_instance["replication"]
        am_partid = rank_col * (gcn_instance["size"] // gcn_instance["replication"] ** 2) + i
        unique_cols = gcn_instance["row_indices_send"][q]
        if rank == q:
            for j in col_procs:
                if rank != j:
                    rows_send = inputs[row_indices_recv[j].long(), :].clone()
                    row_data_send.append(rows_send)
                else:
                    row_data_send.append(torch.cuda.DoubleTensor(device=gcn_instance["device"]))
        else: # receiving data from q
            row_data_recv[q_c] = torch.cuda.DoubleTensor(device=gcn_instance["device"]).resize_((unique_cols.size(0), inputs.size(1))).fill_(0)

    if len(row_data_send) == 0: # not q for either stage
        row_data_send = [torch.cuda.DoubleTensor(device=gcn_instance["device"])]*len(col_procs) 
    stop_time(gcn_instance, "gather_row_data", start, barrier=False)

    start = time.time()
    dist.all_to_all(row_data_recv, row_data_send, group=gcn_instance["col_groups"][rank_col])
    stop_time(gcn_instance, "a2a", start, barrier=False)

    start = time.time()
    for i in range(len(col_procs)):
        if row_data_recv[i].size()[0] != 0:
            inputs_mul = torch.cuda.DoubleTensor(device=gcn_instance["device"]).resize_(ampbyp[i].size(1), inputs.size(1)).fill_(0)
            inputs_mul[row_indices_send[col_procs[i]]] = row_data_recv[i]

            spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(),
                                ampbyp[i].values(), ampbyp[i].size(0),
                                ampbyp[i].size(1), inputs_mul, z_loc)
        elif rank == col_procs[i] and i >= rank_col * (gcn_instance["size"] // (gcn_instance["replication"] ** 2)) and i < rank_col * (gcn_instance["size"] // (gcn_instance["replication"] ** 2)) + stages:
            inputs_mul = inputs.clone()
            spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(),
                                ampbyp[i].values(), ampbyp[i].size(0),
                                ampbyp[i].size(1), inputs_mul, z_loc)
    stop_time(gcn_instance, "spmm_gpu", start, barrier=False)

    z_loc = z_loc.contiguous()
    start = time.time()
    dist.all_reduce(z_loc, op=dist.reduce_op.SUM, group=gcn_instance["row_groups"][rank_c])
    stop_time(gcn_instance, "reduce", start, barrier=False)

    return z_loc

# Process group setup function from gcn_15d.py
def get_proc_groups(rank, size, replication):
    """Setup row and column process groups for 1.5D partitioning"""
    rank_c = rank // replication
     
    row_procs = []
    for i in range(0, size, replication):
        row_procs.append(list(range(i, i + replication)))

    col_procs = []
    for i in range(replication):
        col_procs.append(list(range(i, size, replication)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(dist.new_group(row_procs[i]))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(dist.new_group(col_procs[i]))

    return row_groups, col_groups

# 1.5D partitioning function from gcn_15d.py (modified, normalize removed)
def one5d_partition(rank, size, inputs, adj_matrix, data, features, classes, replication, device, partitions=[]):
    """Modified one5d_partition function from gcn_15d.py (normalize removed)"""
    node_count = inputs.size(0)
    
    if not partitions:
        n_per_proc = math.ceil(float(node_count) / (size // replication))
        partitions = [n_per_proc]*(size // replication)
        partitions[size//replication -1] = inputs.size(0) - math.ceil(float(node_count) / (size//replication))*((size//replication) - 1)  

    am_partitions = None
    am_pbyp = None

    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))
    
    # Swap rows of adj_matrix (swap row indices and column indices)
    # adj_matrix is in format [2, nnz], swap the two rows
    adj_matrix_swapped = torch.stack([adj_matrix[1], adj_matrix[0]], dim=0)
    print(f"rank: {rank} adj_matrix rows swapped for partition processing", flush=True)

    rank_c = rank // replication
    # Compute the adj_matrix and inputs partitions for this process
    with torch.no_grad():
        # Column partitions - now using row-swapped adj_matrix
        am_partitions, vtx_indices = split_coo(adj_matrix_swapped, partitions, 1)
        print(vtx_indices)
        print(rank_c)
        proc_node_count = vtx_indices[rank_c + 1] - vtx_indices[rank_c]
        am_pbyp, _ = split_coo(am_partitions[rank_c], partitions, 0)
        for i in range(len(am_pbyp)):
            if i == size // replication - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1), dtype=torch.float64), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)
                # scale_elements removed
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1), dtype=torch.float64), 
                                                        size=(vtx_indices[i + 1] - vtx_indices[i], proc_node_count),
                                                        requires_grad=False)
                # scale_elements removed

        print("rank: {}, am_pbyp size: {}".format(rank, [x.size() for x in am_pbyp]), flush=True)

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1), dtype=torch.float64), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            # scale_elements removed

        input_partitions = torch.split(inputs, partitions, dim=0)

        adj_matrix_loc = am_partitions[rank_c]
        inputs_loc = input_partitions[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)
    return list(input_partitions), am_partitions, am_pbyp

def split_am_partition(am_partition, partitions, rank, size, replication):
    """Split adjacency matrix partition for 1.5D - from gcn_15d.py (normalize removed)"""
    node_count = am_partition.size(0)
    if not partitions:
        n_per_proc = math.ceil(float(node_count) / (size // replication))
        partitions = [n_per_proc]*(size // replication)
        partitions[size//replication -1] = node_count - math.ceil(float(node_count) / (size//replication))*((size//replication) - 1)  

    am_pbyp = None
    print(am_partition.size())
    proc_node_count = am_partition.size(1) # column count
    proc_node_count_row = int(math.ceil(node_count / (size // replication)))
    am_pbyp, vtx_indices = split_coo(am_partition._indices(), partitions, 0)
    rank_c = size // replication
    rank_col = size % replication
    for i in range(len(am_pbyp)):
        if i == size // replication - 1:
            last_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1), dtype=torch.float64), 
                                                    size=(last_node_count, proc_node_count),
                                                    requires_grad=False)
            # scale_elements removed
        else:
            proc_node_count_row = vtx_indices[i + 1] - vtx_indices[i]
            am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1), dtype=torch.float64), 
                                                    size=(proc_node_count_row, proc_node_count),
                                                    requires_grad=False)
            # scale_elements removed

    return am_pbyp

def benchmark_broad_func_one5d(args):
    """Main benchmarking function for broad_func_one5d"""
    
    # Print benchmark header
    mode_str = "SPARSE-UNAWARE" if args.sparse_unaware else "SPARSE-AWARE"
    print(f"\n=== CAGNET 1.5D Benchmark ({mode_str}) ===")
    
    # Setup distributed environment
    if args.distributed:
        rank, size, local_rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
    else:
        # Single process mode for testing
        rank = 0
        size = 1
        local_rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Process {rank}/{size} on device {device} (local_rank: {local_rank})")
    
    # Validate replication factor
    if size % args.replication != 0:
        raise ValueError(f"World size ({size}) must be divisible by replication factor ({args.replication})")
    
    proc_row = size // args.replication
    rank_c = rank // args.replication
    rank_col = rank % args.replication
    
    if rank_c >= proc_row:
        print(f"Rank {rank} is outside valid range for replication {args.replication}")
        return
    
    # Setup process groups for 1.5D
    if args.distributed:
        row_groups, col_groups = get_proc_groups(rank, size, args.replication)
    else:
        row_groups = [None]
        col_groups = [None]
    
    # Load graph data
    if args.npz_file:
        edge_index, num_nodes, num_edges = load_npz_file(args.npz_file)
    else:
        # Generate synthetic data for testing
        num_nodes = args.num_nodes
        num_edges = args.num_edges
        print(f"Generating synthetic graph: {num_nodes} nodes, {num_edges} edges")
        
        # Generate random edges
        edges = torch.randint(0, num_nodes, (2, num_edges))
        edge_index = edges
    
    # Create input features with fp64 precision
    num_features = args.num_features
    torch.manual_seed(1233)
    inputs = torch.randn(num_nodes, num_features, dtype=torch.float64)
    # inputs = torch.ones(num_nodes, num_features, dtype=torch.float64)
    
    print(f"Input features shape: {inputs.shape}")
    
    # 1.5D Partition the data
    print("Performing 1.5D partitioning...")
    # Create dummy data/features/classes for compatibility
    dummy_data = None
    dummy_features = inputs
    dummy_classes = num_features  # Just use num_features as dummy
    
    input_partitions, am_partitions, ampbyp = one5d_partition(
        rank, size, inputs, edge_index, dummy_data, dummy_features, dummy_classes, 
        args.replication, device
    )
    
    # Get local data
    inputs_loc = input_partitions[rank_c]
    adj_matrix_loc = am_partitions[rank_c]
    
    # Move data to device
    inputs_loc = inputs_loc.to(device)
    # adj_matrix_loc = adj_matrix_loc.to(device)  # Keep on CPU initially
    for i in range(len(ampbyp)):
        ampbyp[i] = ampbyp[i].t().coalesce().to(device)
    
    print(f"Local input shape: {inputs_loc.shape}")
    print(f"Local adjacency matrix shape: {adj_matrix_loc.shape}")
    print(f"Number of ampbyp partitions: {len(ampbyp)}")
    
    # Setup row indices for sparse-aware communication
    if args.distributed:
        col_procs = list(range(rank_col, size, args.replication))
        
        counts_send = []
        row_indices_send = []
        
        for i in range(len(ampbyp)):
            unique_cols = ampbyp[i]._indices()[1].unique()
            for j in range(args.replication):
                row_indices_send.append(unique_cols)
                counts_send.append(torch.cuda.LongTensor([unique_cols.size()], device=device).resize_(1, 1))
        
        counts_recv = [torch.cuda.LongTensor(1, 1, device=device).fill_(0) for i in range(size)]
        
        print("Exchanging row indices information...")
        dist.all_to_all(counts_recv, counts_send)
        
        row_indices_recv = [torch.cuda.LongTensor(device=device).resize_(counts_recv[i].int().item(),).fill_(0) for i in range(len(counts_recv))]
        dist.all_to_all(row_indices_recv, row_indices_send)
    else:
        # For single process, create dummy row indices
        row_indices_send = [torch.cuda.LongTensor(range(inputs_loc.size(0)), device=device)] * size
        row_indices_recv = [torch.cuda.LongTensor(range(inputs_loc.size(0)), device=device)] * size
    
    # Create GCN instance dictionary for compatibility with original function signature
    gcn_instance = {
        "rank": rank,
        "size": size,
        "device": device,
        "sparse_unaware": args.sparse_unaware,
        "timers": True,
        "epoch": 1,
        "timings": defaultdict(float),
        "replication": args.replication,
        "node_count": num_nodes,
        "col_groups": col_groups,
        "row_groups": row_groups,
        "row_indices_send": row_indices_send,
        "row_indices_recv": row_indices_recv
    }
    
    # Benchmark the function
    print("Starting benchmark...")
    
    # Warmup runs
    for _ in range(args.warmup):
        gcn_instance["epoch"] = 0  # Don't record timing for warmup
        _ = broad_func_one5d(gcn_instance, adj_matrix_loc, ampbyp, inputs_loc)
        if args.distributed:
            dist.barrier()
    
    # Clear timings after warmup
    gcn_instance["timings"].clear()
    
    # Benchmark runs
    start_time = time.time()
    
    for epoch in range(1, args.num_runs + 1):
        gcn_instance["epoch"] = epoch
        result = broad_func_one5d(gcn_instance, adj_matrix_loc, ampbyp, inputs_loc)
        if args.distributed:
            dist.barrier()
    
    total_time = time.time() - start_time
    avg_time = total_time / args.num_runs
    
    print(f"Results for rank {rank}:")
    print(f"Total time for {args.num_runs} runs: {total_time:.4f}s")
    print(f"Average time per run: {avg_time:.4f}s")
    print(f"Output shape: {result.shape}")
    
    if gcn_instance["timings"]:
        print("Detailed timings (total):")
        for key, value in gcn_instance["timings"].items():
            print(f"  {key}: {value:.4f}s ({value/args.num_runs:.4f}s avg)")
    
    # Verify distributed result against reference computation (1.5D specific)
    if args.verify_result and args.distributed:
        print(f"\n=== Verifying Local 1.5D Distributed Results (Rank {rank}) ===")
        try:
            # Each rank computes the full reference result using the original edge_index format
            full_reference_result = compute_full_reference_spmm(edge_index, num_nodes, inputs, device)
            
            # For 1.5D, we need to consider the replication factor
            # Extract the local partition that this rank should compute
            # Note: 1.5D partitioning may be more complex than simple row partitioning
            expected_local_result = extract_local_partition(full_reference_result, rank_c, proc_row)
            
            # Verify the local result (z_loc) against expected local result
            is_correct, max_diff, rel_diff = verify_distributed_result(
                result, expected_local_result, rank
            )
            
            if is_correct:
                print(f"🎉 Rank {rank}: Local 1.5D distributed computation is CORRECT!")
            else:
                print(f"⚠️  Rank {rank}: Local 1.5D distributed computation has discrepancies!")
            
        except Exception as e:
            print(f"❌ Rank {rank}: 1.5D Verification failed with error: {e}")
            import traceback
            traceback.print_exc()
    elif args.verify_result and not args.distributed:
        print("⚠️  Verification only available in distributed mode")
    
    if args.distributed:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Benchmark broad_func_one5d function')
    
    # Data options
    parser.add_argument('--npz-file', type=str, help='Path to NPZ sparse matrix file')
    # parser.add_argument('--num-nodes', type=int, default=10000, 
    #                    help='Number of nodes for synthetic graph')
    # parser.add_argument('--num-edges', type=int, default=50000, 
    #                    help='Number of edges for synthetic graph')
    parser.add_argument('--num-features', type=int, default=128, 
                       help='Number of input features')
    
    # Distributed options
    parser.add_argument('--distributed', action='store_true', 
                       help='Run in distributed mode (use OpenMPI environment variables)')
    parser.add_argument('--replication', type=int, default=2,
                       help='Replication factor for 1.5D partitioning')
    
    # Algorithm options
    parser.add_argument('--sparse-unaware', action='store_true', 
                       help='Use sparse-unaware implementation')
    
    # Benchmark options
    parser.add_argument('--num-runs', type=int, default=10, 
                       help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=5, 
                       help='Number of warmup runs')
    parser.add_argument('--verify-result', action='store_true',
                       help='Verify distributed result against reference computation')
    
    args = parser.parse_args()
    
    print("Benchmark configuration:")
    print(f"  NPZ file: {args.npz_file}")
    # print(f"  Synthetic graph: {args.num_nodes} nodes, {args.num_edges} edges")
    print(f"  Features: {args.num_features}")
    print(f"  Distributed: {args.distributed}")
    print(f"  Replication: {args.replication}")
    print(f"  Mode: {'Sparse-unaware' if args.sparse_unaware else 'Sparse-aware'}")
    print(f"  Runs: {args.num_runs}, Warmup: {args.warmup}")
    print(f"  Verify result: {args.verify_result}")
    print(f"  Data precision: fp64")
    
    benchmark_broad_func_one5d(args)

if __name__ == '__main__':
    main()
