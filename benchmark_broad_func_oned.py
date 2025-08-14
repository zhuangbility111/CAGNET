#!/usr/bin/env python3
"""
Performance benchmarking script for broad_func_oned function.
This script extracts and tests the broad_func_oned function from the CAGNET project.
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
from scipy.sparse import csr_matrix
import socket

# Import the sparse extension
try:
    from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu
except ImportError:
    print("Warning: sparse_coo_tensor_cpp not available. Please compile the sparse extension first.")
    print("Run: cd sparse-extension && python setup.py build_ext --inplace")
    exit(1)

def sparse_coo_tensor_gpu_fallback(indices, values, size):
    """Fallback implementation if custom sparse extension is not available"""
    return torch.sparse_coo_tensor(indices, values, size, device=indices.device)

# Timing utility function (extracted from original)
def stop_time(gcn_instance, range_name, start, barrier=True):
    """Original stop_time function from gcn_conv.py"""
    barrier=False
    if gcn_instance.get("timers", False) and gcn_instance.get("epoch", 1) > 0:
        torch.cuda.synchronize()
        gcn_instance["timings"][range_name] += time.time() - start
    else:
        return 0.0
    if barrier:
        start = time.time()
        dist.barrier()
        gcn_instance["timings"]["barrier"] += time.time() - start

# Extracted broad_func_oned function - exact copy from original
def broad_func_oned(gcn_instance, graph, ampbyp, inputs):
    """
    Exact copy of broad_func_oned function from gcn_conv.py
    gcn_instance is a dict containing: rank, size, device, group, sparse_unaware, timers, epoch, timings, row_indices_send, row_indices_recv
    """
    
    if gcn_instance["sparse_unaware"]:
        # this is the old function
        n_per_proc = math.ceil(float(graph.size(0) / gcn_instance["size"]))

        z_loc = torch.cuda.DoubleTensor(ampbyp[0].size(0), inputs.size(1), device=gcn_instance["device"]).fill_(0)
        
        inputs_recv = torch.cuda.DoubleTensor(n_per_proc, inputs.size(1), device=gcn_instance["device"]).fill_(0)

        for i in range(gcn_instance["size"]):
            if i == gcn_instance["rank"]:
                inputs_recv = inputs.clone()
            elif i == gcn_instance["size"] - 1:
                inputs_recv = torch.cuda.DoubleTensor(ampbyp[i].size(1), \
                                                            inputs.size(1), \
                                                            device=gcn_instance["device"]).fill_(0)
            start = time.time()
            dist.broadcast(inputs_recv, src=i, group=gcn_instance["group"])
            stop_time(gcn_instance, "bcast", start, barrier=False)
            start = time.time()
            spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(), 
                            ampbyp[i].values(), ampbyp[i].size(0), 
                            ampbyp[i].size(1), inputs_recv, z_loc)

            stop_time(gcn_instance, "spmm_gpu", start, barrier=False)
        return z_loc
    
    start = time.time()
    z_loc = torch.cuda.DoubleTensor(ampbyp[0].size(0), inputs.size(1), device=gcn_instance["device"]).fill_(0)
    
    row_indices_send = gcn_instance["row_indices_send"]
    # row_indices_send[gcn_instance["rank"]] = torch.cuda.LongTensor(0, device=gcn_instance["device"]).fill_(0)
    row_data_send = [torch.cuda.DoubleTensor(device=gcn_instance["device"])]*gcn_instance["size"]
    row_indices_recv = gcn_instance["row_indices_recv"]
    # row_indices_recv[gcn_instance["rank"]] = torch.cuda.LongTensor(0, device=gcn_instance["device"]).fill_(0)
    row_data_recv = [torch.cuda.DoubleTensor(device=gcn_instance["device"]).resize_(row_indices_send[i].size(0), inputs.size(1)).fill_(0) for i in range(gcn_instance["size"])]
    stop_time(gcn_instance, "allocate tensors", start, barrier=False)
    
    start = time.time()
    for i in range(gcn_instance["size"]):
        row_data_send[i] = inputs[row_indices_recv[i].long(), :]
    stop_time(gcn_instance, "gather_row_data", start, barrier=False)

    # print("rank {} send_split: {}".format(gcn_instance["rank"], [row_data_send[i].size(0) for i in range(gcn_instance["size"])]))
    # print("rank {} recv_split: {}".format(gcn_instance["rank"], [row_data_recv[i].size(0) for i in range(gcn_instance["size"])]))

    start = time.time()
    dist.all_to_all(row_data_recv, row_data_send, group=gcn_instance["group"])
    stop_time(gcn_instance, "a2a3", start, barrier=False)

    start = time.time()
    for i in range(gcn_instance["size"]):
       inputs_mul = torch.cuda.DoubleTensor(device=gcn_instance["device"]).resize_(ampbyp[i].size(1), inputs.size(1)).fill_(0)
       inputs_mul[row_indices_send[i]] = row_data_recv[i]
       spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(),
                        ampbyp[i].values(), ampbyp[i].size(0),
                        ampbyp[i].size(1), inputs_mul, z_loc)
    stop_time(gcn_instance, "spmm_gpu", start, barrier=False)
    #del inputs_mul
    #torch.cuda.empty_cache()
    return z_loc

# Data preparation functions from gcn_1d.py (normalize removed)

def split_coo(adj_matrix, partitions, dim):
    """Original split_coo function from gcn_1d.py"""
    # vtx_indices = list(range(0, node_count, n_per_proc))
    # vtx_indices.append(node_count)

    vtx_indices = [0]
    vtx_indices.extend(list(accumulate(partitions)))

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def oned_partition(rank, size, inputs, adj_matrix, data, features, classes, device, partitions=[]):
    """Modified oned_partition function from gcn_1d.py (normalize removed)"""
    node_count = inputs.size(0)

    if not partitions:
        n_per_proc = math.ceil(float(node_count) / size)
        partitions = [math.ceil(float(node_count) / size)]*size
        partitions[size-1] = inputs.size(0) - math.ceil(float(node_count) / size)*(size - 1)    

    am_partitions = None
    am_pbyp = None

    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))
    
    # Swap rows of adj_matrix (swap row indices and column indices)
    # adj_matrix is in format [2, nnz], swap the two rows
    adj_matrix_swapped = torch.stack([adj_matrix[1], adj_matrix[0]], dim=0)
    # print(f"rank: {rank} adj_matrix rows swapped for partition processing", flush=True)

    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions - now using row-swapped adj_matrix
        am_partitions, vtx_indices = split_coo(adj_matrix_swapped, partitions, 1)

        proc_node_count = vtx_indices[rank + 1] - vtx_indices[rank]
        am_pbyp, _ = split_coo(am_partitions[rank], partitions, 0)
        for i in range(len(am_pbyp)):
            if i == size - 1:
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
            # print("rank: {}, am_pbyp[{}].size: {}".format(rank, i, am_pbyp[i].size()), flush=True)

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1), dtype=torch.float64), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            # scale_elements removed

        input_partitions = torch.split(inputs, partitions, dim=0)

        adj_matrix_loc = am_partitions[rank]
        inputs_loc = input_partitions[rank]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs.size: {inputs.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp

def load_npz_file(npz_path):
    """Load NPZ sparse matrix file and convert to COO format with fp64 precision"""
    print(f"Loading sparse matrix from {npz_path}")
    
    # Load NPZ file
    loader = np.load(npz_path)
    data = loader['data'].astype(np.float64)  # Convert to fp64
    rowptr = loader['rowptr'].astype(np.int32)  # Convert to int32
    col = loader['col'].astype(np.int32)  # Convert to int32
    shape = tuple(loader['shape'])
    
    print(f"Loaded CSR matrix: shape={shape}, nnz={len(data)}")
    
    # Convert CSR to COO format
    csr_matrix_obj = csr_matrix((data, col, rowptr), shape=shape)
    coo_matrix = csr_matrix_obj.tocoo()
    
    # Create edge_index tensor (COO format: [row_indices, col_indices])
    edge_index = torch.stack([
        torch.from_numpy(coo_matrix.row.astype(np.int32)), 
        torch.from_numpy(coo_matrix.col.astype(np.int32))
    ], dim=0)
    
    print(f"Graph loaded: {shape[0]} nodes, {coo_matrix.nnz} edges")
    return edge_index, shape[0], coo_matrix.nnz

def setup_distributed():
    """Setup distributed environment using OpenMPI runtime parameters"""
    # Get OpenMPI environment variables
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    
    # Get hostname to identify the node
    hostname = socket.gethostname()
    
    print(f"OpenMPI Info - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    print(f"Process Location - Host: {hostname}, GPU: {local_rank}")
    
    # Set environment variables for PyTorch distributed
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    
    # Set master address and port if not already set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"
    
    # Initialize distributed training
    backend = "nccl"
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    # Print backend information
    if rank == 0:
        print(f"Distributed Backend: {backend}")
        print(f"Backend available: {dist.is_nccl_available()}")
        print(f"Process group initialized with {world_size} processes")
    
    # Bind to local GPU
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {rank} bound to GPU {local_rank} on {hostname}")
    
    return rank, world_size, local_rank

def compute_full_reference_spmm(edge_index, num_nodes, inputs, device):
    """Compute full reference SpMM result for verification"""
    print("Computing full reference SpMM result...")
    
    # Create adjacency matrix from edge_index
    adj_matrix = torch.sparse_coo_tensor(
        edge_index, 
        torch.ones(edge_index.size(1), dtype=torch.float64),
        size=(num_nodes, num_nodes),
        device=device
    ).coalesce()
    
    # Perform SpMM: A @ X
    reference_result = torch.sparse.mm(adj_matrix, inputs.to(device))
    
    print(f"Full reference result shape: {reference_result.shape}")
    print(f"Full reference result norm: {torch.norm(reference_result).item():.6f}")
    
    return reference_result

def extract_local_partition(full_result, rank, size):
    """Extract the local partition from full result based on 1D partitioning"""
    num_nodes = full_result.size(0)
    n_per_proc = math.ceil(float(num_nodes) / size)
    
    start_idx = rank * n_per_proc
    end_idx = min((rank + 1) * n_per_proc, num_nodes)
    
    local_partition = full_result[start_idx:end_idx, :]
    print(f"Rank {dist.get_rank()}: extracted partition [{start_idx}:{end_idx}] from full result")

    return local_partition

def verify_distributed_result(distributed_result, reference_result, rank, tolerance=1e-6):
    """Verify distributed computation result against reference"""
    print(f"Verifying distributed result on rank {rank}...")
    
    # Move to CPU for comparison if needed
    if distributed_result.is_cuda:
        distributed_cpu = distributed_result.cpu()
    else:
        distributed_cpu = distributed_result
        
    if reference_result.is_cuda:
        reference_cpu = reference_result.cpu()
    else:
        reference_cpu = reference_result
    
    # Compute differences
    abs_diff = torch.abs(distributed_cpu - reference_cpu)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    
    # Compute relative differences
    reference_norm = torch.norm(reference_cpu).item()
    distributed_norm = torch.norm(distributed_cpu).item()
    rel_diff = abs(distributed_norm - reference_norm) / (reference_norm + 1e-12)
    
    print(f"Verification results for rank {rank}:")
    print(f"  Distributed result norm: {distributed_norm:.6f}")
    print(f"  Reference result norm: {reference_norm:.6f}")
    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"  Relative norm difference: {rel_diff:.2e}")
    
    # Check if results match within tolerance
    is_correct = max_abs_diff < tolerance and rel_diff < tolerance
    
    if is_correct:
        print(f"✅ Rank {rank}: Results match within tolerance ({tolerance:.0e})")
    else:
        print(f"❌ Rank {rank}: Results do NOT match (tolerance: {tolerance:.0e})")
        if max_abs_diff >= tolerance:
            print(f"   Max abs diff {max_abs_diff:.2e} >= {tolerance:.0e}")
        if rel_diff >= tolerance:
            print(f"   Rel norm diff {rel_diff:.2e} >= {tolerance:.0e}")
    
    return is_correct, max_abs_diff, rel_diff

def benchmark_broad_func_oned(args):
    """Main benchmarking function"""
    
    # Print benchmark header
    mode_str = "SPARSE-UNAWARE" if args.sparse_unaware else "SPARSE-AWARE"
    print(f"\n=== CAGNET 1D Benchmark ({mode_str}) ===")
    
    # Setup distributed environment
    if args.distributed:
        rank, size, local_rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
        group = dist.new_group(list(range(size)))
    else:
        # Single process mode for testing
        rank = 0
        size = 1
        local_rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        group = None
    
    print(f"Process {rank}/{size} on device {device} (local_rank: {local_rank})")
    
    # Load graph data
    if args.npz_file:
        edge_index, num_nodes, num_edges = load_npz_file(args.npz_file)
    # else:
    #     # Generate synthetic data for testing
    #     num_nodes = args.num_nodes
    #     num_edges = args.num_edges
    #     print(f"Generating synthetic graph: {num_nodes} nodes, {num_edges} edges")
        
    #     # Generate random edges
    #     edges = torch.randint(0, num_nodes, (2, num_edges))
    #     edge_index = edges
    
    # Create input features with fp64 precision
    num_features = args.num_features
    torch.manual_seed(1)
    inputs = torch.randn(num_nodes, num_features, dtype=torch.float64)
    # inputs = torch.ones(num_nodes, num_features, dtype=torch.float64)
    
    print(f"Input features shape: {inputs.shape}")
    
    # Partition the data
    print("Partitioning data...")
    # Create dummy data/features/classes for compatibility
    dummy_data = None
    dummy_features = inputs
    dummy_classes = num_features  # Just use num_features as dummy
    
    inputs_loc, adj_matrix_loc, ampbyp = oned_partition(
        rank, size, inputs, edge_index, dummy_data, dummy_features, dummy_classes, device
    )
    print(f"done partitioning", flush=True)
    
    # Move data to device
    inputs_loc = inputs_loc.to(device)
    adj_matrix_loc = adj_matrix_loc.to(device)
    for i in range(len(ampbyp)):
        ampbyp[i] = ampbyp[i].t().coalesce().to(device)
    
    print(f"Local input shape: {inputs_loc.shape}")
    print(f"Local adjacency matrix shape: {adj_matrix_loc.shape}")
    print(f"Number of partitions: {len(ampbyp)}")
    
    if args.distributed:
        # Setup row indices for sparse-aware communication
        row_indices_send = []
        row_indices_recv = []
        
        for i in range(size):
            unique_cols = ampbyp[i]._indices()[1].unique()
            row_indices_send.append(unique_cols)
        
        # Exchange row indices information
        counts_send = [torch.cuda.LongTensor([indices.size(0)], device=device) for indices in row_indices_send]
        counts_recv = [torch.cuda.LongTensor([0], device=device) for _ in range(size)]
        
        dist.all_to_all(counts_recv, counts_send, group=group)
        
        row_indices_recv = [torch.cuda.LongTensor(counts_recv[i].item(), device=device).fill_(0) for i in range(size)]
        dist.all_to_all(row_indices_recv, row_indices_send, group=group)
    else:
        # For single process, create dummy row indices
        row_indices_send = [torch.cuda.LongTensor(range(inputs_loc.size(0)), device=device)] * size
        row_indices_recv = [torch.cuda.LongTensor(range(inputs_loc.size(0)), device=device)] * size
    
    # Create GCN instance dictionary for compatibility with original function signature
    gcn_instance = {
        "rank": rank,
        "size": size,
        "device": device,
        "group": group,
        "sparse_unaware": args.sparse_unaware,
        "timers": True,
        "epoch": 1,
        "timings": defaultdict(float),
        "row_indices_send": row_indices_send,
        "row_indices_recv": row_indices_recv
    }
    
    # Benchmark the function
    print("Starting benchmark...")
    
    # Warmup runs
    for _ in range(args.warmup):
        gcn_instance["epoch"] = 0  # Don't record timing for warmup
        _ = broad_func_oned(gcn_instance, adj_matrix_loc, ampbyp, inputs_loc)
        if args.distributed:
            dist.barrier()
    
    # Clear timings after warmup
    gcn_instance["timings"].clear()
    
    # Benchmark runs
    start_time = time.time()
    
    for epoch in range(1, args.num_runs + 1):
        gcn_instance["epoch"] = epoch
        result = broad_func_oned(gcn_instance, adj_matrix_loc, ampbyp, inputs_loc)
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
    
    # Verify distributed result against reference computation
    if args.verify_result and args.distributed:
        print(f"\n=== Verifying Local Distributed Results (Rank {rank}) ===")
        try:
            # Each rank computes the full reference result using the original edge_index format
            full_reference_result = compute_full_reference_spmm(edge_index, num_nodes, inputs, device)
            
            # Extract the local partition that this rank should compute
            expected_local_result = extract_local_partition(full_reference_result, rank, size)
            
            # Verify the local distributed result (z_loc) against expected local result
            is_correct, max_diff, rel_diff = verify_distributed_result(
                result, expected_local_result, rank
            )
            
            if is_correct:
                print(f"🎉 Rank {rank}: Local distributed computation is CORRECT!")
            else:
                print(f"⚠️  Rank {rank}: Local distributed computation has discrepancies!")
                
        except Exception as e:
            print(f"❌ Rank {rank}: Verification failed with error: {e}")
            import traceback
            traceback.print_exc()
    elif args.verify_result and not args.distributed:
        print("⚠️  Verification only available in distributed mode")
    
    if args.distributed:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Benchmark broad_func_oned function')
    
    # Data options
    parser.add_argument('--npz-file', type=str, help='Path to NPZ sparse matrix file')
    # parser.add_argument('--num-nodes', type=int, default=10000, 
    #                    help='Number of nodes for synthetic graph')
    # parser.add_argument('--num-edges', type=int, default=50000, 
    #                    help='Number of edges for synthetic graph')
    parser.add_argument('--num-features', type=int, default=128, 
                       help='Number of input features')
    # parser.add_argument('--normalize', action='store_true', 
    #                    help='Apply GCN normalization')
    
    # Distributed options
    parser.add_argument('--distributed', action='store_true', 
                       help='Run in distributed mode (use OpenMPI environment variables)')
    
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
    print(f"  Mode: {'Sparse-unaware' if args.sparse_unaware else 'Sparse-aware'}")
    print(f"  Runs: {args.num_runs}, Warmup: {args.warmup}")
    print(f"  Verify result: {args.verify_result}")
    print(f"  Data precision: fp64")
    
    benchmark_broad_func_oned(args)

if __name__ == '__main__':
    main()
