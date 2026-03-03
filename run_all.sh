#!/bin/bash

# Configuration
TIME_LIMIT=300
MEMORY_LIMIT=8192
NUM_CORES=45

# Problem directories
PROBLEMS=(
    "tsptw"
    "m-pdtsp"
    "optw-Cordeau"
    "optw-Solomon"
    "salbp-1"
    "wt"
    "graph-clear"
    "mosp"
    "talent-scheduling"
)

# Run benchmarks for each problem in parallel
for problem in "${PROBLEMS[@]}"; do
    # Convert problem name to binary name (replace dashes with underscores, strip suffixes like -Cordeau)
    binary_name=$(echo "$problem" | sed 's/-Cordeau//; s/-Solomon//; s/-/_/g')
    binary="target/release/${binary_name}_rpid"
    problem_dir="didp-problems/$problem"
    output="${problem//-/_}_results.csv"
    
    echo "Starting benchmarks for $problem..."
    ./run_benchmarks.py --binary "$binary" --time-limit $TIME_LIMIT --memory-limit $MEMORY_LIMIT --directory "$problem_dir" --num-cores $NUM_CORES --output "$output" 
done

# Wait for all background jobs to complete
# echo "Waiting for all benchmarks to complete..."
# wait
echo "All benchmarks completed!"
