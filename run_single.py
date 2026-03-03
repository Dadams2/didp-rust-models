#!/usr/bin/env python3

import argparse
import subprocess
import re
import resource
import sys
from typing import Optional

class SolutionResult:
    def __init__(self):
        self.cost: Optional[float] = None
        self.optimal_cost: Optional[float] = None
        self.best_bound: Optional[float] = None
        self.search_time: Optional[float] = None
        self.expanded: Optional[int] = None
        self.generated: Optional[int] = None
        self.is_optimal: bool = False
        self.is_infeasible: bool = False
        self.is_valid_solution: bool = False
        self.tour: Optional[str] = None
        self.timeout: bool = False
        self.out_of_memory: bool = False

def parse_output(output: str) -> SolutionResult:
    """Parse solver output and extract relevant statistics."""
    result = SolutionResult()
    
    # Check for infeasibility
    if "The problem is infeasible" in output:
        result.is_infeasible = True
    
    if "out of memory" in output.lower() or "cannot allocate memory" in output.lower():
        result.out_of_memory = True
    
    # Parse cost (can be negative or floating point)
    cost_match = re.search(r'^cost:\s*(-?[\d.]+(?:e[+-]?\d+)?)', output, re.MULTILINE)
    if cost_match:
        result.cost = float(cost_match.group(1))
    
    # Parse optimal cost - if this line exists, the solution is optimal
    optimal_match = re.search(r'^optimal cost:\s*(-?[\d.]+(?:e[+-]?\d+)?)', output, re.MULTILINE)
    if optimal_match:
        result.optimal_cost = float(optimal_match.group(1))
        result.is_optimal = True
    
    # Parse best bound
    bound_match = re.search(r'^best bound:\s*(-?[\d.]+(?:e[+-]?\d+)?)', output, re.MULTILINE)
    if bound_match:
        result.best_bound = float(bound_match.group(1))
    
    # Parse search time
    time_match = re.search(r'^Search time:\s*([\d.]+(?:e[+-]?\d+)?)s', output, re.MULTILINE)
    if time_match:
        result.search_time = float(time_match.group(1))
    
    # Parse expanded
    expanded_match = re.search(r'^Expanded:\s*(\d+)', output, re.MULTILINE)
    if expanded_match:
        result.expanded = int(expanded_match.group(1))
    
    # Parse generated
    generated_match = re.search(r'^Generated:\s*(\d+)', output, re.MULTILINE)
    if generated_match:
        result.generated = int(generated_match.group(1))
    
    # Parse tour (optional, for some problem types)
    tour_match = re.search(r'^Tour:\s*(.+)$', output, re.MULTILINE)
    if tour_match:
        result.tour = tour_match.group(1).strip()

    # Parse validity from library output
    if "The solution is valid." in output:
        result.is_valid_solution = True
    
    return result

def set_resource_limits(memory_limit_mb: Optional[int] = None, verbose: bool = False):
    """Set resource limits for the current process."""
    if memory_limit_mb is not None:
        mem_bytes = memory_limit_mb * 1024 * 1024
        
        if verbose:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                print(f"Current RLIMIT_AS: soft={soft}, hard={hard}", file=sys.stderr)
            except (ValueError, OSError) as e:
                print(f"Warning: Cannot get RLIMIT_AS: {e}", file=sys.stderr)
        
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            if verbose:
                print(f"Set memory limit to {memory_limit_mb}MB ({mem_bytes} bytes)", file=sys.stderr)
        except (ValueError, OSError) as e:
            # If RLIMIT_AS fails on macOS, try RLIMIT_DATA instead
            try:
                resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
                if verbose:
                    print(f"Set memory limit (RLIMIT_DATA) to {memory_limit_mb}MB ({mem_bytes} bytes)", file=sys.stderr)
            except (ValueError, OSError) as e2:
                print(f"Warning: Failed to set memory limit: {e}, {e2}", file=sys.stderr)

def run_solver(binary_path: str, input_file: str, solver: str, 
               time_limit: float = 300.0,
               pe_delta: Optional[int] = None,
               sma_max_queue_size: Optional[int] = None,
               verbose: bool = False) -> SolutionResult:
    """Run a solver on an input file and return the parsed result."""
    cmd = [
        binary_path,
        input_file,
        '--solver', solver,
        '--time-limit', str(time_limit)
    ]
    
    # Add solver-specific parameters
    if solver == 'partial-expansion-astar' and pe_delta is not None:
        cmd.extend(['--pe-delta', str(pe_delta)])
    elif solver == 'sma-star' and sma_max_queue_size is not None:
        cmd.extend(['--sma-max-queue-size', str(sma_max_queue_size)])
    
    cmd_str = ' '.join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])
    
    if verbose:
        print(f"\nCommand: {cmd_str}", file=sys.stderr)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=time_limit + 10)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            returncode = process.returncode
            print(f"Timeout after {time_limit}s", file=sys.stderr)
            result = SolutionResult()
            result.timeout = True
            return result
        
        output = stdout + stderr
        
        if verbose:
            print(f"Return code: {returncode}", file=sys.stderr)
            print(f"\nOutput:\n{output}", file=sys.stderr)
        
        parsed_result = parse_output(output)
        
        # Check for memory errors in return code or output
        # Return codes: -9 (SIGKILL), 137 (128+9), 247 (256-9 on some systems)
        # Also check for common OOM strings
        if returncode != 0 and not parsed_result.out_of_memory:
            if (returncode in [-9, 137, 247] or 
                "out of memory" in output.lower() or 
                "cannot allocate memory" in output.lower() or
                "memory allocation" in output.lower() or
                "bad allocation" in output.lower()):
                parsed_result.out_of_memory = True
                if verbose:
                    print(f"Detected OOM: returncode={returncode}", file=sys.stderr)
        
        return parsed_result
    except Exception as e:
        print(f"Error running solver: {e}", file=sys.stderr)
        return SolutionResult()

def format_time(seconds: Optional[float]) -> str:
    """Format time in seconds to a readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 0.001:
        return f"{seconds*1000000:.2f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f}ms"
    else:
        return f"{seconds:.3f}s"

def main():
    parser = argparse.ArgumentParser(
        description='Run a single solver on a single instance with resource limits.'
    )
    parser.add_argument(
        'binary',
        help='Path to the solver binary (e.g., target/release/tsptw_rpid)'
    )
    parser.add_argument(
        'input_file',
        help='Path to the instance filefirefox'
    )
    parser.add_argument(
        '--solver',
        default='cabs',
        choices=['cabs', 'blind-cabs', 'astar', 'dijkstra', 'partial-expansion-astar', 'sma-star'],
        help='Solver to run (default: cabs)'
    )
    parser.add_argument(
        '--time-limit',
        type=float,
        default=300.0,
        help='Time limit in seconds (default: 300.0)'
    )
    parser.add_argument(
        '--memory-limit',
        type=int,
        default=None,
        help='Memory limit in MB (optional)'
    )
    parser.add_argument(
        '--pe-delta',
        type=int,
        default=3000,
        help='Delta parameter for partial-expansion-astar solver (default: 3000)'
    )
    parser.add_argument(
        '--sma-max-queue-size',
        type=int,
        default=None,
        help='Max queue size parameter for sma-star solver (optional)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Set resource limits for the entire process
    set_resource_limits(args.memory_limit, args.verbose)
    
    print(f"Running {args.solver} on {args.input_file}")
    print(f"Time limit: {args.time_limit}s")
    if args.memory_limit:
        print(f"Memory limit: {args.memory_limit}MB")
    print()
    
    # Run the solver
    result = run_solver(
        args.binary,
        args.input_file,
        args.solver,
        args.time_limit,
        args.pe_delta,
        args.sma_max_queue_size,
        args.verbose
    )
    
    # Print results
    print("Results:")
    print("-" * 60)
    
    if result.timeout:
        print("Status: TIMEOUT")
    elif result.out_of_memory:
        print("Status: OUT OF MEMORY")
    elif result.is_infeasible:
        print("Status: INFEASIBLE")
    elif result.is_valid_solution:
        print("Status: VALID SOLUTION")
        if result.is_optimal:
            print("        OPTIMAL ✓")
    else:
        print("Status: FAILED")
    
    if result.cost is not None:
        print(f"Cost: {result.cost}")
    if result.optimal_cost is not None:
        print(f"Optimal Cost: {result.optimal_cost}")
    if result.best_bound is not None:
        print(f"Best Bound: {result.best_bound}")
    if result.search_time is not None:
        print(f"Search Time: {format_time(result.search_time)}")
    if result.expanded is not None:
        print(f"Expanded: {result.expanded}")
    if result.generated is not None:
        print(f"Generated: {result.generated}")
    if result.tour is not None:
        print(f"Tour: {result.tour}")

if __name__ == '__main__':
    main()
