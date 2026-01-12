#!/usr/bin/env python3

import argparse
import subprocess
import re
import os
import resource
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

class SolutionResult:
    def __init__(self):
        self.cost: Optional[float] = None
        self.optimal_cost: Optional[float] = None
        self.best_bound: Optional[float] = None
        self.search_time: Optional[float] = None
        self.expanded: Optional[int] = None
        self.generated: Optional[int] = None
        self.is_optimal: bool = False
        self.is_feasible: bool = True
        self.tour: Optional[str] = None
        self.timeout: bool = False
        self.out_of_memory: bool = False

def parse_output(output: str) -> SolutionResult:
    """Parse solver output and extract relevant statistics."""
    result = SolutionResult()
    
    if "The problem is infeasible" in output or "No solution is found" in output:
        result.is_feasible = False
    
    if "out of memory" in output.lower() or "cannot allocate memory" in output.lower():
        result.out_of_memory = True
    
    # Parse cost
    cost_match = re.search(r'^cost:\s*(-?\d+\.?\d*)', output, re.MULTILINE)
    if cost_match:
        result.cost = float(cost_match.group(1))
    
    # Parse optimal cost
    optimal_match = re.search(r'^optimal cost:\s*(-?\d+\.?\d*)', output, re.MULTILINE)
    if optimal_match:
        result.optimal_cost = float(optimal_match.group(1))
    
    # Parse best bound
    bound_match = re.search(r'^best bound:\s*(-?\d+\.?\d*)', output, re.MULTILINE)
    if bound_match:
        result.best_bound = float(bound_match.group(1))
    
    # Parse search time
    time_match = re.search(r'^Search time:\s*([\d.]+)s', output, re.MULTILINE)
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
    
    # Parse tour
    tour_match = re.search(r'^Tour:\s*(.+)$', output, re.MULTILINE)
    if tour_match:
        result.tour = tour_match.group(1).strip()
    
    # Check if optimal (cost == optimal_cost or cost == best_bound)
    if result.cost is not None:
        if result.optimal_cost is not None and result.cost == result.optimal_cost:
            result.is_optimal = True
        elif result.best_bound is not None and result.cost == result.best_bound:
            result.is_optimal = True
    
    return result

def get_limit_resource(memory_limit: Optional[int]):
    """Create a resource limit function for subprocess preexec_fn."""
    def limit_resources():
        if memory_limit is not None:
            # Set address space limit (in bytes)
            mem_bytes = memory_limit * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError) as e:
                # If RLIMIT_AS fails on macOS, try RLIMIT_DATA instead
                try:
                    resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
                except (ValueError, OSError):
                    # If all fails, just continue without memory limit
                    # TODO should we raise exception? it would kill the subprocess
                    pass
    
    return limit_resources

def run_solver(binary_path: str, input_file: str, solver: str, 
               time_limit: float = 300.0, memory_limit: Optional[int] = None,
               pe_delta: Optional[int] = None,
               sma_max_queue_size: Optional[int] = None,
               verbose: bool = False) -> Tuple[SolutionResult, str, str]:
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
        print(f"\n  Command: {cmd_str}", file=sys.stderr)
        if memory_limit:
            print(f"  Memory limit: {memory_limit}MB ({memory_limit * 1024 * 1024} bytes)", file=sys.stderr)
            # Test if we can actually set this limit
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                print(f"  Current RLIMIT_AS: soft={soft}, hard={hard}", file=sys.stderr)
            except (ValueError, OSError) as e:
                print(f"  Warning: Cannot get RLIMIT_AS: {e}", file=sys.stderr)
    
    # Create resource limit function (only for memory, time limit is handled by the solver)
    limit_fn = get_limit_resource(memory_limit) if memory_limit else None
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 10,  # Add buffer to time limit for subprocess timeout
            preexec_fn=limit_fn  # Set resource limits
        )
        output = result.stdout + result.stderr
        
        if verbose:
            print(f"  Return code: {result.returncode}", file=sys.stderr)
            if result.returncode != 0:
                print(f"  STDERR: {result.stderr}", file=sys.stderr)
                print(f"  STDOUT: {result.stdout}", file=sys.stderr)
        
        parsed_result = parse_output(output)
        
        # Check for memory errors in return code
        if result.returncode != 0 and not parsed_result.out_of_memory:
            if "out of memory" in output.lower() or "cannot allocate memory" in output.lower():
                parsed_result.out_of_memory = True
        
        return parsed_result, output, cmd_str
    except subprocess.TimeoutExpired:
        error_msg = f"  Timeout for {solver} on {os.path.basename(input_file)}"
        if verbose:
            print(error_msg, file=sys.stderr)
        result = SolutionResult()
        result.timeout = True
        return result, "", cmd_str
    except Exception as e:
        error_msg = f"  Error running {solver} on {input_file}: {e}"
        print(error_msg, file=sys.stderr)
        return SolutionResult(), "", cmd_str

def find_input_files(base_directory: str) -> Dict[str, List[str]]:
    """Find all files in subdirectories one level deep.
    
    Returns a dictionary mapping problem class (subdirectory name) to list of files.
    """
    base_path = Path(base_directory)
    if not base_path.exists():
        print(f"Error: Directory {base_directory} does not exist", file=sys.stderr)
        return {}
    
    if not base_path.is_dir():
        print(f"Error: {base_directory} is not a directory", file=sys.stderr)
        return {}
    
    problem_classes = {}
    
    # Iterate through subdirectories (one level deep)
    for subdir in sorted(base_path.iterdir()):
        if subdir.is_dir():
            # Get all files in this subdirectory
            files = [str(f) for f in subdir.iterdir() if f.is_file()]
            if files:
                problem_class_name = subdir.name
                problem_classes[problem_class_name] = sorted(files)
    
    return problem_classes

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

def print_statistics_table(stats: Dict[str, List[SolutionResult]], problem_class: str = ""):
    """Print a formatted table of statistics for each solver."""
    if not stats:
        print("No results to display.")
        return
    
    if problem_class:
        print(f"\n{'='*110}")
        print(f"Statistics for problem class: {problem_class}")
        print(f"{'='*110}")
    else:
        print(f"\n{'='*110}")
        print(f"Overall Statistics")
        print(f"{'='*110}")
    
    # Calculate statistics
    solver_stats = {}
    for solver, results in stats.items():
        # Count failures
        timeout_count = sum(1 for r in results if r.timeout)
        oom_count = sum(1 for r in results if r.out_of_memory)
        
        # Filter out results with no data (only failed runs)
        valid_results = [r for r in results if r.search_time is not None]
        
        if not valid_results:
            solver_stats[solver] = {
                'count': len(results),
                'solved_count': 0,
                'avg_time': None,
                'avg_expanded': None,
                'avg_generated': None,
                'optimal_count': 0,
                'feasible_count': 0,
                'timeout_count': timeout_count,
                'oom_count': oom_count
            }
            continue
        
        times = [r.search_time for r in valid_results if r.search_time is not None]
        expanded = [r.expanded for r in valid_results if r.expanded is not None]
        generated = [r.generated for r in valid_results if r.generated is not None]
        optimal_count = sum(1 for r in valid_results if r.is_optimal)
        feasible_count = sum(1 for r in valid_results if r.is_feasible and r.cost is not None)
        
        solver_stats[solver] = {
            'count': len(results),
            'solved_count': feasible_count,
            'avg_time': sum(times) / len(times) if times else None,
            'avg_expanded': sum(expanded) / len(expanded) if expanded else None,
            'avg_generated': sum(generated) / len(generated) if generated else None,
            'optimal_count': optimal_count,
            'feasible_count': feasible_count,
            'timeout_count': timeout_count,
            'oom_count': oom_count
        }
    
    # Print table header
    print(f"\n{'Solver':<20} {'Solved':<8} {'Timeouts':<10} {'OOM':<6} {'Avg Time':<12} {'Avg Expanded':<14} {'Avg Generated':<14} {'Optimal':<10}")
    print("-" * 110)
    
    # Print each solver's statistics
    for solver in sorted(solver_stats.keys()):
        stats_data = solver_stats[solver]
        solved_count = stats_data['solved_count']
        timeout_count = stats_data['timeout_count']
        oom_count = stats_data['oom_count']
        avg_time = format_time(stats_data['avg_time'])
        avg_expanded = f"{stats_data['avg_expanded']:.0f}" if stats_data['avg_expanded'] is not None else "N/A"
        avg_generated = f"{stats_data['avg_generated']:.0f}" if stats_data['avg_generated'] is not None else "N/A"
        optimal = f"{stats_data['optimal_count']}/{solved_count}"
        
        print(f"{solver:<20} {solved_count:<8} {timeout_count:<10} {oom_count:<6} {avg_time:<12} {avg_expanded:<14} {avg_generated:<14} {optimal:<10}")
    
    print()

def run_solver_task(task_info: Dict) -> Dict:
    """Worker function for running a single solver task in parallel."""
    result, output, cmd = run_solver(
        task_info['binary_path'],
        task_info['input_file'],
        task_info['solver'],
        task_info['time_limit'],
        task_info['memory_limit'],
        task_info['pe_delta'],
        task_info['sma_max_queue_size'],
        task_info['verbose']
    )
    
    return {
        'problem_class': task_info['problem_class'],
        'file': os.path.basename(task_info['input_file']),
        'solver': task_info['solver'],
        'result': result,
        'output': output if task_info['verbose'] else None,
        'command': cmd,
        'task_id': task_info['task_id']
    }

def main():
    parser = argparse.ArgumentParser(
        description='Run solvers on multiple instances and collect statistics.'
    )
    parser.add_argument(
        'directory',
        help='Base directory containing subdirectories (problem classes) with instance files'
    )
    parser.add_argument(
        '--binary',
        required=True,
        help='Path to the solver binary (e.g., target/release/tsptw_rpid)'
    )
    parser.add_argument(
        '--solvers',
        nargs='+',
        default=['cabs', 'blind-cabs', 'astar', 'dijkstra', 'partial-expansion-astar', 'sma-star'],
        choices=['cabs', 'blind-cabs', 'astar', 'dijkstra', 'partial-expansion-astar', 'sma-star'],
        help='Solvers to run (default: cabs blind-cabs astar dijkstra)'
    )
    parser.add_argument(
        '--time-limit',
        type=float,
        default=300.0,
        help='Time limit per instance in seconds (default: 300.0)'
    )
    parser.add_argument(
        '--memory-limit',
        type=int,
        default=None,
        help='Memory limit per instance in MB (optional, uses ulimit)'
    )
    parser.add_argument(
        '--pe-delta',
        type=int,
        default=3000, # A reasonable enough default
        help='Delta parameter for partial-expansion-astar solver (optional)'
    )
    parser.add_argument(
        '--sma-max-queue-size',
        type=int,
        default=None,
        help='Max queue size parameter for sma-star solver (optional)'
    )
    parser.add_argument(
        '--num-cores',
        type=int,
        default=1,
        help=f'Number of parallel processes to use (default: 1, max available: {cpu_count()})'
    )
    parser.add_argument(
        '--output',
        help='Output file for detailed results (optional)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output for each run'
    )
    
    args = parser.parse_args()
    
    # Check if binary exists
    if not os.path.exists(args.binary):
        print(f"Error: Binary not found at {args.binary}", file=sys.stderr)
        print("You may need to build the project first with: cargo build --release", file=sys.stderr)
        sys.exit(1)
    
    # Find all input files organized by problem class
    problem_classes = find_input_files(args.directory)
    
    if not problem_classes:
        print("No problem classes (subdirectories with files) found in the specified directory.", file=sys.stderr)
        sys.exit(1)
    
    total_files = sum(len(files) for files in problem_classes.values())
    print(f"Found {len(problem_classes)} problem classes with {total_files} total instances")
    for problem_class, files in problem_classes.items():
        print(f"  {problem_class}: {len(files)} instances")
    print(f"Testing {len(args.solvers)} solvers: {', '.join(args.solvers)}")
    print(f"Time limit: {args.time_limit}s per instance")
    if args.memory_limit:
        print(f"Memory limit: {args.memory_limit}MB per instance")
    
    # Validate and display number of cores
    num_cores = min(args.num_cores, cpu_count())
    if args.num_cores > cpu_count():
        print(f"Warning: Requested {args.num_cores} cores, but only {cpu_count()} available. Using {num_cores}.", file=sys.stderr)
    print(f"Using {num_cores} parallel process(es)")
    print()
    
    # Build list of all tasks to run
    tasks = []
    task_id = 0
    for problem_class, input_files in problem_classes.items():
        for input_file in input_files:
            for solver in args.solvers:
                tasks.append({
                    'task_id': task_id,
                    'problem_class': problem_class,
                    'input_file': input_file,
                    'solver': solver,
                    'binary_path': args.binary,
                    'time_limit': args.time_limit,
                    'memory_limit': args.memory_limit,
                    'pe_delta': args.pe_delta,
                    'sma_max_queue_size': args.sma_max_queue_size,
                    'verbose': args.verbose
                })
                task_id += 1
    
    total_runs = len(tasks)
    print(f"Starting {total_runs} solver runs...\n")
    
    # Run tasks in parallel
    if num_cores > 1:
        with Pool(processes=num_cores) as pool:
            # Use imap_unordered for progress tracking
            completed = 0
            detailed_results = []
            for result_dict in pool.imap_unordered(run_solver_task, tasks):
                completed += 1
                detailed_results.append(result_dict)
                
                # Print progress
                result = result_dict['result']
                print(f"[{completed}/{total_runs}] {result_dict['problem_class']}/{result_dict['file']} - {result_dict['solver']}: ", end='', flush=True)
                
                if result.timeout:
                    print("TIMEOUT")
                elif result.out_of_memory:
                    print("OUT OF MEMORY")
                elif result.search_time is not None:
                    print(f"{format_time(result.search_time)}", end='')
                    if result.is_optimal:
                        print(" ✓", end='')
                    print()
                else:
                    print("Failed")
    else:
        # Sequential execution (same as before for num_cores=1)
        detailed_results = []
        for i, task in enumerate(tasks):
            print(f"[{i+1}/{total_runs}] {task['problem_class']}/{os.path.basename(task['input_file'])} - {task['solver']}: ", end='', flush=True)
            result_dict = run_solver_task(task)
            detailed_results.append(result_dict)
            
            result = result_dict['result']
            if result.timeout:
                print("TIMEOUT")
            elif result.out_of_memory:
                print("OUT OF MEMORY")
            elif result.search_time is not None:
                print(f"{format_time(result.search_time)}", end='')
                if result.is_optimal:
                    print(" ✓", end='')
                print()
            else:
                print("Failed")
    
    # Organize results by problem class and solver
    results_by_class = defaultdict(lambda: defaultdict(list))
    all_results = defaultdict(list)
    
    for result_dict in detailed_results:
        problem_class = result_dict['problem_class']
        solver = result_dict['solver']
        result = result_dict['result']
        
        results_by_class[problem_class][solver].append(result)
        all_results[solver].append(result)
    
    # Print statistics for each problem class
    for problem_class in sorted(problem_classes.keys()):
        print_statistics_table(results_by_class[problem_class], problem_class)
    
    # Print overall summary statistics
    print_statistics_table(all_results)
    
    # Write detailed results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("Problem Class,File,Solver,Cost,Optimal Cost,Best Bound,Search Time,Expanded,Generated,Is Optimal,Is Feasible,Timeout,Out Of Memory\n")
            for entry in detailed_results:
                r = entry['result']
                f.write(f"{entry['problem_class']},{entry['file']},{entry['solver']},")
                f.write(f"{r.cost if r.cost is not None else ''},")
                f.write(f"{r.optimal_cost if r.optimal_cost is not None else ''},")
                f.write(f"{r.best_bound if r.best_bound is not None else ''},")
                f.write(f"{r.search_time if r.search_time is not None else ''},")
                f.write(f"{r.expanded if r.expanded is not None else ''},")
                f.write(f"{r.generated if r.generated is not None else ''},")
                f.write(f"{r.is_optimal},{r.is_feasible},{r.timeout},{r.out_of_memory}\n")
        print(f"\nDetailed results written to: {args.output}")

if __name__ == '__main__':
    main()
