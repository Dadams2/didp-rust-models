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
		result.is_optimal = True  # The presence of "optimal cost:" indicates optimality

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

	return result


def get_limit_resource(memory_limit: Optional[int]):
	"""Create a resource limit function for subprocess preexec_fn."""
	def limit_resources():
		if memory_limit is not None:
			# Set address space limit (in bytes)
			mem_bytes = memory_limit * 1024 * 1024
			try:
				resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
				# Verify the limit was set by reading it back
				soft, hard = resource.getrlimit(resource.RLIMIT_AS)
				print(f"[SUBPROCESS] RLIMIT_AS set: soft={soft} ({soft/(1024*1024):.0f}MB), hard={hard} ({hard/(1024*1024):.0f}MB)", file=sys.stderr)
			except (ValueError, OSError) as e:
				# If RLIMIT_AS fails on macOS, try RLIMIT_DATA instead
				try:
					resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
					soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
					print(f"[SUBPROCESS] RLIMIT_DATA set: soft={soft} ({soft/(1024*1024):.0f}MB), hard={hard} ({hard/(1024*1024):.0f}MB)", file=sys.stderr)
				except (ValueError, OSError) as e2:
					# If all fails, print warning but continue
					# Cannot raise exception here as it would kill the subprocess immediately
					print(f"[SUBPROCESS] Warning: Failed to set memory limit: {e}, {e2}", file=sys.stderr)

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

	# Create resource limit function (only for memory, time limit is handled by the solver)
	limit_fn = get_limit_resource(memory_limit) if memory_limit else None

	try:
		# Use Popen instead of run() because preexec_fn doesn't work with run() on some systems
		process = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			preexec_fn=limit_fn  # Set resource limits
		)

		try:
			stdout, stderr = process.communicate(timeout=time_limit + 10)
			returncode = process.returncode
		except subprocess.TimeoutExpired:
			process.kill()
			stdout, stderr = process.communicate()
			returncode = process.returncode
			error_msg = f"  Timeout for {solver} on {os.path.basename(input_file)}"
			if verbose:
				print(error_msg, file=sys.stderr)
			result = SolutionResult()
			result.timeout = True
			return result, "", cmd_str

		output = stdout + stderr

		if verbose:
			print(f"  Return code: {returncode}", file=sys.stderr)
			print(f"  STDERR: {stderr}", file=sys.stderr)
			print(f"  STDOUT: {stdout}", file=sys.stderr)

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
					print(f"  Detected OOM: returncode={returncode}", file=sys.stderr)

		# If process terminated without finding optimal solution and no other failure criteria,
		# treat it as a timeout (likely hit internal time limit)
		if (not parsed_result.is_optimal and
			not parsed_result.is_infeasible and
			not parsed_result.out_of_memory and
			not parsed_result.timeout):
			parsed_result.timeout = True
			if verbose:
				print(f"  Marking as timeout: no optimal solution found and no other failure criteria", file=sys.stderr)

		return parsed_result, output, cmd_str
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
				'infeasible_count': sum(1 for r in results if r.is_infeasible),
				'timeout_count': timeout_count,
				'oom_count': oom_count
			}
			continue

		times = [r.search_time for r in valid_results if r.search_time is not None]
		expanded = [r.expanded for r in valid_results if r.expanded is not None]
		generated = [r.generated for r in valid_results if r.generated is not None]
		optimal_count = sum(1 for r in valid_results if r.is_optimal)
		solved_count = sum(1 for r in results if r.is_optimal)
		infeasible_count = sum(1 for r in results if r.is_infeasible)

		solver_stats[solver] = {
			'count': len(results),
			'solved_count': solved_count,
			'avg_time': sum(times) / len(times) if times else None,
			'avg_expanded': sum(expanded) / len(expanded) if expanded else None,
			'avg_generated': sum(generated) / len(generated) if generated else None,
			'optimal_count': optimal_count,
			'infeasible_count': infeasible_count,
			'timeout_count': timeout_count,
			'oom_count': oom_count
		}

	# Print table header
	print(f"\n{'Solver':<20} {'Solved':<8} {'Infeasible':<11} {'Timeouts':<10} {'OOM':<6} {'Avg Time':<12} {'Avg Expanded':<14} {'Avg Generated':<14} {'Optimal':<10}")
	print("-" * 122)

	# Print each solver's statistics
	for solver in sorted(solver_stats.keys()):
		stats_data = solver_stats[solver]
		solved_count = stats_data['solved_count']
		infeasible_count = stats_data['infeasible_count']
		timeout_count = stats_data['timeout_count']
		oom_count = stats_data['oom_count']
		avg_time = format_time(stats_data['avg_time'])
		avg_expanded = f"{stats_data['avg_expanded']:.0f}" if stats_data['avg_expanded'] is not None else "N/A"
		avg_generated = f"{stats_data['avg_generated']:.0f}" if stats_data['avg_generated'] is not None else "N/A"
		optimal = f"{stats_data['optimal_count']}/{solved_count}"

		print(f"{solver:<20} {solved_count:<8} {infeasible_count:<11} {timeout_count:<10} {oom_count:<6} {avg_time:<12} {avg_expanded:<14} {avg_generated:<14} {optimal:<10}")

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
		'problem_group': task_info['problem_group'],
		'problem_class': task_info['problem_class'],
		'file': os.path.basename(task_info['input_file']),
		'solver': task_info['solver'],
		'result': result,
		'output': output if task_info['verbose'] else None,
		'command': cmd,
		'task_id': task_info['task_id']
	}

def print_task_result(result: SolutionResult) -> None:
	"""Print the result of a solver task."""
	if result.timeout:
		print("TIMEOUT")
	elif result.out_of_memory:
		print("OUT OF MEMORY")
	elif result.search_time is not None:
		status = "optimal" if result.is_optimal else "solved"
		print(f"{format_time(result.search_time)} ({status})")
	else:
		print("Failed")


def normalize_problem_name(problem: str) -> str:
	"""Normalize a problem name to the binary prefix used in run_all.sh."""
	return problem.replace('-Cordeau', '').replace('-Solomon', '').replace('-', '_')


def build_problem_specs(problems: List[str], base_dir: str, binaries_dir: str) -> List[Dict[str, str]]:
	"""Build problem spec entries with binary path and instance directory."""
	specs: List[Dict[str, str]] = []
	for problem in problems:
		binary_name = f"{normalize_problem_name(problem)}_rpid"
		binary_path = os.path.join(binaries_dir, binary_name)
		problem_dir = os.path.join(base_dir, problem)
		specs.append({
			'problem': problem,
			'binary': binary_path,
			'directory': problem_dir
		})
	return specs


def main():
	parser = argparse.ArgumentParser(
		description='Run solvers on multiple problem sets using a shared process pool.'
	)
	parser.add_argument(
		'--problems',
		nargs='+',
		default=[
			'tsptw',
			'm-pdtsp',
			'optw-Cordeau',
			'optw-Solomon',
			'salbp-1',
			'wt',
			'graph-clear',
			'mosp',
			'talent-scheduling'
		],
		help='Problem set directories under --problems-root (default mirrors run_all.sh)'
	)
	parser.add_argument(
		'--problems-root',
		default='didp-problems',
		help='Root directory containing problem set directories (default: didp-problems)'
	)
	parser.add_argument(
		'--binaries-root',
		default='target/release',
		help='Root directory containing solver binaries (default: target/release)'
	)
	parser.add_argument(
		'--solvers',
		nargs='+',
		default=['cabs', 'blind-cabs', 'astar', 'dijkstra', 'partial-expansion-astar', 'sma-star'],
		choices=['cabs', 'blind-cabs', 'astar', 'dijkstra', 'partial-expansion-astar', 'sma-star'],
		help='Solvers to run (default: cabs blind-cabs astar dijkstra partial-expansion-astar sma-star)'
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
		default=8192,
		help='Memory limit per instance in MB (default: 8192, mirrors run_all.sh)'
	)
	parser.add_argument(
		'--pe-delta',
		type=int,
		default=3000,  # A reasonable enough default
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
		default=5,
		help=f'Number of parallel processes to use (default: 5, mirrors run_all.sh, max available: {cpu_count()})'
	)
	parser.add_argument(
		'--output-dir',
		default='.',
		help='Output directory for per-problem CSV results (default: current directory)'
	)
	parser.add_argument(
		'--verbose',
		action='store_true',
		help='Print detailed output for each run'
	)

	args = parser.parse_args()

	problem_specs = build_problem_specs(args.problems, args.problems_root, args.binaries_root)

	# Validate binaries and directories first
	invalid = False
	for spec in problem_specs:
		if not os.path.exists(spec['binary']):
			print(f"Error: Binary not found at {spec['binary']}", file=sys.stderr)
			invalid = True
		if not os.path.exists(spec['directory']):
			print(f"Error: Directory not found at {spec['directory']}", file=sys.stderr)
			invalid = True
	if invalid:
		print("You may need to build the project first with: cargo build --release", file=sys.stderr)
		sys.exit(1)

	all_problem_classes = {}
	for spec in problem_specs:
		problem_classes = find_input_files(spec['directory'])
		if not problem_classes:
			print(f"Warning: No problem classes found in {spec['directory']}", file=sys.stderr)
			continue
		all_problem_classes[spec['problem']] = problem_classes

	if not all_problem_classes:
		print("No problem classes (subdirectories with files) found in the specified directories.", file=sys.stderr)
		sys.exit(1)

	total_files = sum(len(files) for problem_classes in all_problem_classes.values() for files in problem_classes.values())
	print(f"Found {len(all_problem_classes)} problem sets with {total_files} total instances")
	for problem, problem_classes in all_problem_classes.items():
		count = sum(len(files) for files in problem_classes.values())
		print(f"  {problem}: {count} instances")
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

	# Build list of all tasks across all problems
	tasks = []
	task_id = 0
	for spec in problem_specs:
		problem = spec['problem']
		if problem not in all_problem_classes:
			continue
		for problem_class, input_files in all_problem_classes[problem].items():
			for input_file in input_files:
				for solver in args.solvers:
					tasks.append({
						'task_id': task_id,
						'problem_group': problem,
						'problem_class': problem_class,
						'input_file': input_file,
						'solver': solver,
						'binary_path': spec['binary'],
						'time_limit': args.time_limit,
						'memory_limit': args.memory_limit,
						'pe_delta': args.pe_delta,
						'sma_max_queue_size': args.sma_max_queue_size,
						'verbose': args.verbose
					})
					task_id += 1

	total_runs = len(tasks)
	print(f"Starting {total_runs} solver runs...\n")

	# Run tasks in parallel or sequentially
	detailed_results = []
	if num_cores > 1:
		with Pool(processes=num_cores) as pool:
			completed = 0
			for result_dict in pool.imap_unordered(run_solver_task, tasks):
				completed += 1
				detailed_results.append(result_dict)
				print(f"[{completed}/{total_runs}] {result_dict['problem_group']}/{result_dict['problem_class']}/{result_dict['file']} - {result_dict['solver']}: ", end='', flush=True)
				print_task_result(result_dict['result'])
	else:
		for i, task in enumerate(tasks):
			print(f"[{i+1}/{total_runs}] {task['problem_group']}/{task['problem_class']}/{os.path.basename(task['input_file'])} - {task['solver']}: ", end='', flush=True)
			result_dict = run_solver_task(task)
			detailed_results.append(result_dict)
			print_task_result(result_dict['result'])

	# Organize results by problem set and class and solver
	results_by_problem = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	overall_results = defaultdict(list)

	for result_dict in detailed_results:
		problem = result_dict['problem_group']
		problem_class = result_dict['problem_class']
		solver = result_dict['solver']
		result = result_dict['result']

		results_by_problem[problem][problem_class][solver].append(result)
		overall_results[solver].append(result)

	# Print statistics for each problem set and class
	for problem in sorted(results_by_problem.keys()):
		print(f"\n{'#'*120}")
		print(f"Problem set: {problem}")
		print(f"{'#'*120}")
		for problem_class in sorted(results_by_problem[problem].keys()):
			print_statistics_table(results_by_problem[problem][problem_class], problem_class)

	# Print overall summary statistics
	print_statistics_table(overall_results)

	# Write detailed results to per-problem CSVs
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	results_by_problem_file = defaultdict(list)
	for entry in detailed_results:
		results_by_problem_file[entry['problem_group']].append(entry)

	for problem, entries in results_by_problem_file.items():
		output_path = output_dir / f"{problem.replace('-', '_')}_results.csv"
		with open(output_path, 'w') as f:
			f.write("Problem Class,File,Solver,Cost,Optimal Cost,Best Bound,Search Time,Expanded,Generated,Is Optimal,Is Infeasible,Timeout,Out Of Memory\n")
			for entry in entries:
				r = entry['result']
				f.write(f"{entry['problem_class']},{entry['file']},{entry['solver']},")
				f.write(f"{r.cost if r.cost is not None else ''},")
				f.write(f"{r.optimal_cost if r.optimal_cost is not None else ''},")
				f.write(f"{r.best_bound if r.best_bound is not None else ''},")
				f.write(f"{r.search_time if r.search_time is not None else ''},")
				f.write(f"{r.expanded if r.expanded is not None else ''},")
				f.write(f"{r.generated if r.generated is not None else ''},")
				f.write(f"{r.is_optimal},{r.is_infeasible},{r.timeout},{r.out_of_memory}\n")
		print(f"Detailed results written to: {output_path}")


if __name__ == '__main__':
	main()
