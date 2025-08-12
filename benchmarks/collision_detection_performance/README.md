# Collision Detection Performance Benchmarks

This directory contains performance benchmarking tools for the Helios collision detection ray tracer optimization project.

## Overview

The benchmarks are designed to:
- Establish performance baselines for the current collision detection implementation
- Compare against simulated master v1.3.44 LiDAR performance characteristics  
- Track performance improvements during optimization phases
- Provide automated regression testing for performance

## Tools

### 1. Main Performance Benchmark (`collision_detection_benchmark`)

Comprehensive benchmark suite testing various scenarios:
- **Simple Spheres**: 1K primitives, 10K rays - basic algorithm validation
- **Dense Cubes**: 5K primitives, 50K rays - memory access patterns
- **Mixed Geometry**: 10K primitives, 100K rays - real-world complexity
- **Complex Plant**: 25K primitives, 250K rays - plant architecture simulation
- **Production Scale**: 100K primitives, 1M rays - large-scale performance
- **Stress Test**: 250K primitives, 2.5M rays - maximum throughput

**Usage:**
```bash
# Run all benchmarks
./collision_detection_benchmark

# Run specific benchmark
./collision_detection_benchmark "Simple Spheres"

# Help and available benchmarks
./collision_detection_benchmark --help
```

**Output:**
- Real-time performance metrics during execution
- CSV file with detailed results in `results/` directory
- Performance summary with rays-per-second and timing breakdowns

### 2. LiDAR Baseline Comparison (`lidar_baseline_comparison`)

Compares collision detection ray tracer against a simulated "old-style" brute-force approach that mimics the performance characteristics expected from master v1.3.44.

**Features:**
- Progressive complexity testing (1K to 10K primitives)
- Synthetic LiDAR scan patterns (spherical ray distribution)
- Direct performance comparison with speedup calculations
- Hit accuracy validation between implementations

**Usage:**
```bash
./lidar_baseline_comparison
```

**Output:**
- Side-by-side performance comparison
- Speedup factor analysis
- Validation that both approaches produce similar hit results

## Integration with Test Framework

The benchmarks are integrated with the existing Helios test framework:

```bash
# From utilities/ directory:

# Run collision detection benchmarks
./run_tests.sh --benchmark --test collisiondetection

# Run all available benchmarks  
./run_tests.sh --benchmark

# Build benchmarks in specific project directory
./run_tests.sh --benchmark --project-dir /path/to/persistent/project
```

## Performance Metrics Collected

- **Throughput**: Rays per second, Million rays per second (MRPS)
- **Timing**: BVH construction time, ray traversal time
- **Memory**: Peak GPU/CPU memory usage (when available)
- **Accuracy**: Hit rates and intersection validation
- **Hardware**: Automatic GPU detection and CPU identification

## Benchmark Results Format

Results are saved as CSV files in `results/` with format:
```
Test Name,Hardware,Build Type,Total Primitives,Total Rays,BVH Construction (ms),Ray Traversal (ms),Rays Per Second,Million Rays Per Second
```

This format is compatible with performance tracking dashboards and regression analysis tools.

## Development Workflow

### Phase 1: Baseline Establishment ✓
- [x] Create benchmark infrastructure
- [x] Establish current performance baselines  
- [x] Compare against master v1.3.44 simulated performance
- [x] Integrate with CI/testing framework

### Phase 2-5: Optimization Phases (Planned)
Each optimization phase will use these benchmarks to:
1. Measure performance before changes
2. Validate improvements after implementation  
3. Detect performance regressions
4. Compare against baseline and previous phases

## Hardware Requirements

- **CPU-only**: All benchmarks work on CPU-only systems with OpenMP
- **GPU acceleration**: Automatically detects and uses CUDA when available
- **Memory**: Benchmarks scale from <100MB to ~2GB depending on scenario

## Troubleshooting

### Build Issues
```bash
# Ensure BUILD_BENCHMARKS is enabled
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON

# Check collision detection plugin is built
make collision_detection_benchmark
```

### Performance Issues
- Ensure Release build for accurate performance measurements
- GPU benchmarks require CUDA toolkit and compatible hardware
- Large stress tests may require sufficient system memory

### Result Validation
- Compare hit rates between baseline and collision detection implementations
- Verify speedup factors are reasonable (>1.0x expected)
- Check CSV output files contain complete data sets

## Future Enhancements

- **Multi-GPU support**: Benchmark scaling across multiple GPUs
- **Memory profiling**: Detailed memory usage and allocation patterns  
- **Automated CI integration**: Performance regression detection in CI/CD
- **Visualization**: Performance trend analysis and graphical reports
- **Plugin integration**: Benchmarks for radiation and aerial LiDAR plugins