/**
 * \file main.cpp
 * \brief Performance benchmarking suite for Helios collision detection ray tracer
 * 
 * This benchmark suite evaluates the performance of the collision detection plugin
 * ray tracer across various scenarios and compares against baseline implementations.
 */

#include "Context.h"
#include "CollisionDetection.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <memory>

using namespace helios;

/**
 * Performance metrics structure for comprehensive benchmarking
 */
struct PerformanceMetrics {
    double rays_per_second = 0.0;
    double million_rays_per_second = 0.0;
    double peak_memory_mb = 0.0;
    double bvh_construction_ms = 0.0;
    double ray_traversal_ms = 0.0;
    double gpu_utilization_percent = 0.0;
    double cpu_utilization_percent = 0.0;
    size_t total_rays = 0;
    size_t total_primitives = 0;
    std::string test_name;
    std::string hardware_info;
    std::string build_type;
};

/**
 * Benchmark scenario configuration
 */
struct BenchmarkScenario {
    std::string name;
    int primitive_count;
    int ray_count;
    std::string geometry_type;
    float scene_density;
};

/**
 * High-precision timer class for accurate performance measurements
 */
class BenchmarkTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double getElapsedMilliseconds() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
    
    double getElapsedSeconds() const {
        return getElapsedMilliseconds() / 1000.0;
    }
};

/**
 * Comprehensive benchmark suite class
 */
class CollisionDetectionBenchmark {
private:
    Context context;
    std::unique_ptr<CollisionDetection> collision_detector;
    std::vector<BenchmarkScenario> scenarios;
    std::vector<PerformanceMetrics> results;
    std::string output_filename;
    std::string hardware_info;
    std::string build_type;
    
public:
    CollisionDetectionBenchmark() {
        collision_detector = std::make_unique<CollisionDetection>(&context);
        
        // Detect hardware information
        detectHardwareInfo();
        
        // Detect build type
        #ifdef NDEBUG
        build_type = "Release";
        #else
        build_type = "Debug";
        #endif
        
        // Initialize benchmark scenarios
        initializeScenarios();
        
        // Set output filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "collision_detection_benchmark_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".csv";
        output_filename = ss.str();
    }
    
    /**
     * Detect GPU/CPU hardware information
     */
    void detectHardwareInfo() {
        hardware_info = "Unknown Hardware";
        
        #ifdef HELIOS_CUDA_AVAILABLE
        // Try to get GPU information
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error == cudaSuccess && deviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            hardware_info = std::string(prop.name);
        } else {
            hardware_info = "CPU-Only";
        }
        #else
        hardware_info = "CPU-Only";
        #endif
    }
    
    /**
     * Initialize benchmark scenarios from simple to complex
     */
    void initializeScenarios() {
        scenarios = {
            {"Phase2 SoA Test", 1000, 10000, "cubes", 0.3f},
            {"Phase2 Quantized Test", 1000, 10000, "cubes", 0.3f},
            {"Phase3 GPU Test", 1000, 10000, "cubes", 0.3f},
            {"Phase3 GPU Complex", 5000, 50000, "plant", 0.4f}
        };
    }
    
    /**
     * Create geometric primitives for testing
     */
    void createTestGeometry(const BenchmarkScenario& scenario) {
        // Clear any existing geometry (Helios doesn't have deleteGeometry, so we create fresh context)
        // context.deleteGeometry(); // This method doesn't exist
        
        srand(42); // Fixed seed for reproducible results
        
        float scene_size = 100.0f;
        int primitives_created = 0;
        
        if (scenario.geometry_type == "spheres") {
            for (int i = 0; i < scenario.primitive_count; i++) {
                vec3 center = make_vec3(
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size
                );
                float radius = 0.5f + (rand() / float(RAND_MAX)) * 2.0f;
                context.addSphere(10, center, radius);
                primitives_created += 10; // Each sphere has ~10 primitives
                if (primitives_created >= scenario.primitive_count) break;
            }
        } else if (scenario.geometry_type == "cubes") {
            for (int i = 0; i < scenario.primitive_count / 12; i++) { // 12 triangles per box
                vec3 center = make_vec3(
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size
                );
                vec3 size = make_vec3(
                    1.0f + (rand() / float(RAND_MAX)) * 3.0f,
                    1.0f + (rand() / float(RAND_MAX)) * 3.0f,
                    1.0f + (rand() / float(RAND_MAX)) * 3.0f
                );
                context.addBox(center, size, make_int3(1, 1, 1));
                primitives_created += 12;
                if (primitives_created >= scenario.primitive_count) break;
            }
        } else if (scenario.geometry_type == "mixed" || scenario.geometry_type == "mixed_large") {
            // Mix of spheres and boxes
            for (int i = 0; i < scenario.primitive_count / 15; i++) {
                vec3 center = make_vec3(
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size
                );
                
                if (i % 2 == 0) {
                    float radius = 0.5f + (rand() / float(RAND_MAX)) * 2.0f;
                    context.addSphere(10, center, radius);
                    primitives_created += 10;
                } else {
                    vec3 size = make_vec3(
                        1.0f + (rand() / float(RAND_MAX)) * 3.0f,
                        1.0f + (rand() / float(RAND_MAX)) * 3.0f,
                        1.0f + (rand() / float(RAND_MAX)) * 3.0f
                    );
                    context.addBox(center, size, make_int3(1, 1, 1));
                    primitives_created += 12;
                }
                if (primitives_created >= scenario.primitive_count) break;
            }
        } else if (scenario.geometry_type == "plant") {
            // Simulate plant architecture with thin elongated geometry
            for (int i = 0; i < scenario.primitive_count / 20; i++) {
                vec3 base = make_vec3(
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size * 0.5f,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size * 0.5f,
                    0.0f
                );
                
                // Create stem-like structures
                int segments = 5 + rand() % 10;
                for (int j = 0; j < segments; j++) {
                    vec3 center = base + make_vec3(0, 0, j * 2.0f);
                    float radius = 0.1f + (rand() / float(RAND_MAX)) * 0.3f;
                    context.addSphere(8, center, radius);
                    primitives_created += 8;
                    
                    // Add leaves occasionally
                    if (j > 2 && rand() % 3 == 0) {
                        vec3 leaf_center = center + make_vec3(
                            (rand() / float(RAND_MAX) - 0.5f) * 2.0f,
                            (rand() / float(RAND_MAX) - 0.5f) * 2.0f,
                            0.5f
                        );
                        context.addBox(leaf_center, make_vec3(0.8f, 0.4f, 0.1f), make_int3(1, 1, 1));
                        primitives_created += 12;
                    }
                }
                if (primitives_created >= scenario.primitive_count) break;
            }
        } else if (scenario.geometry_type == "stress") {
            // Dense random geometry for stress testing
            scene_size *= 0.7f; // Denser packing
            for (int i = 0; i < scenario.primitive_count / 10; i++) {
                vec3 center = make_vec3(
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size,
                    (rand() / float(RAND_MAX) - 0.5f) * scene_size
                );
                
                int geom_type = rand() % 3;
                if (geom_type == 0) {
                    float radius = 0.2f + (rand() / float(RAND_MAX)) * 1.0f;
                    context.addSphere(8, center, radius);
                    primitives_created += 8;
                } else if (geom_type == 1) {
                    vec3 size = make_vec3(
                        0.5f + (rand() / float(RAND_MAX)) * 1.5f,
                        0.5f + (rand() / float(RAND_MAX)) * 1.5f,
                        0.5f + (rand() / float(RAND_MAX)) * 1.5f
                    );
                    context.addBox(center, size, make_int3(1, 1, 1));
                    primitives_created += 12;
                } else {
                    // Add some disk primitives for variety
                    float radius = 0.3f + (rand() / float(RAND_MAX)) * 1.2f;
                    context.addDiskObject(8, center, make_vec2(radius, radius));
                    primitives_created += 2;
                }
                if (primitives_created >= scenario.primitive_count) break;
            }
        }
        
        std::cout << "Created " << context.getPrimitiveCount() << " primitives for " << scenario.name << std::endl;
    }
    
    /**
     * Generate test rays for benchmarking
     */
    std::vector<CollisionDetection::RayQuery> generateTestRays(int ray_count) {
        std::vector<CollisionDetection::RayQuery> rays;
        rays.reserve(ray_count);
        
        srand(123); // Fixed seed for reproducible ray patterns
        
        float scene_size = 150.0f; // Slightly larger than geometry bounds
        
        for (int i = 0; i < ray_count; i++) {
            // Generate rays from random origins pointing toward scene center with some spread
            vec3 origin = make_vec3(
                (rand() / float(RAND_MAX) - 0.5f) * scene_size * 1.5f,
                (rand() / float(RAND_MAX) - 0.5f) * scene_size * 1.5f,
                (rand() / float(RAND_MAX) - 0.5f) * scene_size * 1.5f
            );
            
            // Point toward scene center with some random spread
            vec3 target = make_vec3(
                (rand() / float(RAND_MAX) - 0.5f) * scene_size * 0.5f,
                (rand() / float(RAND_MAX) - 0.5f) * scene_size * 0.5f,
                (rand() / float(RAND_MAX) - 0.5f) * scene_size * 0.5f
            );
            
            vec3 direction = (target - origin).normalize();
            
            CollisionDetection::RayQuery ray;
            ray.origin = origin;
            ray.direction = direction;
            ray.max_distance = scene_size * 2.0f;
            
            rays.push_back(ray);
        }
        
        return rays;
    }
    
    /**
     * Run a single benchmark scenario
     */
    PerformanceMetrics runBenchmark(const BenchmarkScenario& scenario) {
        std::cout << "\n=== Running " << scenario.name << " ===" << std::endl;
        std::cout << "Target primitives: " << scenario.primitive_count << ", rays: " << scenario.ray_count << std::endl;
        
        PerformanceMetrics metrics;
        metrics.test_name = scenario.name;
        metrics.hardware_info = hardware_info;
        metrics.build_type = build_type;
        
        BenchmarkTimer timer;
        
        // Step 1: Create test geometry
        std::cout << "Creating test geometry..." << std::flush;
        timer.start();
        createTestGeometry(scenario);
        double geometry_creation_time = timer.getElapsedMilliseconds();
        std::cout << " (" << geometry_creation_time << " ms)" << std::endl;
        
        metrics.total_primitives = context.getPrimitiveCount();
        
        // Step 2: BVH construction timing
        std::cout << "Building BVH acceleration structure..." << std::flush;
        timer.start();
        collision_detector->buildBVH();
        metrics.bvh_construction_ms = timer.getElapsedMilliseconds();
        std::cout << " (" << metrics.bvh_construction_ms << " ms)" << std::endl;
        
        // Step 3: Generate test rays
        std::cout << "Generating " << scenario.ray_count << " test rays..." << std::flush;
        timer.start();
        auto rays = generateTestRays(scenario.ray_count);
        double ray_generation_time = timer.getElapsedMilliseconds();
        std::cout << " (" << ray_generation_time << " ms)" << std::endl;
        
        metrics.total_rays = rays.size();
        
        // Step 4: Warm-up run (not timed)
        std::cout << "Warm-up run..." << std::flush;
        auto warmup_results = collision_detector->castRays(std::vector<CollisionDetection::RayQuery>(rays.begin(), rays.begin() + std::min(1000, (int)rays.size())));
        std::cout << " (completed)" << std::endl;
        
        // Step 5: Main benchmark - ray tracing performance
        std::cout << "Ray tracing benchmark..." << std::flush;
        timer.start();
        
        std::vector<CollisionDetection::HitResult> results;
        
        // Phase 2 Optimization Testing
        if (scenario.name == "Phase2 SoA Test") {
            // Test Structure-of-Arrays optimization
            collision_detector->setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
            results = collision_detector->castRaysOptimized(rays);
            std::cout << " [SoA Mode]";
        } else if (scenario.name == "Phase2 Quantized Test") {
            // Test quantized BVH optimization  
            collision_detector->setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);
            results = collision_detector->castRaysOptimized(rays);
            std::cout << " [Quantized Mode]";
        } else if (scenario.name == "Phase3 GPU Test" || scenario.name == "Phase3 GPU Complex") {
            // Test Phase 3 warp-efficient GPU kernels
            CollisionDetection::RayTracingStats gpu_stats;
            results = collision_detector->castRaysGPUPhase3(rays, gpu_stats);
            std::cout << " [Phase3 GPU Mode]";
        } else {
            // Standard legacy benchmark
            collision_detector->setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::LEGACY_AOS);
            results = collision_detector->castRays(rays);
            std::cout << " [Legacy Mode]";
        }
        
        metrics.ray_traversal_ms = timer.getElapsedMilliseconds();
        std::cout << " (" << metrics.ray_traversal_ms << " ms)" << std::endl;
        
        // Step 6: Calculate performance metrics
        double total_time_seconds = metrics.ray_traversal_ms / 1000.0;
        metrics.rays_per_second = rays.size() / total_time_seconds;
        metrics.million_rays_per_second = metrics.rays_per_second / 1000000.0;
        
        // Step 7: Analyze results
        int hits = 0;
        for (const auto& result : results) {
            if (result.hit) hits++;
        }
        double hit_rate = (double)hits / results.size() * 100.0;
        
        // Display results
        std::cout << "\n--- Results ---" << std::endl;
        std::cout << "Total primitives: " << metrics.total_primitives << std::endl;
        std::cout << "Total rays: " << metrics.total_rays << std::endl;
        std::cout << "Hits: " << hits << " (" << std::fixed << std::setprecision(1) << hit_rate << "%)" << std::endl;
        std::cout << "BVH construction: " << std::fixed << std::setprecision(2) << metrics.bvh_construction_ms << " ms" << std::endl;
        std::cout << "Ray tracing time: " << std::fixed << std::setprecision(2) << metrics.ray_traversal_ms << " ms" << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(2) << metrics.million_rays_per_second << " million rays/sec" << std::endl;
        std::cout << "Throughput: " << std::scientific << std::setprecision(2) << metrics.rays_per_second << " rays/sec" << std::endl;
        
        // Phase 2 & 3 Optimization: Display memory usage statistics
        if (scenario.name.find("Phase2") != std::string::npos || scenario.name.find("Phase3") != std::string::npos) {
            auto memory_stats = collision_detector->getBVHMemoryUsage();
            std::cout << "\n--- Phase 2 Memory Optimization ---" << std::endl;
            std::cout << "Legacy memory: " << (memory_stats.legacy_memory_bytes / 1024.0) << " KB" << std::endl;
            std::cout << "SoA memory: " << (memory_stats.soa_memory_bytes / 1024.0) << " KB";
            if (memory_stats.soa_reduction_percent > 0) {
                std::cout << " (" << std::fixed << std::setprecision(1) << memory_stats.soa_reduction_percent << "% reduction)";
            }
            std::cout << std::endl;
            std::cout << "Quantized memory: " << (memory_stats.quantized_memory_bytes / 1024.0) << " KB";
            if (memory_stats.quantized_reduction_percent > 0) {
                std::cout << " (" << std::fixed << std::setprecision(1) << memory_stats.quantized_reduction_percent << "% reduction)";
            }
            std::cout << std::endl;
        }
        
        return metrics;
    }
    
    /**
     * Run all benchmark scenarios
     */
    void runAllBenchmarks() {
        std::cout << "Starting Collision Detection Performance Benchmarks" << std::endl;
        std::cout << "Hardware: " << hardware_info << std::endl;
        std::cout << "Build: " << build_type << std::endl;
        std::cout << "=============================================" << std::endl;
        
        results.clear();
        results.reserve(scenarios.size());
        
        for (const auto& scenario : scenarios) {
            try {
                auto metrics = runBenchmark(scenario);
                results.push_back(metrics);
            } catch (const std::exception& e) {
                std::cerr << "Error in benchmark " << scenario.name << ": " << e.what() << std::endl;
                continue;
            }
        }
        
        // Generate summary report
        generateSummaryReport();
        
        // Save results to standard Helios benchmark format
        saveResultsToFile();
    }
    
    /**
     * Generate and display summary report
     */
    void generateSummaryReport() {
        if (results.empty()) {
            std::cout << "\nNo benchmark results to summarize." << std::endl;
            return;
        }
        
        std::cout << "\n\n=== PERFORMANCE SUMMARY ===" << std::endl;
        std::cout << std::left << std::setw(25) << "Test Name" 
                  << std::setw(12) << "Primitives" 
                  << std::setw(10) << "Rays" 
                  << std::setw(15) << "MRPS" 
                  << std::setw(12) << "BVH (ms)" 
                  << std::setw(12) << "Trace (ms)" << std::endl;
        std::cout << std::string(90, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(25) << result.test_name
                      << std::setw(12) << result.total_primitives
                      << std::setw(10) << result.total_rays
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.million_rays_per_second
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.bvh_construction_ms
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.ray_traversal_ms << std::endl;
        }
        
        // Calculate aggregate statistics
        double total_rays = std::accumulate(results.begin(), results.end(), 0.0, 
            [](double sum, const PerformanceMetrics& m) { return sum + m.total_rays; });
        double total_time = std::accumulate(results.begin(), results.end(), 0.0, 
            [](double sum, const PerformanceMetrics& m) { return sum + m.ray_traversal_ms; }) / 1000.0;
        double avg_mrps = std::accumulate(results.begin(), results.end(), 0.0, 
            [](double sum, const PerformanceMetrics& m) { return sum + m.million_rays_per_second; }) / results.size();
        
        std::cout << std::string(90, '-') << std::endl;
        std::cout << "TOTALS: " << total_rays << " rays in " << std::fixed << std::setprecision(2) 
                  << total_time << " seconds" << std::endl;
        std::cout << "OVERALL PERFORMANCE: " << std::fixed << std::setprecision(2) 
                  << (total_rays / total_time / 1000000.0) << " MRPS (average: " 
                  << avg_mrps << " MRPS)" << std::endl;
    }
    
    /**
     * Save results in Helios benchmark format to ../results/runtime.txt
     */
    void saveResultsToFile() {
        // Create results directory if it doesn't exist
        system("mkdir -p results");
        
        std::ofstream outfile("results/runtime.txt");
        if (!outfile.is_open()) {
            std::cerr << "Failed to create results file: ../results/runtime.txt" << std::endl;
            return;
        }
        
        // Write results in the standard Helios benchmark format: label,runtime
        for (const auto& result : results) {
            outfile << result.test_name << " - BVH Construction," << std::fixed << std::setprecision(3) << result.bvh_construction_ms << "\n";
            outfile << result.test_name << " - Ray Traversal," << std::fixed << std::setprecision(3) << result.ray_traversal_ms << "\n";
            outfile << result.test_name << " - Total Time," << std::fixed << std::setprecision(3) << (result.bvh_construction_ms + result.ray_traversal_ms) << "\n";
        }
        
        outfile.close();
        std::cout << "\nResults saved to: results/runtime.txt" << std::endl;
    }
    
    /**
     * Run specific benchmark by name
     */
    void runSpecificBenchmark(const std::string& benchmark_name) {
        auto it = std::find_if(scenarios.begin(), scenarios.end(),
            [&benchmark_name](const BenchmarkScenario& s) {
                return s.name == benchmark_name;
            });
        
        if (it != scenarios.end()) {
            auto metrics = runBenchmark(*it);
            results.push_back(metrics);
            saveResultsToFile();
        } else {
            std::cout << "Benchmark '" << benchmark_name << "' not found." << std::endl;
            std::cout << "Available benchmarks:" << std::endl;
            for (const auto& scenario : scenarios) {
                std::cout << "  - " << scenario.name << std::endl;
            }
        }
    }
};

/**
 * Display usage information
 */
void showUsage() {
    std::cout << "Collision Detection Performance Benchmark\n" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  collision_detection_benchmark                    # Run all benchmarks" << std::endl;
    std::cout << "  collision_detection_benchmark [benchmark_name]   # Run specific benchmark" << std::endl;
    std::cout << "\nAvailable benchmarks:" << std::endl;
    std::cout << "  - Simple Spheres" << std::endl;
    std::cout << "  - Dense Cubes" << std::endl;
    std::cout << "  - Mixed Geometry" << std::endl;
    std::cout << "  - Complex Plant" << std::endl;
    std::cout << "  - Production Scale" << std::endl;
    std::cout << "  - Stress Test" << std::endl;
}

/**
 * Main benchmark execution - runs all collision detection benchmarks
 */
int main() {
    try {
        CollisionDetectionBenchmark benchmark;
        benchmark.runAllBenchmarks();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}