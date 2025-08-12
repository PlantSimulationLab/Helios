/**
 * \file lidar_baseline_comparison.cpp
 * \brief LiDAR performance baseline comparison for collision detection benchmarking
 * 
 * This tool creates a baseline comparison between the current collision detection
 * ray tracer and a simplified LiDAR-style synthetic scan implementation that 
 * mimics the master v1.3.44 performance characteristics.
 */

#include "Context.h"
#include "CollisionDetection.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>

using namespace helios;

/**
 * Simple ray intersection test that mimics the old LiDAR implementation
 * This provides a baseline comparison point
 */
class SimpleLiDARBaseline {
private:
    Context* context;
    
public:
    explicit SimpleLiDARBaseline(Context* ctx) : context(ctx) {}
    
    /**
     * Perform simple brute-force ray intersection (mimics old approach)
     */
    std::vector<float> performSyntheticScan(const std::vector<vec3>& origins,
                                           const std::vector<vec3>& directions,
                                           float max_distance = 1000.0f) {
        std::vector<float> hit_distances;
        hit_distances.reserve(origins.size());
        
        // Get all primitive UUIDs for intersection testing
        std::vector<uint> primitives = context->getAllUUIDs();
        
        for (size_t i = 0; i < origins.size(); i++) {
            float closest_distance = max_distance;
            bool hit = false;
            
            // Test intersection against all primitives (brute force)
            for (uint primitive_id : primitives) {
                vec3 hit_point;
                if (context->rayIntersectsPrimitive(origins[i], directions[i], primitive_id, hit_point)) {
                    float distance = (hit_point - origins[i]).magnitude();
                    if (distance < closest_distance) {
                        closest_distance = distance;
                        hit = true;
                    }
                }
            }
            
            hit_distances.push_back(hit ? closest_distance : -1.0f);
        }
        
        return hit_distances;
    }
};

/**
 * Create a test scene with various geometric primitives
 */
void createBaselineTestScene(Context& context, int complexity_level) {
    context.deleteGeometry();
    
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> pos_dist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> size_dist(0.5f, 3.0f);
    
    int primitive_count = complexity_level * 1000; // Scale with complexity
    
    for (int i = 0; i < primitive_count / 15; i++) { // ~15 primitives per iteration
        vec3 center(pos_dist(gen), pos_dist(gen), pos_dist(gen));
        
        if (i % 3 == 0) {
            // Add sphere
            float radius = size_dist(gen);
            context.addSphere(8, center, radius);
        } else if (i % 3 == 1) {
            // Add box
            vec3 size(size_dist(gen), size_dist(gen), size_dist(gen));
            context.addBox(center, size, SphericalCoord(0, 0));
        } else {
            // Add disk
            float radius = size_dist(gen);
            context.addDisk(center, vec2(radius, radius), SphericalCoord(0, 0));
        }
    }
    
    std::cout << "Created baseline scene with " << context.getPrimitiveCount() << " primitives" << std::endl;
}

/**
 * Generate synthetic scan pattern (mimics LiDAR scanning)
 */
std::pair<std::vector<vec3>, std::vector<vec3>> generateScanPattern(int ray_count) {
    std::vector<vec3> origins, directions;
    origins.reserve(ray_count);
    directions.reserve(ray_count);
    
    vec3 scanner_position(0, 0, 20); // Scanner above the scene
    
    std::random_device rd;
    std::mt19937 gen(123); // Fixed seed
    std::uniform_real_distribution<float> theta_dist(0, 2 * M_PI);
    std::uniform_real_distribution<float> phi_dist(0, M_PI);
    
    for (int i = 0; i < ray_count; i++) {
        // Generate rays in spherical pattern
        float theta = theta_dist(gen);
        float phi = phi_dist(gen) * 0.7f; // Limit to downward hemisphere mostly
        
        vec3 direction(
            sin(phi) * cos(theta),
            sin(phi) * sin(theta),
            -cos(phi) // Mostly downward
        );
        
        origins.push_back(scanner_position);
        directions.push_back(direction);
    }
    
    return {origins, directions};
}

/**
 * Performance comparison between baseline and collision detection
 */
struct ComparisonResults {
    double baseline_time_ms;
    double collision_detection_time_ms;
    double speedup_factor;
    int total_rays;
    int baseline_hits;
    int collision_hits;
    double baseline_mrps;
    double collision_mrps;
};

ComparisonResults runPerformanceComparison(int complexity_level, int ray_count) {
    std::cout << "\n=== Performance Comparison (Level " << complexity_level << ") ===" << std::endl;
    std::cout << "Ray count: " << ray_count << std::endl;
    
    ComparisonResults results{};
    results.total_rays = ray_count;
    
    // Create test scene
    Context context;
    createBaselineTestScene(context, complexity_level);
    
    // Generate scan pattern
    auto [origins, directions] = generateScanPattern(ray_count);
    
    // Test 1: Simple baseline (mimics old LiDAR approach)
    std::cout << "Running baseline synthetic scan..." << std::flush;
    SimpleLiDARBaseline baseline(&context);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto baseline_hits = baseline.performSyntheticScan(origins, directions);
    auto end = std::chrono::high_resolution_clock::now();
    
    results.baseline_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    results.baseline_hits = std::count_if(baseline_hits.begin(), baseline_hits.end(), 
                                         [](float d) { return d > 0; });
    results.baseline_mrps = (ray_count / (results.baseline_time_ms / 1000.0)) / 1000000.0;
    
    std::cout << " (" << results.baseline_time_ms << " ms)" << std::endl;
    
    // Test 2: Collision detection ray tracer
    std::cout << "Running collision detection ray tracer..." << std::flush;
    CollisionDetection collision_detector(&context);
    
    // Convert to collision detection format
    std::vector<CollisionDetection::RayQuery> queries;
    queries.reserve(ray_count);
    
    for (size_t i = 0; i < origins.size(); i++) {
        CollisionDetection::RayQuery query;
        query.origin = origins[i];
        query.direction = directions[i];
        query.max_distance = 1000.0f;
        queries.push_back(query);
    }
    
    start = std::chrono::high_resolution_clock::now();
    auto collision_results = collision_detector.castRays(queries);
    end = std::chrono::high_resolution_clock::now();
    
    results.collision_detection_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    results.collision_hits = std::count_if(collision_results.begin(), collision_results.end(), 
                                          [](const CollisionDetection::HitResult& r) { return r.hit; });
    results.collision_mrps = (ray_count / (results.collision_detection_time_ms / 1000.0)) / 1000000.0;
    
    std::cout << " (" << results.collision_detection_time_ms << " ms)" << std::endl;
    
    // Calculate speedup
    results.speedup_factor = results.baseline_time_ms / results.collision_detection_time_ms;
    
    // Display comparison results
    std::cout << "\n--- Comparison Results ---" << std::endl;
    std::cout << "Baseline (old-style):     " << std::fixed << std::setprecision(2) 
              << results.baseline_time_ms << " ms, " << results.baseline_hits << " hits, "
              << results.baseline_mrps << " MRPS" << std::endl;
    std::cout << "Collision Detection:      " << std::fixed << std::setprecision(2) 
              << results.collision_detection_time_ms << " ms, " << results.collision_hits << " hits, "
              << results.collision_mrps << " MRPS" << std::endl;
    std::cout << "Speedup Factor:           " << std::fixed << std::setprecision(2) 
              << results.speedup_factor << "x" << std::endl;
    
    if (std::abs(results.baseline_hits - results.collision_hits) > ray_count * 0.05) {
        std::cout << "WARNING: Hit count mismatch > 5% - results may not be comparable" << std::endl;
    }
    
    return results;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "LiDAR Baseline Performance Comparison Tool" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        // Run progressively complex scenarios
        std::vector<std::pair<int, int>> test_scenarios = {
            {1, 10000},   // Simple: 1k primitives, 10k rays
            {3, 50000},   // Medium: 3k primitives, 50k rays  
            {5, 100000},  // Complex: 5k primitives, 100k rays
            {10, 250000}  // Production: 10k primitives, 250k rays
        };
        
        std::vector<ComparisonResults> all_results;
        
        for (auto [complexity, rays] : test_scenarios) {
            auto results = runPerformanceComparison(complexity, rays);
            all_results.push_back(results);
        }
        
        // Generate summary report
        std::cout << "\n\n=== OVERALL PERFORMANCE SUMMARY ===" << std::endl;
        std::cout << std::left << std::setw(12) << "Complexity" 
                  << std::setw(10) << "Rays" 
                  << std::setw(15) << "Baseline(ms)" 
                  << std::setw(15) << "Collision(ms)" 
                  << std::setw(12) << "Speedup"
                  << std::setw(12) << "Base MRPS"
                  << std::setw(12) << "Coll MRPS" << std::endl;
        std::cout << std::string(90, '-') << std::endl;
        
        for (size_t i = 0; i < all_results.size(); i++) {
            auto& r = all_results[i];
            auto [complexity, rays] = test_scenarios[i];
            std::cout << std::left << std::setw(12) << ("Level " + std::to_string(complexity))
                      << std::setw(10) << r.total_rays
                      << std::setw(15) << std::fixed << std::setprecision(1) << r.baseline_time_ms
                      << std::setw(15) << std::fixed << std::setprecision(1) << r.collision_detection_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.speedup_factor << "x"
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.baseline_mrps
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.collision_mrps << std::endl;
        }
        
        // Calculate average speedup
        double avg_speedup = 0.0;
        for (const auto& r : all_results) {
            avg_speedup += r.speedup_factor;
        }
        avg_speedup /= all_results.size();
        
        std::cout << std::string(90, '-') << std::endl;
        std::cout << "AVERAGE SPEEDUP: " << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;
        
        if (avg_speedup > 2.0) {
            std::cout << "✓ EXCELLENT: Collision detection shows significant performance improvement!" << std::endl;
        } else if (avg_speedup > 1.2) {
            std::cout << "✓ GOOD: Collision detection shows measurable performance improvement." << std::endl;
        } else if (avg_speedup > 0.9) {
            std::cout << "~ NEUTRAL: Performance is comparable to baseline." << std::endl;
        } else {
            std::cout << "✗ REGRESSION: Performance is worse than baseline - optimization needed." << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}