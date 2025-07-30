// QSM (Quantitative Structure Model) reconstruction implementation
// TreeGraph-inspired approach for improved tree architecture extraction

#include "LiDAR.h"
#include <algorithm>
#include <limits>
#include <fstream>

// TreeGraph-inspired graph-based distance calculation using Dijkstra's algorithm
std::vector<float> LiDARcloud::buildDistanceGraph(const std::vector<helios::vec3>& points, const helios::vec3& base_point) {
    if (points.empty()) return {};
    
    std::vector<float> distances_from_base(points.size(), std::numeric_limits<float>::infinity());
    std::vector<bool> visited(points.size(), false);
    
    // Find base point index (nearest to provided base_point)
    size_t base_idx = 0;
    float min_dist_to_base = (points[0] - base_point).magnitude();
    for (size_t i = 1; i < points.size(); i++) {
        float dist = (points[i] - base_point).magnitude();
        if (dist < min_dist_to_base) {
            min_dist_to_base = dist;
            base_idx = i;
        }
    }
    
    // Initialize base distance
    distances_from_base[base_idx] = 0.0f;
    
    // Build k-NN graph with adaptive neighborhood size based on point density
    const size_t k_neighbors = std::min(size_t(50), points.size() / 10); // Adaptive k
    const float max_neighbor_dist = 0.15f; // Maximum connection distance (15cm)
    
    // Simple priority queue implementation for Dijkstra
    std::vector<std::pair<float, size_t>> queue; // {distance, point_index}
    queue.push_back({0.0f, base_idx});
    
    while (!queue.empty()) {
        // Find minimum distance node in queue (simple linear search)
        size_t min_idx = 0;
        for (size_t i = 1; i < queue.size(); i++) {
            if (queue[i].first < queue[min_idx].first) {
                min_idx = i;
            }
        }
        
        float current_dist = queue[min_idx].first;
        size_t current_idx = queue[min_idx].second;
        queue.erase(queue.begin() + min_idx);
        
        if (visited[current_idx]) continue;
        visited[current_idx] = true;
        
        // Find k nearest neighbors
        std::vector<std::pair<float, size_t>> neighbors;
        for (size_t i = 0; i < points.size(); i++) {
            if (i == current_idx || visited[i]) continue;
            
            float dist = (points[current_idx] - points[i]).magnitude();
            if (dist <= max_neighbor_dist) {
                neighbors.push_back({dist, i});
            }
        }
        
        // Keep only k nearest neighbors
        if (neighbors.size() > k_neighbors) {
            std::partial_sort(neighbors.begin(), neighbors.begin() + k_neighbors, neighbors.end());
            neighbors.resize(k_neighbors);
        }
        
        // Update distances to neighbors
        for (const auto& neighbor : neighbors) {
            float edge_dist = neighbor.first;
            size_t neighbor_idx = neighbor.second;
            
            float new_dist = current_dist + edge_dist;
            if (new_dist < distances_from_base[neighbor_idx]) {
                distances_from_base[neighbor_idx] = new_dist;
                queue.push_back({new_dist, neighbor_idx});
            }
        }
    }
    
    // Handle any unreachable points by using Euclidean distance as fallback
    for (size_t i = 0; i < points.size(); i++) {
        if (distances_from_base[i] == std::numeric_limits<float>::infinity()) {
            distances_from_base[i] = (points[i] - base_point).magnitude();
        }
    }
    
    return distances_from_base;
}

std::vector<uint> LiDARcloud::reconstructQSM(helios::Context *context, const std::string &filename,
                                             uint radial_subdivisions) {
    std::cout << "DEBUG: QSM reconstruction called with " << hits.size() << " hit points" << std::endl;
    
    if (printmessages) {
        std::cout << "Reconstructing QSM from LiDAR point cloud..." << std::flush;
    }

    // Validate input
    if (hits.empty()) {
        std::cout << "failed." << std::endl;
        helios::helios_runtime_error(
            "ERROR (LiDARcloud::reconstructQSM): No hit points loaded. Call addHitPoint() or synthetic scan first.");
    }

    // Data structures for QSM reconstruction
    struct SkeletonNode {
        helios::vec3 position;
        float radius;
        std::vector<uint> neighbors;
        uint parent;
        float distance_from_base;
        uint node_id;
        SkeletonNode() : position(0, 0, 0), radius(0.01f), parent(UINT_MAX), distance_from_base(0), node_id(UINT_MAX) {}
    };
    
    struct CylinderModel {
        helios::vec3 start, end;
        helios::vec3 axis;
        float radius;
        float length;
        uint branch_id;
        uint parent_cylinder;
        uint branch_order;
        uint position_in_branch;
        CylinderModel() : start(0, 0, 0), end(0, 0, 0), axis(0, 0, 1), radius(0.01f), length(0),
                          branch_id(-1), parent_cylinder(-1), branch_order(1), position_in_branch(0) {}
    };

    std::vector<SkeletonNode> skeleton_nodes;
    std::vector<CylinderModel> cylinders;

    // Phase 1: Build distance graph and extract skeleton
    if (printmessages) {
        std::cout << "\n  Building distance graph..." << std::flush;
    }

    // Create point cloud from hits
    std::vector<helios::vec3> points;
    std::vector<float> distances_from_base;
    points.reserve(hits.size());
    distances_from_base.reserve(hits.size());

    // Find base point (lowest z-coordinate)
    helios::vec3 base_point = hits[0].position;
    for (const auto &hit: hits) {
        if (hit.position.z < base_point.z) {
            base_point = hit.position;
        }
        points.push_back(hit.position);
    }

    // Calculate distance from base using graph-based shortest paths (TreeGraph approach)
    distances_from_base = buildDistanceGraph(points, base_point);

    if (printmessages) {
        std::cout << "done.\n  Extracting skeleton..." << std::flush;
    }

    // Phase 2: 3D spatial clustering with DBSCAN to detect branching (TreeGraph approach)
    float max_distance = *std::max_element(distances_from_base.begin(), distances_from_base.end());
    uint num_bins = std::max(20, (int) (max_distance / 0.05f)); // 5cm bins  
    float bin_size = max_distance / num_bins;

    if (printmessages) {
        std::cout << "\n    Debug: " << points.size() << " points, max_distance=" << max_distance 
                  << ", num_bins=" << num_bins << ", bin_size=" << bin_size << std::flush;
    }

    // DBSCAN clustering parameters  
    const float eps = 0.08f;  // 8cm neighborhood radius for clustering
    const uint min_pts = 8;   // Minimum points to form a cluster
    
    uint bins_with_points = 0;
    uint total_points_processed = 0;
    uint total_clusters_found = 0;

    for (uint bin = 0; bin < num_bins; bin++) {
        float bin_start = bin * bin_size;
        float bin_end = (bin + 1) * bin_size;

        // Get points in this distance range
        std::vector<helios::vec3> bin_points;
        std::vector<size_t> bin_indices;
        for (size_t i = 0; i < points.size(); i++) {
            if (distances_from_base[i] >= bin_start && distances_from_base[i] < bin_end) {
                bin_points.push_back(points[i]);
                bin_indices.push_back(i);
            }
        }

        if (bin_points.size() < min_pts) {
            if (printmessages && bin_points.size() > 0) {
                std::cout << "\n    Bin " << bin << " has only " << bin_points.size() << " points (< " << min_pts << ")" << std::flush;
            }
            continue;
        }

        bins_with_points++;
        total_points_processed += bin_points.size();

        // DBSCAN clustering to detect multiple branches in this distance range
        std::vector<int> cluster_labels(bin_points.size(), -1); // -1 = noise
        int cluster_id = 0;
        
        for (size_t i = 0; i < bin_points.size(); i++) {
            if (cluster_labels[i] != -1) continue; // Already processed
            
            // Find neighbors within eps distance
            std::vector<size_t> neighbors;
            for (size_t j = 0; j < bin_points.size(); j++) {
                if (i != j && (bin_points[i] - bin_points[j]).magnitude() <= eps) {
                    neighbors.push_back(j);
                }
            }
            
            if (neighbors.size() < min_pts - 1) continue; // Not enough neighbors (core point needs min_pts total)
            
            // Create new cluster
            cluster_labels[i] = cluster_id;
            std::vector<size_t> seed_set = neighbors;
            
            // Expand cluster
            for (size_t k = 0; k < seed_set.size(); k++) {
                size_t q = seed_set[k];
                
                if (cluster_labels[q] == -1) { // Was noise, now border point
                    cluster_labels[q] = cluster_id;
                }
                if (cluster_labels[q] != -1) continue; // Already assigned
                
                cluster_labels[q] = cluster_id;
                
                // Find neighbors of q
                std::vector<size_t> q_neighbors;
                for (size_t j = 0; j < bin_points.size(); j++) {
                    if (q != j && (bin_points[q] - bin_points[j]).magnitude() <= eps) {
                        q_neighbors.push_back(j);
                    }
                }
                
                // If q is core point, add its neighbors to seed set
                if (q_neighbors.size() >= min_pts - 1) {
                    for (size_t neighbor : q_neighbors) {
                        if (std::find(seed_set.begin(), seed_set.end(), neighbor) == seed_set.end()) {
                            seed_set.push_back(neighbor);
                        }
                    }
                }
            }
            cluster_id++;
        }
        
        // Create skeleton nodes from clusters
        for (int c = 0; c < cluster_id; c++) {
            std::vector<helios::vec3> cluster_points;
            for (size_t i = 0; i < bin_points.size(); i++) {
                if (cluster_labels[i] == c) {
                    cluster_points.push_back(bin_points[i]);
                }
            }
            
            if (cluster_points.size() < min_pts / 2) continue; // Skip very small clusters
            
            // Calculate cluster centroid
            helios::vec3 centroid(0, 0, 0);
            for (const auto &point: cluster_points) {
                centroid = centroid + point;
            }
            centroid = centroid / cluster_points.size();
            
            // Improved radius estimation with branch tapering
            float radius_sum = 0;
            for (const auto &point: cluster_points) {
                radius_sum += (point - centroid).magnitude();
            }
            float spread_radius = radius_sum / cluster_points.size();
            
            // Apply tapering based on distance from base (further = smaller radius)
            float taper_factor = std::max(0.2f, 1.0f - (bin_start + bin_end) * 0.5f / max_distance);
            
            // Scale by cluster size (smaller clusters = smaller radius)
            float density_factor = std::min(1.0f, (float)cluster_points.size() / 20.0f);
            
            float radius = std::max(0.005f, spread_radius * taper_factor * density_factor);
            
            SkeletonNode node;
            node.position = centroid;
            node.radius = radius;
            node.distance_from_base = (bin_start + bin_end) * 0.5f;
            node.node_id = skeleton_nodes.size();
            
            skeleton_nodes.push_back(node);
            total_clusters_found++;
        }

        if (printmessages && (bin < 5 || cluster_id > 1)) {  // Debug first few bins or bins with multiple clusters
            std::cout << "\n    Bin " << bin << " (" << bin_start << "-" << bin_end 
                      << "): " << bin_points.size() << " points → " << cluster_id << " clusters" << std::flush;
        }
    }

    if (printmessages) {
        std::cout << "\n    Debug: " << bins_with_points << " bins with >=" << min_pts << " points, " 
                  << total_points_processed << " total points processed, " 
                  << total_clusters_found << " clusters found" << std::flush;
    }

    if (printmessages) {
        std::cout << "extracted " << skeleton_nodes.size() << " skeleton nodes.\n  Building connectivity..." <<
                std::flush;
    }

    // Phase 3: Graph-based skeleton connectivity (TreeGraph approach)
    if (skeleton_nodes.empty()) {
        if (printmessages) {
            std::cout << "failed - no skeleton nodes extracted." << std::endl;
        }
        return {};
    }

    // Build connectivity based on 3D proximity and graph-based distance ordering
    const float max_connection_distance = 0.20f; // 20cm maximum connection distance
    
    // Sort nodes by distance from base to establish hierarchy
    std::vector<std::pair<float, size_t>> sorted_nodes;
    for (size_t i = 0; i < skeleton_nodes.size(); i++) {
        sorted_nodes.push_back({skeleton_nodes[i].distance_from_base, i});
    }
    std::sort(sorted_nodes.begin(), sorted_nodes.end());
    
    // Connect each node to its nearest valid parent (closer to base)
    for (size_t i = 1; i < sorted_nodes.size(); i++) {
        size_t current_idx = sorted_nodes[i].second;
        float current_dist = sorted_nodes[i].first;
        
        float min_parent_distance = std::numeric_limits<float>::infinity();
        size_t best_parent = 0;
        
        // Find closest parent node (must be closer to base and within connection distance)
        for (size_t j = 0; j < i; j++) {
            size_t potential_parent_idx = sorted_nodes[j].second;
            float parent_dist = sorted_nodes[j].first;
            
            // Parent must be closer to base
            if (parent_dist >= current_dist) continue;
            
            // Check 3D distance
            float spatial_dist = (skeleton_nodes[current_idx].position - 
                                skeleton_nodes[potential_parent_idx].position).magnitude();
            
            if (spatial_dist <= max_connection_distance && spatial_dist < min_parent_distance) {
                min_parent_distance = spatial_dist;
                best_parent = potential_parent_idx;
            }
        }
        
        // Connect to best parent
        skeleton_nodes[current_idx].parent = best_parent;
        skeleton_nodes[best_parent].neighbors.push_back(current_idx);
        
        if (printmessages && i < 5) {  // Debug first few connections
            std::cout << "\n      Node " << current_idx << " -> parent " << best_parent 
                      << " (dist=" << min_parent_distance << "m)" << std::flush;
        }
    }

    if (printmessages) {
        std::cout << "done.\n  Phase 3: Reconstructing shoot topology..." << std::flush;
    }

    // Phase 4: Branch topology detection based on connectivity
    struct Shoot {
        std::vector<uint> node_path;
        uint shoot_id;
        uint parent_shoot;
        uint branch_order;
        Shoot() : shoot_id(0), parent_shoot(UINT_MAX), branch_order(1) {}
    };

    std::vector<Shoot> shoots;
    std::vector<bool> node_visited(skeleton_nodes.size(), false);
    
    // Find branch points (nodes with multiple children) and analyze connectivity
    std::vector<uint> branch_points;
    uint nodes_with_0_children = 0, nodes_with_1_child = 0, nodes_with_multiple_children = 0;
    
    for (size_t i = 0; i < skeleton_nodes.size(); i++) {
        uint num_children = skeleton_nodes[i].neighbors.size();
        if (num_children == 0) nodes_with_0_children++;
        else if (num_children == 1) nodes_with_1_child++;
        else {
            nodes_with_multiple_children++;
            branch_points.push_back(i);
        }
    }
    
    if (printmessages) {
        std::cout << "\n    Connectivity analysis: " << nodes_with_0_children << " leaves, " 
                  << nodes_with_1_child << " linear nodes, " << nodes_with_multiple_children 
                  << " branch points" << std::flush;
    }
    
    // Create shoots by following paths from base to branch points and leaves
    uint shoot_id = 0;
    
    // Start from base (node with distance_from_base closest to 0)
    size_t base_node = 0;
    float min_base_dist = skeleton_nodes[0].distance_from_base;
    for (size_t i = 1; i < skeleton_nodes.size(); i++) {
        if (skeleton_nodes[i].distance_from_base < min_base_dist) {
            min_base_dist = skeleton_nodes[i].distance_from_base;
            base_node = i;
        }
    }
    
    // Recursive function to trace shoots
    std::function<void(uint, uint, uint, std::vector<uint>&)> trace_shoot = 
        [&](uint start_node, uint parent_shoot_id, uint branch_order, std::vector<uint>& current_path) {
        
        current_path.push_back(start_node);
        node_visited[start_node] = true;
        
        // If this is a leaf node or all children are visited, complete the shoot
        bool has_unvisited_children = false;
        for (uint child : skeleton_nodes[start_node].neighbors) {
            if (!node_visited[child]) {
                has_unvisited_children = true;
                break;
            }
        }
        
        if (!has_unvisited_children || skeleton_nodes[start_node].neighbors.empty()) {
            // Complete current shoot
            if (current_path.size() >= 2) {  // Need at least 2 nodes for a shoot
                Shoot shoot;
                shoot.shoot_id = shoot_id++;
                shoot.parent_shoot = parent_shoot_id;
                shoot.branch_order = branch_order;
                shoot.node_path = current_path;
                shoots.push_back(shoot);
            }
            return;
        }
        
        // If multiple children, create branches
        if (skeleton_nodes[start_node].neighbors.size() > 1) {
            // Complete current shoot at branch point
            if (current_path.size() >= 2) {
                Shoot shoot;
                shoot.shoot_id = shoot_id++;
                shoot.parent_shoot = parent_shoot_id;
                shoot.branch_order = branch_order;
                shoot.node_path = current_path;
                shoots.push_back(shoot);
                parent_shoot_id = shoot.shoot_id;
            }
            
            // Create child shoots
            for (uint child : skeleton_nodes[start_node].neighbors) {
                if (!node_visited[child]) {
                    std::vector<uint> child_path = {start_node}; // Start child path with branch point
                    trace_shoot(child, parent_shoot_id, branch_order + 1, child_path);
                }
            }
        } else {
            // Continue current shoot
            for (uint child : skeleton_nodes[start_node].neighbors) {
                if (!node_visited[child]) {
                    trace_shoot(child, parent_shoot_id, branch_order, current_path);
                    break;
                }
            }
        }
    };
    
    // Start tracing from base node
    std::vector<uint> base_path;
    trace_shoot(base_node, UINT_MAX, 1, base_path);

    if (printmessages) {
        std::cout << "\n    Found " << shoots.size() << " shoot paths..." << std::flush;
    }

    // Phase 5: Generate cylinders
    for (const auto& shoot : shoots) {
        for (size_t i = 0; i < shoot.node_path.size() - 1; i++) {
            uint node_idx = shoot.node_path[i];
            uint next_node_idx = shoot.node_path[i + 1];
            
            CylinderModel cylinder;
            cylinder.start = skeleton_nodes[node_idx].position;
            cylinder.end = skeleton_nodes[next_node_idx].position;
            cylinder.axis = (cylinder.end - cylinder.start).normalize();
            cylinder.length = (cylinder.end - cylinder.start).magnitude();
            cylinder.radius = (skeleton_nodes[node_idx].radius + skeleton_nodes[next_node_idx].radius) * 0.5f;
            cylinder.branch_id = shoot.shoot_id + 1;
            cylinder.branch_order = shoot.branch_order;
            cylinder.position_in_branch = i;
            
            cylinders.push_back(cylinder);
        }
    }

    if (printmessages) {
        std::cout << "generated " << cylinders.size() << " cylinders." << std::endl;
    }

    // Write QSM file
    if (printmessages) {
        std::cout << "  Writing QSM file..." << std::flush;
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        helios::helios_runtime_error("ERROR (LiDARcloud::reconstructQSM): Could not open output file: " + filename);
    }

    // Write header (tab-separated format expected by loadTreeQSM)
    outfile << "radius (m)\tlength (m)\tstart_point\taxis_direction\tparent\textension\tbranch\tbranch_order\tposition_in_branch\tmad\tSurfCov\tadded\tUnmodRadius (m)" << std::endl;

    // Write cylinder data with tab separators
    for (size_t i = 0; i < cylinders.size(); i++) {
        const auto& cyl = cylinders[i];
        outfile << cyl.radius << "\t" 
                << cyl.length << "\t"
                << cyl.start.x << "\t" << cyl.start.y << "\t" << cyl.start.z << "\t"
                << cyl.axis.x << "\t" << cyl.axis.y << "\t" << cyl.axis.z << "\t"
                << (i > 0 ? i-1 : 0) << "\t"  // parent cylinder
                << 0 << "\t"  // extension
                << cyl.branch_id << "\t"
                << cyl.branch_order << "\t"
                << i << "\t"  // position_in_branch
                << 0.0002 << "\t"  // mad (using reference value)
                << 1 << "\t"  // surf_cov
                << 0 << "\t"    // added
                << cyl.radius << std::endl;  // unmod_radius
    }

    outfile.close();

    if (printmessages) {
        std::cout << "done.\n  Creating visualization..." << std::flush;
    }

    // Create tube objects for visualization
    std::vector<uint> tube_object_IDs;
    tube_object_IDs.reserve(cylinders.size());

    for (const auto& cylinder : cylinders) {
        if (cylinder.length > 1e-6) {
            // Create tube using vector of nodes and radii
            std::vector<helios::vec3> nodes = {cylinder.start, cylinder.end};
            std::vector<float> radii = {cylinder.radius, cylinder.radius};
            
            uint objID = context->addTubeObject(radial_subdivisions, nodes, radii);
            
            // Set brown color for branches
            context->setObjectColor(objID, helios::make_RGBcolor(0.55f, 0.27f, 0.07f));
            tube_object_IDs.push_back(objID);
        }
    }

    if (printmessages) {
        std::cout << "done." << std::endl;
        std::cout << "QSM reconstruction complete: " << cylinders.size() << " cylinders, " 
                  << tube_object_IDs.size() << " tube objects created." << std::endl;
    }

    return tube_object_IDs;
}

std::vector<uint> LiDARcloud::leafReconstructionTreeQSM(helios::Context *context, const std::vector<uint> &tube_object_IDs, const helios::vec2 &leaf_size, float proximity_distance, const char *mask_file) {

    if (printmessages) {
        cout << "Performing TreeQSM-based leaf reconstruction..." << flush;
    }

    if (triangles.size() == 0) {
        std::cout << "failed." << std::endl;
        helios_runtime_error("ERROR (LiDARcloud::leafReconstructionTreeQSM): There are no triangulated points. Either the triangulation failed or 'triangulateHitPoints()' was not called.");
    }

    if (tube_object_IDs.empty()) {
        std::cout << "failed." << std::endl;
        helios_runtime_error("ERROR (LiDARcloud::leafReconstructionTreeQSM): No tube object IDs provided. Call 'loadTreeQSM()' first to generate tube objects.");
    }

    // Validate mask file
    std::string file = mask_file;
    if (file.substr(file.find_last_of(".") + 1) != "png") {
        std::cout << "failed." << std::endl;
        helios_runtime_error("ERROR (LiDARcloud::leafReconstructionTreeQSM): Mask data file " + std::string(mask_file) + " must be PNG image format.");
    }

    reconstructed_alphamasks_maskfile = mask_file;

    // Clear previous reconstruction data
    reconstructed_alphamasks_center.clear();
    reconstructed_alphamasks_size.clear();
    reconstructed_alphamasks_rotation.clear();
    reconstructed_alphamasks_gridcell.clear();
    reconstructed_alphamasks_direct_flag.clear();

    uint Ncells = getGridCellCount();

    // Create a prototype leaf tile object with the alpha mask
    uint prototype_objID = context->addTileObject(helios::make_vec3(0, 0, 0), leaf_size, helios::make_SphericalCoord(0, 0), helios::make_int2(1, 1), mask_file);

    // Get the actual solid area of the prototype leaf
    float prototype_leaf_area = context->getObjectArea(prototype_objID);

    // Calculate leaf angle distributions for all voxels
    const uint Nbins = 36; // 10-degree bins for angle distribution
    std::vector<std::vector<float>> CDF_theta, CDF_phi;
    calculateLeafAngleCDF(Nbins, CDF_theta, CDF_phi);

    // Convert to proper angle ranges
    const float dtheta = 0.5f * M_PI / float(Nbins); // Zenith angles: 0 to π/2
    const float dphi = 2.0f * M_PI / float(Nbins); // Azimuth angles: 0 to 2π

    if (printmessages) {
        cout << "Calculated CDF for " << CDF_theta.size() << " voxels, " << Nbins << " bins each." << endl;

        // Debug: Check how many voxels have valid CDF data
        int valid_theta_count = 0, valid_phi_count = 0;
        for (uint v = 0; v < CDF_theta.size() && v < 10; v++) {
            bool has_theta = (!CDF_theta[v].empty() && CDF_theta[v].back() > 0.001f);
            bool has_phi = (!CDF_phi[v].empty() && CDF_phi[v].back() > 0.001f);
            if (has_theta)
                valid_theta_count++;
            if (has_phi)
                valid_phi_count++;

            if (v < 5) { // Debug first few voxels
                cout << "Voxel " << v << ": theta_CDF_max=" << (CDF_theta[v].empty() ? 0.0f : CDF_theta[v].back()) << ", phi_CDF_max=" << (CDF_phi[v].empty() ? 0.0f : CDF_phi[v].back()) << endl;
            }
        }
        cout << "Valid theta CDFs: " << valid_theta_count << "/" << std::min(10u, (uint) CDF_theta.size()) << ", Valid phi CDFs: " << valid_phi_count << "/" << std::min(10u, (uint) CDF_phi.size()) << endl;
    }

    uint total_leaves_placed = 0;
    std::vector<uint> leaf_object_IDs; // Collect all created leaf object IDs

    // Process each voxel
    int voxels_with_leaf_area = 0;
    int voxels_with_nearby_branches = 0;
    for (uint v = 0; v < Ncells; v++) {
        float voxel_leaf_area = getCellLeafArea(v);
        if (voxel_leaf_area <= 0.0f) {
            continue; // Skip voxels with no leaf area
        }
        voxels_with_leaf_area++;

        if (printmessages && voxels_with_leaf_area <= 5) {
            cout << "Voxel " << v << " has leaf area: " << voxel_leaf_area << endl;
        }

        helios::vec3 voxel_center = getCellCenter(v);
        helios::vec3 voxel_size = getCellSize(v);

        // Find nearby branches within proximity_distance
        std::vector<uint> nearby_branches;
        std::vector<float> branch_distances;

        for (uint tube_id: tube_object_IDs) {
            // Get tube object geometry from context
            std::vector<helios::vec3> tube_nodes = context->getTubeObjectNodes(tube_id);

            // Find minimum distance from voxel center to this tube
            float min_distance = 1e6f;
            for (size_t i = 0; i < tube_nodes.size() - 1; i++) {
                helios::vec3 segment_start = tube_nodes[i];
                helios::vec3 segment_end = tube_nodes[i + 1];

                // Calculate distance from point to line segment
                helios::vec3 segment_vec = segment_end - segment_start;
                helios::vec3 point_vec = voxel_center - segment_start;

                float segment_length = segment_vec.magnitude();
                float segment_length_sq = segment_length * segment_length;
                if (segment_length_sq < 1e-10f) {
                    // Degenerate segment, use point distance
                    min_distance = std::min(min_distance, (voxel_center - segment_start).magnitude());
                } else {
                    float t = std::max(0.0f, std::min(1.0f, (point_vec * segment_vec) / segment_length_sq));
                    helios::vec3 closest_point = segment_start + t * segment_vec;
                    min_distance = std::min(min_distance, (voxel_center - closest_point).magnitude());
                }
            }

            if (min_distance <= proximity_distance) {
                nearby_branches.push_back(tube_id);
                branch_distances.push_back(min_distance);
            }
        }

        if (nearby_branches.empty()) {
            if (printmessages && voxels_with_leaf_area <= 5) {
                cout << "Voxel " << v << " has no nearby branches within distance " << proximity_distance << endl;
            }
            continue; // No nearby branches, skip this voxel
        }
        voxels_with_nearby_branches++;

        // Calculate number of leaves needed for this voxel
        int num_leaves = std::max(1, (int) round(voxel_leaf_area / prototype_leaf_area));

        if (printmessages && v < 5) { // Debug first few voxels
            cout << "Voxel " << v << ": leaf_area=" << voxel_leaf_area << ", prototype_area=" << prototype_leaf_area << ", num_leaves=" << num_leaves << endl;
        }

        // Place leaves near branches
        for (int leaf_idx = 0; leaf_idx < num_leaves; leaf_idx++) {
            // Select a nearby branch (weighted by proximity)
            float total_weight = 0.0f;
            std::vector<float> weights;
            for (float dist: branch_distances) {
                float weight = 1.0f / (1.0f + dist); // Closer branches have higher weight
                weights.push_back(weight);
                total_weight += weight;
            }

            // Random selection weighted by proximity
            float rand_val = context->randu() * total_weight;
            uint selected_branch_idx = 0;
            float cumulative_weight = 0.0f;
            for (size_t i = 0; i < weights.size(); i++) {
                cumulative_weight += weights[i];
                if (rand_val <= cumulative_weight) {
                    selected_branch_idx = i;
                    break;
                }
            }

            uint selected_tube_id = nearby_branches[selected_branch_idx];

            // Get a position near the selected branch
            helios::vec3 leaf_position;

            // Get tube nodes for the selected branch
            std::vector<helios::vec3> tube_nodes = context->getTubeObjectNodes(selected_tube_id);
            if (!tube_nodes.empty()) {
                // Find the closest segment to voxel center
                float min_dist = 1e6;
                helios::vec3 closest_point = tube_nodes[0];

                for (size_t i = 0; i < tube_nodes.size() - 1; i++) {
                    helios::vec3 segment_start = tube_nodes[i];
                    helios::vec3 segment_end = tube_nodes[i + 1];
                    helios::vec3 segment_vec = segment_end - segment_start;
                    helios::vec3 point_vec = voxel_center - segment_start;

                    float segment_length_sq = segment_vec.magnitude() * segment_vec.magnitude();
                    if (segment_length_sq > 0.0001f) {
                        float t = std::max(0.0f, std::min(1.0f, (point_vec * segment_vec) / segment_length_sq));
                        helios::vec3 point_on_segment = segment_start + t * segment_vec;
                        float dist = (voxel_center - point_on_segment).magnitude();
                        if (dist < min_dist) {
                            min_dist = dist;
                            closest_point = point_on_segment;
                        }
                    }
                }

                // Place leaf near the closest point on the branch, with some randomness
                helios::vec3 random_offset;
                random_offset.x = (context->randu() - 0.5f) * voxel_size.x * 0.3f;
                random_offset.y = (context->randu() - 0.5f) * voxel_size.y * 0.3f;
                random_offset.z = (context->randu() - 0.5f) * voxel_size.z * 0.3f;

                leaf_position = closest_point + random_offset;

                // Ensure leaf stays within voxel bounds
                helios::vec3 voxel_min = voxel_center - 0.5f * voxel_size;
                helios::vec3 voxel_max = voxel_center + 0.5f * voxel_size;
                leaf_position.x = std::max(voxel_min.x, std::min(voxel_max.x, leaf_position.x));
                leaf_position.y = std::max(voxel_min.y, std::min(voxel_max.y, leaf_position.y));
                leaf_position.z = std::max(voxel_min.z, std::min(voxel_max.z, leaf_position.z));
            } else {
                // Fallback: random position within voxel
                leaf_position = voxel_center;
                leaf_position.x += (context->randu() - 0.5f) * voxel_size.x * 0.8f;
                leaf_position.y += (context->randu() - 0.5f) * voxel_size.y * 0.8f;
                leaf_position.z += (context->randu() - 0.5f) * voxel_size.z * 0.8f;
            }

            // Sample leaf orientation from the angle distribution for this voxel
            float zenith_angle = 0.0f;
            float azimuth_angle = 0.0f;

            // Check if we have valid CDF data for this voxel
            bool has_valid_theta_cdf = (v < CDF_theta.size() && !CDF_theta[v].empty() && CDF_theta[v].back() > 0.001f);
            bool has_valid_phi_cdf = (v < CDF_phi.size() && !CDF_phi[v].empty() && CDF_phi[v].back() > 0.001f);

            // Sample zenith angle from CDF only
            if (has_valid_theta_cdf) {
                float rand_theta = context->randu();
                float normalized_rand = rand_theta * CDF_theta[v].back(); // Normalize by total CDF value
                for (uint bin = 0; bin < CDF_theta[v].size() && bin < Nbins; bin++) {
                    if (normalized_rand <= CDF_theta[v][bin] || bin == CDF_theta[v].size() - 1) {
                        zenith_angle = (float(bin) + context->randu()) * dtheta;
                        break;
                    }
                }
            } else {
                if (printmessages && total_leaves_placed < 5) {
                    cout << "Warning: No valid zenith CDF for voxel " << v << ", skipping leaf placement." << endl;
                }
                continue; // Skip this leaf if no valid CDF data
            }

            // Sample azimuth angle from CDF only
            if (has_valid_phi_cdf) {
                float rand_phi = context->randu();
                float normalized_rand = rand_phi * CDF_phi[v].back(); // Normalize by total CDF value
                for (uint bin = 0; bin < CDF_phi[v].size() && bin < Nbins; bin++) {
                    if (normalized_rand <= CDF_phi[v][bin] || bin == CDF_phi[v].size() - 1) {
                        azimuth_angle = (float(bin) + context->randu()) * dphi;
                        break;
                    }
                }
            } else {
                if (printmessages && total_leaves_placed < 5) {
                    cout << "Warning: No valid azimuth CDF for voxel " << v << ", skipping leaf placement." << endl;
                }
                continue; // Skip this leaf if no valid CDF data
            }

            helios::SphericalCoord leaf_orientation = helios::make_SphericalCoord(1.0f, zenith_angle, azimuth_angle);

            if (printmessages && total_leaves_placed < 3) { // Debug first few leaves
                cout << "Leaf " << total_leaves_placed << ": zenith=" << zenith_angle << ", azimuth=" << azimuth_angle << endl;
            }

            // Copy the prototype and apply transformations
            uint leaf_objID = context->copyObject(prototype_objID);

            // Apply rotations - zenith rotation around y-axis, azimuth rotation around z-axis
            if (zenith_angle > 0.001f) { // Only rotate if angle is meaningful
                context->rotateObject(leaf_objID, zenith_angle, "y");
            }
            if (azimuth_angle > 0.001f) { // Only rotate if angle is meaningful
                context->rotateObject(leaf_objID, azimuth_angle, "z");
            }

            context->translateObject(leaf_objID, leaf_position);

            // Add object data for tracking
            context->setObjectData(leaf_objID, "gridCell", (int) v);
            context->setObjectData(leaf_objID, "directFlag", 1);

            // Collect the leaf object ID
            leaf_object_IDs.push_back(leaf_objID);

            // Store in the reconstruction data structures for compatibility
            reconstructed_alphamasks_center.push_back(leaf_position);
            reconstructed_alphamasks_size.push_back(leaf_size);
            reconstructed_alphamasks_rotation.push_back(leaf_orientation);
            reconstructed_alphamasks_gridcell.push_back(v);
            reconstructed_alphamasks_direct_flag.push_back(1);

            total_leaves_placed++;
        }
    }

    // Delete the prototype object
    context->deleteObject(prototype_objID);

    if (printmessages) {
        cout << "done." << endl;
        cout << "Found " << voxels_with_leaf_area << " voxels with leaf area, " << voxels_with_nearby_branches << " with nearby branches." << endl;
        cout << "TreeQSM-based reconstruction placed " << total_leaves_placed << " leaves across " << Ncells << " voxels." << endl;
    }

    return leaf_object_IDs;
}