/** \file "PhysicsEngine.h" Primary header file for the physics engine plug-in, which simulates rigid-body dynamics with MuJoCo.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#ifndef HELIOS_PHYSICS_ENGINE_H
#define HELIOS_PHYSICS_ENGINE_H

#include "Context.h"

// Forward declarations of the MuJoCo types, so that including this header does not require the MuJoCo headers. Code that uses the model and data directly includes <mujoco/mujoco.h>.
struct mjModel_;
struct mjData_;

//! Settings of the physics simulation
struct PhysicsEngineParameters {
    //! Physics time step in seconds
    float timestep = 0.002f;
    //! MuJoCo integrator: "discrete", "implicitfast", "implicit", "Euler", or "RK4". The discrete integrator treats joint stiffness implicitly, so stiff short segments need no joint armature.
    std::string integrator = "discrete";
    //! MuJoCo constraint solver: "PGS", "Newton", or "CG". With the discrete integrator all three treat joint stiffness implicitly, and PGS is several times faster for articulations with thousands of links.
    std::string solver = "PGS";
    //! Time constant in seconds over which contact penetration is corrected. Smaller values let less of an articulation sink into kinematic bodies. Must be at least twice the time step.
    float contact_time_constant = 0.004f;
    //! Let the links of articulations collide with each other, within and across articulations, in addition to colliding with kinematic bodies. Linked pairs are skipped as listed in PhysicsArticulation::contact_exclusions.
    bool articulation_collisions = false;
    //! Distance in meters at which colliding articulation links start to push each other apart when articulation_collisions is enabled, so that they come to rest separated rather than slightly overlapping
    float articulation_collision_margin = 0.001f;
    //! Sliding friction coefficient of contacts
    float friction = 0.5f;
    //! Apply gravity
    bool gravity = true;
};

//! Type of a link collision shape
enum PhysicsShapeType {
    //! Cylinder with hemispherical ends along the local z-axis
    PHYSICS_SHAPE_CAPSULE,
    //! Sphere
    PHYSICS_SHAPE_SPHERE,
    //! Rectangular box
    PHYSICS_SHAPE_BOX,
    //! Ellipsoid
    PHYSICS_SHAPE_ELLIPSOID,
    //! Convex hull of a set of vertices
    PHYSICS_SHAPE_CONVEX_MESH
};

//! Collision shape attached to a link
struct PhysicsCollisionShape {
    //! Type of the shape
    PhysicsShapeType type = PHYSICS_SHAPE_SPHERE;
    //! Dimensions in meters. Capsule: x is the radius and y the half-length of the shaft. Sphere: x is the radius. Box and ellipsoid: half-sizes along each axis. Not used for a convex mesh.
    helios::vec3 size;
    //! Position of the shape center in the link frame
    helios::vec3 position;
    //! Orientation of the shape in the link frame as a unit quaternion (w,x,y,z)
    double orientation[4] = {1, 0, 0, 0};
    //! Vertices of a convex mesh in the link frame
    std::vector<helios::vec3> mesh_vertices;
    //! Color in the MuJoCo viewer
    helios::RGBcolor color = helios::make_RGBcolor(0.45f, 0.32f, 0.2f);
};

//! How a link is connected to its parent
enum PhysicsJointType {
    //! Rigidly attached to the parent
    PHYSICS_JOINT_WELDED,
    //! Elastic ball joint that resists rotation away from the rest pose
    PHYSICS_JOINT_BALL
};

//! One rigid link of an articulation
struct PhysicsLink {
    //! Name of the link, unique within the articulation
    std::string name;
    //! Index of the parent link, or -1 if the link is a root and is fixed to the world
    int parent_index = -1;
    //! World-space position of the link frame origin in the rest pose
    helios::vec3 position;
    //! World-space orientation of the link frame in the rest pose as a unit quaternion (w,x,y,z)
    float qw = 1, qx = 0, qy = 0, qz = 0;
    //! Mass in kg
    float mass = 0;
    //! Principal moments of inertia about the link frame axes in kg*m^2
    helios::vec3 inertia_diagonal;
    //! Connection to the parent link. Ignored for roots.
    PhysicsJointType joint_type = PHYSICS_JOINT_WELDED;
    //! Joint anchor in the parent link frame
    helios::vec3 anchor_in_parent;
    //! Rotational spring stiffness in N*m/rad
    float stiffness = 0;
    //! Rotational damping in N*m*s/rad
    float damping = 0;
    //! Artificial joint inertia in kg*m^2 that is needed for stiff joints with integrators that apply joint springs explicitly. Zero adds none.
    float armature = 0;
    //! Maximum joint rotation in radians. Zero leaves the joint unlimited.
    float max_rotation = 0;
    //! Collision shapes of the link
    std::vector<PhysicsCollisionShape> shapes;
};

//! Tree of rigid links connected by joints
struct PhysicsArticulation {
    //! Name of the articulation, unique within the simulation
    std::string name;
    //! Links ordered so that every parent precedes its children
    std::vector<PhysicsLink> links;
    //! Pairs of link indices that never collide. Only used when PhysicsEngineParameters::articulation_collisions is enabled. MuJoCo does not filter contacts with a parent that is fixed to the world, so such pairs must be listed explicitly.
    std::vector<std::pair<uint, uint>> contact_exclusions;
};

//! Rigid-body physics simulation with the MuJoCo engine
/**
 * The plug-in is deliberately specific to MuJoCo: getMuJoCoModel() and getMuJoCoData() give direct access to the model and state for anything the class does not wrap.
 * The MuJoCo state is the source of truth. The Context is only a copy of it for rendering and other plug-ins, updated when updateContextGeometry() is called.
 *
 * One compiled model can have many independently stepped instances (see createInstance()), which is how environments are run in parallel.
 */
class PhysicsEngine {
public:
    //! Constructor
    /**
     * \param[in] context Pointer to the Helios context holding the geometry that follows the simulation.
     */
    explicit PhysicsEngine(helios::Context *context);

    //! Destructor
    ~PhysicsEngine();

    PhysicsEngine(const PhysicsEngine &) = delete;
    PhysicsEngine &operator=(const PhysicsEngine &) = delete;

    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Add an articulation to the simulation. Must be called before initialize().
    /**
     * \param[in] articulation Description of the articulation.
     * \return ID of the articulation.
     */
    uint addArticulation(const PhysicsArticulation &articulation);

    //! Add a kinematic box to the simulation, and a matching box object to the Context. Must be called before initialize().
    /**
     * \param[in] center Initial position of the box center.
     * \param[in] size Edge lengths of the box along x, y, and z.
     * \param[in] color Color of the Context box object.
     * \return ID of the kinematic box, used to move it with setKinematicBoxPosition().
     */
    uint addKinematicBox(const helios::vec3 &center, const helios::vec3 &size, const helios::RGBcolor &color = helios::make_RGBcolor(0.9f, 0.35f, 0.1f));

    //! Make a Context object follow a link rigidly. Must be called before initialize().
    /**
     * \param[in] objID ID of the Context object.
     * \param[in] articulationID ID of the articulation returned by addArticulation().
     * \param[in] link_index Index of the link within the articulation.
     */
    void bindObjectToLink(uint objID, uint articulationID, uint link_index);

    //! Make each node of a Context tube object follow a link, so that the tube bends with the articulation. Must be called before initialize().
    /**
     * \param[in] objID ID of the Context tube object.
     * \param[in] articulationID ID of the articulation returned by addArticulation().
     * \param[in] node_link_indices Index of the link that each tube node follows.
     */
    void bindTubeNodesToLinks(uint objID, uint articulationID, const std::vector<uint> &node_link_indices);

    //! Build and compile the MuJoCo model from what was added, and create the first simulation instance
    /**
     * \param[in] params Settings of the simulation.
     */
    void initialize(const PhysicsEngineParameters &params = PhysicsEngineParameters());

    //! Create another simulation instance of the compiled model, starting from the initial state
    /**
     * Instances share the model and have entirely separate state. Instance 0 is created by initialize().
     * \return ID of the new instance.
     */
    uint createInstance();

    //! Get the number of simulation instances
    [[nodiscard]] uint getInstanceCount() const;

    //! Return an instance to its initial state
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance = 0);

    //! Set the position and velocity of a kinematic box. They take effect at the next step.
    /**
     * The box is placed at the given center and then moves at the given velocity, which is held through every step until
     * the next call. Giving the box's true velocity matters: contacts are resolved against the box's velocity, so a box
     * that is repositioned each step with zero velocity lets what it touches sink into it.
     * \param[in] boxID ID of the box returned by addKinematicBox().
     * \param[in] center New position of the box center.
     * \param[in] velocity Velocity of the box in m/s.
     * \param[in] instance ID of the simulation instance.
     */
    void setKinematicBoxPosition(uint boxID, const helios::vec3 &center, const helios::vec3 &velocity, uint instance = 0);

    //! Advance one simulation instance
    /**
     * \param[in] step_count Number of physics time steps to take.
     * \param[in] instance ID of the simulation instance.
     */
    void step(uint step_count = 1, uint instance = 0);

    //! Advance all simulation instances, in parallel
    /**
     * \param[in] step_count Number of physics time steps to take in each instance.
     * \param[in] thread_count Number of threads to use. Zero uses one per hardware thread.
     */
    void stepInstances(uint step_count = 1, uint thread_count = 0);

    //! Move the Context geometry that follows the simulation to the poses of one instance
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void updateContextGeometry(uint instance = 0);

    //! Write the compiled model in MuJoCo MJCF format, so it can be opened in the MuJoCo viewer
    /**
     * \param[in] filename Path of the output file (should have .xml extension).
     */
    void writeMJCF(const std::string &filename) const;

    //! Get the compiled MuJoCo model, for direct use of the MuJoCo API
    [[nodiscard]] mjModel_ *getMuJoCoModel() const;

    //! Get the MuJoCo state of an instance, for direct use of the MuJoCo API
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] mjData_ *getMuJoCoData(uint instance = 0) const;

    //! Get the simulated time in seconds since the instance was created or reset
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getSimulationTime(uint instance = 0) const;

    //! Get the physics time step in seconds
    [[nodiscard]] float getTimestep() const;

    //! Get the total number of articulation links in the simulation
    [[nodiscard]] uint getLinkCount() const;

    //! Get the number of contacts found in the last step
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] uint getContactCount(uint instance = 0) const;

    //! Get the largest displacement of any articulation link from its rest position, in meters
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getMaxLinkDisplacement(uint instance = 0) const;

    //! Get the largest speed of any articulation link in m/s, which falls to zero as the articulations come to rest
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getMaxLinkSpeed(uint instance = 0) const;

    //! Get the deepest penetration into a kinematic box among the contacts of the last step, in meters
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getMaxKinematicBoxPenetration(uint instance = 0) const;

    //! Get the number of contacts between articulation links that overlap by more than a given depth, among the contacts of the last step
    /**
     * Links only collide with each other when PhysicsEngineParameters::articulation_collisions is enabled; otherwise the count is zero. Overlap is measured between the collision shapes, not the Context geometry.
     * Right after initialize() the count is that of the articulations as they were added.
     * \param[in] min_depth Overlap depth in meters below which a contact is not counted.
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] uint getLinkOverlapCount(float min_depth = 0.f, uint instance = 0) const;

    //! Get the deepest overlap between two articulation links among the contacts of the last step, in meters
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getMaxLinkOverlapDepth(uint instance = 0) const;

    //! Get the current world position of an articulation link
    /**
     * \param[in] articulationID ID of the articulation returned by addArticulation().
     * \param[in] link_index Index of the link within the articulation.
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getLinkPosition(uint articulationID, uint link_index, uint instance = 0) const;

    //! Disable informational messages
    void disableMessages();

    //! Enable informational messages
    void enableMessages();

private:
    struct ArticulationRecord {
        PhysicsArticulation description;
        std::vector<int> link_body_IDs; // MuJoCo body ID of each link
        std::vector<size_t> link_rest_pose_indices;
    };

    struct BoxRecord {
        helios::vec3 center;
        helios::vec3 size;
        uint objID;
        int body_ID = -1;
        int qpos_address = -1; // first of the box's three slide joints
        int dof_address = -1;
    };

    // State of one simulation instance
    struct Instance {
        mjData_ *data = nullptr;
        std::vector<helios::vec3> box_velocities;
    };

    // Rest pose of a MuJoCo body, against which its current pose gives the rigid motion applied to its geometry
    struct BodyRestPose {
        int body_ID;
        helios::vec3 position;
        float rotation[9]; // row-major
    };

    struct ObjectLinkRequest {
        uint objID;
        uint articulationID;
        uint link_index;
    };

    struct TubeLinkRequest {
        uint objID;
        uint articulationID;
        std::vector<uint> node_link_indices;
    };

    struct PrimitiveBinding {
        uint UUID;
        size_t rest_pose_index;
        float rest_transform[16];
    };

    struct TubeBinding {
        uint objID;
        std::vector<size_t> node_rest_pose_index;
        std::vector<helios::vec3> node_local_position;
    };

    [[nodiscard]] std::string buildMJCF(const PhysicsEngineParameters &params) const;
    void writeLinkBody(std::ostringstream &xml, std::ostringstream &assets, const PhysicsArticulation &articulation, size_t link_index, const std::vector<std::vector<size_t>> &children, int depth) const;
    void bindContextGeometry();
    void bindObjectPrimitives(uint objID, size_t rest_pose_index);
    size_t addRestPose(int body_ID);
    void stepInstance(Instance &instance, uint step_count) const;
    void checkSimulationHealth(const mjData_ *data) const;
    [[nodiscard]] bool isLinkContact(const mjData_ *data, int contact_index) const;
    void requireInitialized(const std::string &method) const;
    void requireNotInitialized(const std::string &method) const;
    [[nodiscard]] const Instance &getInstance(uint instance, const std::string &method) const;
    void requireLink(uint articulationID, uint link_index, const std::string &method) const;

    helios::Context *context;

    std::vector<ArticulationRecord> articulations;
    std::vector<BoxRecord> boxes;
    std::vector<ObjectLinkRequest> object_link_requests;
    std::vector<TubeLinkRequest> tube_link_requests;

    std::vector<BodyRestPose> rest_poses;
    std::vector<PrimitiveBinding> primitive_bindings;
    std::vector<TubeBinding> tube_bindings;
    std::vector<char> body_is_kinematic_box; // indexed by MuJoCo body ID

    std::string mjcf_text; // MJCF model that was compiled

    mjModel_ *model = nullptr;
    std::vector<Instance> instances;

    bool printmessages = true;
};

#endif
