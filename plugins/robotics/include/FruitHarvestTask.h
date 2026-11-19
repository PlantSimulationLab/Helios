/** \file "FruitHarvestTask.h" A reinforcement learning task: a mobile arm with a vacuum cup picks fruit from trees, in many parallel simulations.

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

#ifndef HELIOS_FRUIT_HARVEST_TASK_H
#define HELIOS_FRUIT_HARVEST_TASK_H

#include "PlantPhysics.h"
#include "VacuumCup.h"

//! How an episode of a FruitHarvestTask ended
enum FruitHarvestOutcome {
    //! The episode has not ended
    FRUIT_HARVEST_RUNNING = 0,
    //! The fruit detached while the cup held it
    FRUIT_HARVEST_SUCCESS = 1,
    //! The episode ran out of control steps
    FRUIT_HARVEST_TIMEOUT = 2,
    //! The cup's seal broke while it held the fruit
    FRUIT_HARVEST_SLIPPED = 3,
    //! The fruit detached without being held
    FRUIT_HARVEST_KNOCKED_OFF = 4,
    //! The robot pressed on the tree hard enough for its protective stop
    FRUIT_HARVEST_COLLISION = 5
};

//! What happened in an episode of a FruitHarvestTask, for understanding how a controller succeeds or fails
struct FruitHarvestEpisodeRecord {
    //! Force in newtons it took to detach the fruit, drawn for the episode
    float detach_force = 0;
    //! Load in newtons on the fruit's stem when the episode ended
    float final_stem_load = 0;
    //! Pull in newtons of the cup on the fruit when the episode ended
    float final_grip_force = 0;
    //! Unit direction in which the cup last pulled the fruit (zero if it never did)
    helios::vec3 pull_direction;
    //! Largest contact force in newtons of the robot on the tree, other than on the fruit being picked
    float peak_contact_force = 0;
    //! Farthest in meters the wood bearing the fruit moved from where it settled
    float peak_wood_motion = 0;
    //! Largest speed of the tool in m/s
    float peak_tool_speed = 0;
    //! Control steps until the cup first held the fruit, or -1 if it never did
    int steps_to_grip = -1;
    //! Control steps in the episode
    int steps = 0;
};

//! What the wrist camera of a simulation sees, as fractions of its image (see FruitHarvestTask::measureWristCameraView())
struct FruitHarvestCameraView {
    //! Showing the fruit being picked
    float target_fruit = 0;
    //! Showing other fruit
    float other_fruit = 0;
    //! Showing leaves
    float leaves = 0;
    //! Showing leaves within 15 cm of the camera, which a real arm would have pushed aside
    float near_leaves = 0;
    //! Showing wood
    float wood = 0;
};

//! A position servo on a joint of the robot of a FruitHarvestTask (see FruitHarvestTask::getJointServos())
struct FruitHarvestJointServo {
    //! Name of the joint in harvest_arm.urdf
    std::string joint;
    //! Stiffness in N/m or N m/rad
    float stiffness = 0;
    //! Damping in N s/m or N m s/rad
    float damping = 0;
    //! Largest force or torque in N or N m: the joint's effort limit in harvest_arm.urdf
    float force_limit = 0;
    //! Armature in kg or kg m^2
    float armature = 0;
};

//! Settings of a FruitHarvestTask. The defaults are the apple harvest of samples/robotics_apple_harvest: a mature tall-spindle apple wall and the harvest arm shipped with the plug-in.
struct FruitHarvestTaskParameters {

    // The tree

    //! Plant model from the plant architecture library. It must bear fruit at plant_age within reach of the arm.
    std::string plant_model = "apple_fruitingwall";
    //! Age in days at which each tree is grown. The default is the harvest of the sixth leaf, when the tall spindle is a mature wall.
    float plant_age = 2114.f;
    //! Keep the trees' leaves. Leaves are fixed to the wood and collide with nothing, so they only change what a camera sees, and they make a tree much slower to build and draw.
    bool leaves = false;
    //! Young's modulus of the wood in Pa
    float wood_elastic_modulus = 5e9f;
    //! Density of the wood in kg/m^3
    float wood_density = 750.f;
    //! Mass of each fruit in kg
    float fruit_mass = 0.15f;
    //! Lowest height in meters of a fruit that can be picked
    float min_fruit_height = 0.8f;
    //! Highest height in meters of a fruit that can be picked
    float max_fruit_height = 1.75f;
    //! The fruit of each tree that episodes pick from: the reachable fruit nearest the robot (smallest x), at most this many
    uint candidates_per_tree = 8;
    //! Internodes behind each candidate fruit, toward the trunk, that are simulated as flexible; all other wood is rigid, which keeps almost all of the speed of rigid wood
    uint flexible_internodes_per_candidate = 8;
    //! Physics steps (2 ms) for which each tree settles under gravity when it is built. Every episode starts from the pose reached, at rest.
    uint settling_physics_steps = 10000;

    // The robot: the arm of plugins/robotics/assets/harvest_arm.urdf on a platform that drives along the row (y)

    //! Where the robot's platform stands. The default is in the alley beside the row, which runs along y through x = 0.
    helios::vec3 robot_base = helios::make_vec3(-1.55f, 0, 0);
    //! Joint positions the arm starts each episode in (drive in meters, then yaw, shoulder, elbow, wrist and roll in degrees). The drive is set for each episode, near the fruit.
    std::vector<float> start_pose = {0.f, 0.f, 110.f, -150.f, 40.f, 0.f};
    //! Contact force in newtons of the robot on anything other than the fruit being picked at which its protective stop ends the episode
    float protective_stop_force = 50.f;
    //! Give the whole arm geometry, for showing the robot from outside, rather than only the tool, which is all the wrist camera sees
    bool show_arm = false;

    // The cup and the fruit

    //! The vacuum cup on the tool
    VacuumCupParameters cup;
    //! Smallest force in newtons to detach a fruit; each episode draws its fruit's force uniformly between this and max_detach_force
    float min_detach_force = 14.f;
    //! Largest force in newtons to detach a fruit
    float max_detach_force = 28.f;

    // Control

    //! Physics steps (2 ms) per control step: 20 is 25 Hz
    uint physics_steps_per_control = 20;
    //! Control steps after which an episode times out
    uint max_control_steps = 250;
    //! Fastest speed in m/s the actions can command the tool to move
    float max_tool_speed = 0.3f;
    //! Farthest in meters the commanded tool position can lead the actual one, so the target cannot wind up while the arm is blocked
    float max_target_lead = 0.08f;

    // Episodes

    //! Farthest in meters along the row that the robot starts from the fruit
    float start_offset = 0.2f;
    //! Megabytes of MuJoCo memory for each simulation (see PhysicsEngineParameters::memory_arena_megabytes). Episodes use well under 1 MB.
    float memory_arena_megabytes = 32.f;
};

//! A reinforcement learning task: a robot arm with a vacuum cup reaches a fruit on a tree, seals the cup on it, and pulls until it detaches
/**
 * The task holds several trees (scene variants), each grown from its own random seed and simulated in its own PhysicsEngine, and several parallel simulations (copies) of each. Copies are numbered
 * tree by tree, the copies of each tree next to each other. All copies are reset, stepped, observed and rewarded together, on worker threads, through arrays with one row per copy, so that a Python
 * vector environment needs one call per control step.
 *
 * Each episode picks a fruit at random from its tree's candidates, draws the force to detach it, and starts the arm near it. Each control step applies every copy's action (getActionSize() values in
 * [-1, 1]: the tool's velocity along x, y and z as a fraction of FruitHarvestTaskParameters::max_tool_speed, and a positive value to seal the cup if it can), then simulates until the control step ends,
 * the fruit detaches, the seal breaks, or the protective stop trips. A copy whose episode ends is reset at once, so the observations returned are the start of its next episode, and its last
 * observations are returned separately.
 *
 * Observations (getObservationSize() values per copy): the fruit's position relative to the cup's face (3); the gap between the cup's lip and the fruit (1); the tool's position relative to the platform (3)
 * and its linear velocity (3); the drive in meters and the other five joints in radians (6); whether the cup holds the fruit (1); the seal's load as a fraction of what breaks it (1); the contact force on
 * the tree in units of 10 N (1); the fraction of the episode elapsed (1); the offset from the cup's face to the axis of the nearest wood, as the tree stands at rest (3); and the fruit's offset from the cup's
 * axis (3).
 *
 * The reward is shaped for learning: -0.005 per step; 5 per meter the cup gets closer to the fruit before holding it; 1 when it first holds it; 3 per max_detach_force of increase in the stem's load while it
 * holds it; -0.02 per newton of contact on the tree (up to the protective stop force); 10 for a success and -2 for a slip, a fruit knocked off or a protective stop. Evaluation should use the outcomes and
 * episode records, not the reward.
 *
 * The trees, the cameras and the movie functions need the plant architecture plug-in, which the physics engine loads with plants. The wrist and movie cameras need the visualizer plug-in, and the
 * radiation movie cameras the radiation and leafoptics plug-ins, in the project's plug-ins; without them those functions raise an error.
 */
class FruitHarvestTask {
public:
    //! Values in each copy's observation
    static constexpr uint observation_size = 26;
    //! Values in each copy's action
    static constexpr uint action_size = 4;

    //! Build the trees and their simulations
    /**
     * Each tree takes about as long to build as to grow, and then settles for FruitHarvestTaskParameters::settling_physics_steps.
     * \param[in] tree_seeds Random seed of each tree; each must give a tree with fruit within reach (see treeHasReachableFruit())
     * \param[in] copies_per_tree Parallel simulations of each tree
     * \param[in] thread_count Threads that run copies, counting the calling thread. Zero uses one per hardware thread.
     * \param[in] seed Seed of the random draws of the episodes (fruit, force to detach and start). Copy i of tree t draws from seed + 1000 t + i.
     * \param[in] parameters Settings of the task
     */
    FruitHarvestTask(const std::vector<int> &tree_seeds, uint copies_per_tree, uint thread_count, uint seed, const FruitHarvestTaskParameters &parameters = FruitHarvestTaskParameters());

    //! Destructor
    ~FruitHarvestTask();

    FruitHarvestTask(const FruitHarvestTask &) = delete;
    FruitHarvestTask &operator=(const FruitHarvestTask &) = delete;

    //! Number of parallel simulations (copies) of all trees
    [[nodiscard]] uint getCopyCount() const;

    //! Values in each copy's observation
    [[nodiscard]] static uint getObservationSize();

    //! Values in each copy's action
    [[nodiscard]] static uint getActionSize();

    //! Control steps after which an episode times out
    [[nodiscard]] uint getMaxControlSteps() const;

    //! The settings the task was built with
    [[nodiscard]] const FruitHarvestTaskParameters &getParameters() const;

    //! Start a new episode in every copy
    /**
     * \param[out] observations getCopyCount() rows of getObservationSize() values
     */
    void reset(std::vector<float> &observations);

    //! Advance every copy by one control step
    /**
     * A copy whose episode ends is reset, so its row of observations is the start of its next episode, and its row of terminal_observations holds how the last one ended. Rows of copies that go on
     * are left unchanged in terminal_observations.
     * \param[in] actions getCopyCount() rows of getActionSize() values in [-1, 1]
     * \param[out] observations getCopyCount() rows of getObservationSize() values
     * \param[out] rewards Reward of each copy for the step
     * \param[out] outcomes FruitHarvestOutcome of each copy: FRUIT_HARVEST_RUNNING unless its episode ended in the step
     * \param[in,out] terminal_observations getCopyCount() rows of getObservationSize() values; resized if needed
     */
    void step(const std::vector<float> &actions, std::vector<float> &observations, std::vector<float> &rewards, std::vector<int> &outcomes, std::vector<float> &terminal_observations);

    //! Start this share of episodes with the cup a few centimeters short of its fruit and somewhat out of line with it, instead of with the arm in its start pose
    /**
     * For training only, so that it practices the last part of the task; evaluation keeps the default of 0. A placement the arm cannot reach without touching anything falls back to the start pose.
     * Applies from each copy's next episode.
     * \param[in] fraction From 0 to 1
     */
    void setNearStartFraction(float fraction);

    //! For showing: after a success, go on for this many control steps (0 by default), with the cup backing straight away from the tree whatever the action, carrying the freed fruit while it holds it
    /**
     * The episode then ends as a success, with the reward and record it had when the fruit detached. Not for training or evaluation, since the extra steps take time and teach nothing.
     */
    void setFollowThrough(uint control_steps);

    //! The record of a copy's last finished episode
    [[nodiscard]] FruitHarvestEpisodeRecord getLastEpisode(uint copy) const;

    //! Where the fruit a copy is picking settled when its tree was built
    [[nodiscard]] helios::vec3 getTargetFruitPosition(uint copy) const;

    //! Whether a tree seed gives a tree with fruit within reach of the arm. Takes about as long as growing the tree.
    [[nodiscard]] static bool treeHasReachableFruit(int tree_seed, const FruitHarvestTaskParameters &parameters = FruitHarvestTaskParameters());

    //! The candidate fruit of a tree: the reachable fruit nearest the robot (smallest x), at most FruitHarvestTaskParameters::candidates_per_tree. Raises an error if there are none.
    /**
     * \param[in] articulation The tree as PlantArchitecture::getPlantArticulation() gives it
     * \param[in] parameters Settings of the task
     * \return Indices of the candidate fruit links
     */
    [[nodiscard]] static std::vector<uint> chooseCandidateFruit(const PlantArticulation &articulation, const FruitHarvestTaskParameters &parameters);

    //! The internodes simulated as flexible: FruitHarvestTaskParameters::flexible_internodes_per_candidate behind each candidate fruit
    [[nodiscard]] static std::vector<uint> chooseFlexibleInternodes(const PlantArticulation &articulation, const std::vector<uint> &candidates, const FruitHarvestTaskParameters &parameters);

    //! The position servos on the robot's joints, in the order of the joints: drive, yaw, shoulder, elbow, wrist, roll
    [[nodiscard]] static std::vector<FruitHarvestJointServo> getJointServos();

    //! The center of the cup's mouth in the frame of the robot's tool link; the cup's axis is the tool's x axis
    [[nodiscard]] static helios::vec3 getCupFace();

    //! The physical properties of the trees' wood and fruit, as getPlantArticulation() and writePlantStructureUSD() take them
    [[nodiscard]] static USDExportParameters plantArticulationParameters(const FruitHarvestTaskParameters &parameters);

    // ---- The wrist camera (needs the visualizer plug-in) ---- //

    //! Add a camera on the robot's wrist, behind and above the cup and looking along it, to every tree, with images of width by height pixels
    /**
     * Call it on the thread that will render, since OpenGL is tied to a thread. It cannot be used with the movie cameras.
     */
    void enableWristCamera(uint width, uint height);

    //! Render the wrist camera of every copy
    /**
     * \param[out] images getCopyCount() images of height rows of width pixels, 3 bytes each (red, green, blue), the first row at the top
     */
    void renderWristCameras(std::vector<unsigned char> &images);

    //! Render the wrist camera of one copy and write it to an image file (JPEG or PNG, by the file's extension)
    void saveWristCameraImage(uint copy, const std::string &filename);

    //! Measure what the wrist camera of a copy sees. Slow, since it renders the image again: for understanding a policy, not for training.
    [[nodiscard]] FruitHarvestCameraView measureWristCameraView(uint copy);

    // ---- Showing simulations from outside, for movies ---- //

    //! Number of links of the tree simulated by a copy
    [[nodiscard]] uint getPlantLinkCount(uint copy) const;

    //! Names of the links of the tree simulated by a copy, parents first, as PlantArchitecture::getPlantArticulation() names them
    [[nodiscard]] std::vector<std::string> getPlantLinkNames(uint copy) const;

    //! The rest pose of every link of the tree simulated by a copy: getPlantLinkCount() rows of x, y, z, qw, qx, qy, qz in the world frame
    [[nodiscard]] std::vector<float> getRestLinkPoses(uint copy) const;

    //! The pose of a copy, for showing it (a pose shown can come from this simulation or from another simulator of the same tree)
    /**
     * \param[in] copy Copy
     * \param[in] last_episode The pose as the copy's last finished episode ended, rather than now
     * \param[out] link_poses The world pose of every link of the tree as getRestLinkPoses() gives them, NaN for a link without a body of its own
     * \param[out] robot_joints The robot's joint positions (drive in meters, the others in degrees)
     */
    void getDisplayState(uint copy, bool last_episode, std::vector<float> &link_poses, std::vector<float> &robot_joints) const;

    //! Draw the rest of the row around each tree: neighbors_each_side trees on each side along the row, and, if asked, a tall-spindle trellis (wires, a stake at each tree, a line post every ten trees)
    /**
     * The neighbors are grown like each tree, from later random draws. None of it is simulated: nothing collides with it. Call it before enabling a camera.
     */
    void addMovieScenery(uint neighbors_each_side, bool trellis);

    //! Add a camera for showing each tree from outside, with images of width by height pixels (needs the visualizer plug-in). Call it on the thread that will render. It cannot be used with the wrist camera.
    void enableMovieCamera(uint width, uint height);

    //! Render a copy from eye looking at look_at, with its tree in the given link poses and the robot at the given joint positions, as getDisplayState() gives them
    /**
     * A NaN link pose means the link is carried by its parent as at rest. This leaves the copy in the pose shown, so a task used for this should only be used for showing.
     * \param[out] image height rows of width pixels, 3 bytes each (red, green, blue), the first row at the top
     */
    void renderMovieFrame(uint copy, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, const helios::vec3 &eye, const helios::vec3 &look_at, std::vector<unsigned char> &image);

    //! As renderMovieFrame(), seen from where the wrist camera would be, with its field of view and the movie camera's image size
    void renderWristFrame(uint copy, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, std::vector<unsigned char> &image);

    //! Draw movies with radiation cameras rather than the visualizer (needs the radiation and leafoptics plug-ins)
    /**
     * Adds a camera from outside of width by height pixels and one on the wrist of wrist_width by wrist_height, each tracing antialiasing_samples rays per pixel, in the sun and sky of the ASTM G173
     * spectra, with camera flux smoothing so that curved surfaces shade smoothly. Leaves get LeafOptics' default spectra, bark and fruit the spectra of their texture's classes, and the robot, ground and
     * trellis color references close to the colors the visualizer draws them in. The ground becomes an adaptive tile, finest in the alley beside the row. Call it after addMovieScenery(), and instead of
     * enableMovieCamera().
     *
     * With periodic_repeats above 0, the drawn row (trees, trellis and ground) repeats periodic_repeats times each way along it through a periodic boundary, and the robot does not repeat: the ground then
     * spans the row's length exactly, canopy hanging past an end of the row moves to the other end, and anything still crossing an end is cut off. With row_repeats above 0, the row repeats row_repeats
     * times on each side across the rows, 3.5 m apart, in the same way (the ground then spans one row spacing, centred 0.25 m into the robot's alley so that the robot lies within it).
     */
    void enableRadiationMovieCameras(uint width, uint height, uint wrist_width, uint wrist_height, uint antialiasing_samples, uint periodic_repeats, uint row_repeats);

    //! As renderMovieFrame(), with the radiation cameras: renders the view from eye looking at look_at and the wrist camera's view of the same pose, and writes them as JPEG images in a directory
    /**
     * \param[out] movie_image_file Path of the image of the camera from outside
     * \param[out] wrist_image_file Path of the image of the wrist camera
     */
    void renderRadiationMovieFrame(uint copy, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, const helios::vec3 &eye, const helios::vec3 &look_at, const std::string &directory, std::string &movie_image_file,
                                   std::string &wrist_image_file);

private:
    struct Tree;
    struct Copy;
    struct Simulations;

    FruitHarvestTaskParameters parameters;
    std::vector<std::unique_ptr<Tree>> trees;
    std::vector<std::unique_ptr<Copy>> copies;
    std::unique_ptr<Simulations> simulations; // destroyed before the trees it steps

    Copy &requireCopy(uint copy, const std::string &method);
    const Copy &requireCopy(uint copy, const std::string &method) const;
    void resetCopy(Copy &copy);
    void writeObservation(const Copy &copy, float *observation) const;
    void finishStep(Copy &copy, float *observation, float &reward, int &outcome, float *terminal_observation);
    void finishFollowThroughStep(Copy &copy, float *observation, float &reward, int &outcome, float *terminal_observation);
    bool placeCupNearFruit(Copy &copy, const std::vector<float> &start_pose);
    void renderMovieFrameOf(Copy &copy, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, const helios::vec3 &eye, const helios::vec3 &look_at, bool wrist_view, std::vector<unsigned char> &image);
};

#endif
