// Apple harvesting environment: a robot arm with a vacuum cup reaches a fruit on a fruiting-wall apple tree, grips it, and pulls until it detaches.
// A C interface, so that it can be driven from Python (harvest_env.py) as well as from C++ (main.cpp).

#ifndef APPLE_HARVEST_ENV_H
#define APPLE_HARVEST_ENV_H

#ifdef __cplusplus
extern "C" {
#endif

//! How an episode ended
enum HarvestOutcome {
    HARVEST_RUNNING = 0, //!< The episode has not ended
    HARVEST_SUCCESS = 1, //!< The fruit detached while gripped
    HARVEST_TIMEOUT = 2, //!< Time ran out
    HARVEST_SLIPPED = 3, //!< The pull exceeded the grip of the cup
    HARVEST_KNOCKED_OFF = 4, //!< The fruit detached without being gripped
    HARVEST_COLLISION = 5 //!< The arm pressed on the tree hard enough for a protective stop
};

//! Check whether a tree seed gives a tree with fruit within reach of the robot. A tree without any cannot be used in an environment. Takes about as long as growing the tree.
int harvest_env_tree_has_reachable_fruit(int tree_seed);

//! Write a tree, without leaves, for simulating the task in IsaacSim: <directory>/tree_<seed>.usda from PlantArchitecture::writePlantStructureUSD(), and tree_<seed>.json with how each link's joint is simulated here, the candidate fruit, the wood, and the task's constants. The directory must exist. Returns 0 and prints the reason on failure.
int harvest_env_export_tree(int tree_seed, const char *directory);

//! Create an environment with instances_per_tree parallel simulations of each tree. With leaves nonzero, trees keep their leaves, which can hide things from the camera but are fixed to the wood and collide with nothing. Returns null and prints the reason on failure.
void *harvest_env_create(const int *tree_seeds, int tree_count, int instances_per_tree, int thread_count, int seed, int leaves);

//! As harvest_env_create(), and with show_arm nonzero the whole arm has geometry for showing the robot from outside, rather than only the tool
void *harvest_env_create_for_movie(const int *tree_seeds, int tree_count, int instances_per_tree, int thread_count, int seed, int leaves, int show_arm);

void harvest_env_destroy(void *env);
int harvest_env_count(const void *env);
int harvest_env_observation_size(void);
int harvest_env_action_size(void);
int harvest_env_max_steps(void);

//! Reset every simulation. observations has room for count * observation_size floats.
void harvest_env_reset(void *env, float *observations);

//! Advance every simulation by one control step. A simulation that ends is reset, so observations is the start of its next episode, and terminal_observations holds how the last one ended.
void harvest_env_step(void *env, const float *actions, float *observations, float *rewards, int *outcomes, float *terminal_observations);

//! What happened in a simulation's last finished episode, for checking how a policy succeeds
typedef struct {
    float detach_force; //!< N, the force it took to detach this fruit
    float final_stem_load; //!< N, the load on the stem when the episode ended
    float final_grip_force; //!< N, the pull of the cup on the fruit when the episode ended
    float pull_x, pull_y, pull_z; //!< Direction from the fruit toward the cup when the episode ended
    float peak_contact_force; //!< N, the largest total contact force between the robot and the tree
    float peak_wood_motion; //!< m, the farthest the wood bearing the fruit moved from where it settled
    float peak_tool_speed; //!< m/s
    int steps_to_grip; //!< Control steps until the fruit was gripped, or -1 if it never was
    int steps; //!< Control steps in the episode
} HarvestEpisodeRecord;

//! Add a camera on the robot's wrist to every simulation, with images of width by height pixels. Call it on the thread that will call harvest_env_render(), since OpenGL is tied to a thread. Returns 0 and prints the reason on failure.
int harvest_env_enable_camera(void *env, int width, int height);

//! Render the camera of every simulation, one at a time. images has room for count * height * width * 3 bytes, RGB with the first row at the top of the image.
void harvest_env_render(void *env, unsigned char *images);

//! Write the camera image of one simulation to a JPEG file, for looking at it
void harvest_env_save_camera_image(void *env, int index, const char *filename);

//! What the camera of a simulation sees now, as fractions of the image
typedef struct {
    float target_fruit; //!< Showing the fruit being reached for
    float other_fruit; //!< Showing other fruit
    float leaves; //!< Showing leaves
    float near_leaves; //!< Showing leaves within 15 cm of the camera, which a real arm would have pushed aside
    float wood; //!< Showing wood
} HarvestCameraView;

//! Measure what the camera of one simulation sees. Slow, since it draws the image again: for understanding a policy, not for training.
void harvest_env_measure_camera_view(void *env, int index, HarvestCameraView *view);

// Showing simulations from outside, for movies. A pose shown can come from this simulation or from another simulator of the same tree.

//! The number of links of the plant in a simulation, and the name of each, parents first
int harvest_env_link_count(const void *env, int index);
const char *harvest_env_link_name(const void *env, int index, int link);

//! The rest pose of every link of the plant: link_count * 7 floats, x, y, z, qw, qx, qy, qz in the world frame
void harvest_env_rest_link_poses(const void *env, int index, float *link_poses);

//! The world pose of every link of the plant (as harvest_env_rest_link_poses(), NaN for a link that is not simulated) and the robot's 6 joint positions
//! (m, degrees), now or, with last_episode nonzero, as the simulation's last finished episode ended. Returns 0 and prints the reason on failure.
int harvest_env_display_state(const void *env, int index, int last_episode, float *link_poses, float *robot_joints);

//! Where the fruit that a simulation is reaching for settled
void harvest_env_target_fruit(const void *env, int index, float *position);

//! Add a camera for seeing a simulation from outside, with images of width by height pixels. It cannot be used with the wrist camera. Call it on the thread
//! that will render. Returns 0 and prints the reason on failure.
int harvest_env_enable_movie_camera(void *env, int width, int height);

//! Render one simulation from eye looking at look_at, with its plant in the given link poses (NaN: carried by the parent link as at rest) and the robot at
//! the given joint positions, as harvest_env_display_state() gives them. image has room for height * width * 3 bytes, RGB with the first row at the top.
//! This leaves the simulation in the pose shown, so an environment used for this should only be used for showing. Returns 0 and prints the reason on failure.
int harvest_env_render_movie_frame(void *env, int index, const float *link_poses, const float *robot_joints, const float *eye, const float *look_at, unsigned char *image);

//! Get the record of a simulation's last finished episode
void harvest_env_last_episode(const void *env, int index, HarvestEpisodeRecord *record);

#ifdef __cplusplus
}
#endif

#endif
