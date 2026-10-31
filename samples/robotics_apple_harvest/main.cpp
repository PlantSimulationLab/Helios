// Checks the apple harvesting environment without any learning: a scripted controller and random actions, with the speed of each.
// The scripted controller shows that the task can be done, and random actions show what success rate means nothing.
//
// Usage: ./robotics_apple_harvest [episodes_per_slot=3] [threads=4]
// Run from the build directory, so that the robot model is found at ../harvest_arm.urdf.

#include "harvest_env.h"

#include "global.h"

namespace {

    // Counts the first episodes of each simulation, rather than all episodes within a number of steps, which would favor outcomes that end episodes sooner
    void run(const std::string &name, void *env, int episodes_per_slot, bool scripted) {
        const int count = harvest_env_count(env);
        const int observation_size = harvest_env_observation_size();
        const int action_size = harvest_env_action_size();
        std::vector<float> observations(count * observation_size), terminal(count * observation_size), rewards(count), actions(count * action_size);
        std::vector<int> outcomes(count), finished(count, 0);
        std::mt19937 random(1);
        std::uniform_real_distribution<float> unit(-1.f, 1.f);

        std::map<int, int> totals;
        int episodes = 0;
        long steps = 0;
        harvest_env_reset(env, observations.data());
        const auto start = std::chrono::steady_clock::now();
        while (episodes < count * episodes_per_slot) {
            for (int i = 0; i < count; i++) {
                const float *o = observations.data() + i * observation_size;
                float *a = actions.data() + i * action_size;
                if (scripted) {
                    // Head for the fruit and grip it, and once gripped back straight away from the tree
                    const bool gripped = o[16] > 0.5f;
                    a[0] = gripped ? -1.f : 8.f * o[0];
                    a[1] = gripped ? 0.f : 8.f * o[1];
                    a[2] = gripped ? 0.f : 8.f * o[2];
                    a[3] = 1.f;
                } else {
                    for (int k = 0; k < action_size; k++) {
                        a[k] = unit(random);
                    }
                }
            }
            harvest_env_step(env, actions.data(), observations.data(), rewards.data(), outcomes.data(), terminal.data());
            steps += count;
            for (int i = 0; i < count; i++) {
                if (outcomes[i] != HARVEST_RUNNING && finished[i] < episodes_per_slot) {
                    finished[i]++;
                    totals[outcomes[i]]++;
                    episodes++;
                }
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("%-10s %5d episodes: %5.1f%% success, %5.1f%% timeout, %5.1f%% slipped, %5.1f%% knocked off, %5.1f%% collision | %7.0f control steps/s\n", name.c_str(), episodes, 100. * totals[HARVEST_SUCCESS] / episodes, 100. * totals[HARVEST_TIMEOUT] / episodes, 100. * totals[HARVEST_SLIPPED] / episodes,
                    100. * totals[HARVEST_KNOCKED_OFF] / episodes, 100. * totals[HARVEST_COLLISION] / episodes, double(steps) / seconds);
        std::fflush(stdout);
    }

} // namespace

int main(int argc, char **argv) {
    const int episodes_per_slot = argc > 1 ? std::stoi(argv[1]) : 3;
    const int threads = argc > 2 ? std::stoi(argv[2]) : 4;

    const std::vector<int> seeds = {1, 2, 3};
    const int instances_per_tree = 2;
    const auto start = std::chrono::steady_clock::now();
    void *env = harvest_env_create(seeds.data(), static_cast<int>(seeds.size()), instances_per_tree, threads, 0);
    if (env == nullptr) {
        return 1;
    }
    std::cout << seeds.size() << " trees, " << harvest_env_count(env) << " simulations, built in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s" << std::endl;

    run("scripted", env, episodes_per_slot, true);
    run("random", env, episodes_per_slot, false);

    harvest_env_destroy(env);
    return 0;
}
