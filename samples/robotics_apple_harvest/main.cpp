// Checks the apple harvesting environment without any learning: a scripted controller and random actions, with the speed of each.
// The scripted controller shows that the task can be done, and random actions show what success rate means nothing.
//
// Usage: ./robotics_apple_harvest [episodes_per_slot=3] [threads=4]
//        ./robotics_apple_harvest camera [directory for sample images]     (the speed of rendering images in the environment)
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
                    // Head for the fruit and grip it, and once gripped pull straight out of the cup's face, away from the fruit's center, so as not to peel the seal
                    const bool gripped = o[16] > 0.5f;
                    const float toward_fruit = std::max(std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]), 1e-6f);
                    for (int k = 0; k < 3; k++) {
                        a[k] = gripped ? -o[k] / toward_fruit : 8.f * o[k];
                    }
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

// The speed of stepping alone and of stepping with images, with the time split between them
void measureCamera(const std::vector<int> &seeds, int instances_per_tree, int threads, int size, const std::string &image_directory, bool leaves = false) {
    const auto start = std::chrono::steady_clock::now();
    void *env = harvest_env_create(seeds.data(), static_cast<int>(seeds.size()), instances_per_tree, threads, 0, leaves ? 1 : 0);
    if (env == nullptr || !harvest_env_enable_camera(env, size, size)) {
        std::exit(1);
    }
    const double build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const int count = harvest_env_count(env);
    const int observation_size = harvest_env_observation_size();
    const int action_size = harvest_env_action_size();
    std::vector<float> observations(count * observation_size), terminal(count * observation_size), rewards(count), actions(count * action_size);
    std::vector<int> outcomes(count);
    std::vector<unsigned char> images(size_t(count) * size_t(size) * size_t(size) * 3);
    harvest_env_reset(env, observations.data());

    auto seconds = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    double step_seconds = 0;
    double render_seconds = 0;
    const int steps = 120;
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < count; i++) {
            const float *o = observations.data() + i * observation_size;
            float *a = actions.data() + i * action_size;
            const bool gripped = o[16] > 0.5f;
            a[0] = gripped ? -1.f : 8.f * o[0];
            a[1] = gripped ? 0.f : 8.f * o[1];
            a[2] = gripped ? 0.f : 8.f * o[2];
            a[3] = 1.f;
        }
        auto t = std::chrono::steady_clock::now();
        harvest_env_step(env, actions.data(), observations.data(), rewards.data(), outcomes.data(), terminal.data());
        step_seconds += seconds(t);
        t = std::chrono::steady_clock::now();
        harvest_env_render(env, images.data());
        render_seconds += seconds(t);
        if (!image_directory.empty() && (s == 5 || s == 25 || s == 45)) {
            for (int i = 0; i < std::min(count, 3); i++) {
                const std::string name = image_directory + "/s" + std::to_string(s) + "_sim" + std::to_string(i) + ".jpeg";
                harvest_env_save_camera_image(env, i * instances_per_tree, name.c_str());
            }
        }
    }
    const double frames = double(steps * count);
    std::printf("%s%2zu trees x %d, %d threads, %3dx%3d images: built in %5.1f s | step %6.2f ms, render %6.2f ms per simulation | %6.0f steps/s with images, %6.0f without\n", leaves ? "leaves: " : "", seeds.size(), instances_per_tree, threads, size, size, build_seconds, 1000. * step_seconds / frames, 1000. * render_seconds / frames,
                frames / (step_seconds + render_seconds), frames / step_seconds);
    std::fflush(stdout);
    harvest_env_destroy(env);
}

int main(int argc, char **argv) {
    const std::vector<int> training_seeds = {1, 2, 3, 4, 5, 6, 7, 8};
    if (argc > 1 && std::string(argv[1]) == "camera") {
        const std::string image_directory = argc > 2 ? argv[2] : "";
        measureCamera({1}, 1, 1, 128, image_directory);
        measureCamera({1}, 4, 4, 128, "");
        measureCamera(training_seeds, 2, 6, 128, "");
        measureCamera(training_seeds, 2, 6, 64, "");
        measureCamera(training_seeds, 4, 6, 64, "");
        measureCamera(training_seeds, 2, 6, 64, image_directory, true);
        return 0;
    }

    const int episodes_per_slot = argc > 1 ? std::stoi(argv[1]) : 3;
    const int threads = argc > 2 ? std::stoi(argv[2]) : 4;

    const std::vector<int> seeds = {1, 2, 3};
    const int instances_per_tree = 2;
    const auto start = std::chrono::steady_clock::now();
    void *env = harvest_env_create(seeds.data(), static_cast<int>(seeds.size()), instances_per_tree, threads, 0, 0);
    if (env == nullptr) {
        return 1;
    }
    std::cout << seeds.size() << " trees, " << harvest_env_count(env) << " simulations, built in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s" << std::endl;

    run("scripted", env, episodes_per_slot, true);
    run("random", env, episodes_per_slot, false);

    harvest_env_destroy(env);
    return 0;
}
