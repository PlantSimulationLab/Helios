/** \file "ParameterOptimization.cpp" Primary source file for the Parameter Optimization plug-in.

Copyright (C) 2016-2025 Brian Bailey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "ParameterOptimization.h"

using namespace std;
using namespace helios;

ParametersToOptimize readParametersFromFile(const std::string &filename) {
    ParametersToOptimize params;
    // Resolve file path using project-based resolution
    std::filesystem::path resolved_path = resolveProjectFile(filename);
    std::string resolved_filename = resolved_path.string();
    
    std::ifstream infile(resolved_filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open parameter file: " + filename);
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty())
            continue;

        std::istringstream ss(line);
        std::string name;
        float value, min, max;

        if (std::getline(ss, name, ',') && ss >> value && ss.ignore(1) && ss >> min && ss.ignore(1) && ss >> max) {
            params[name] = ParameterToOptimize{value, min, max};
        }
    }

    return params;
}


static ParameterOptimization::Result evaluatePopulation(const std::vector<ParametersToOptimize> &pop, const std::function<float(const ParametersToOptimize &)> &simulation) {

    ParameterOptimization::Result best;
    best.fitness = std::numeric_limits<float>::max();
    for (const auto &ind: pop) {
        float fit = simulation(ind);
        if (fit < best.fitness) {
            best.parameters = ind;
            best.fitness = fit;
        }
    }
    return best;
}

ParameterOptimization::Result ParameterOptimization::run(std::function<float(const ParametersToOptimize &)> simulation, const ParametersToOptimize &parameters, const OptimizationSettings &settings) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uni01(0.f, 1.f);

    std::ofstream outProgress;
    if (!settings.write_progress_to_file.empty()) {
        outProgress = std::ofstream(settings.write_progress_to_file);
        outProgress.clear();
    }


    size_t popsize = std::max<size_t>(2, settings.population_size);
    std::vector<ParametersToOptimize> population(popsize);
    for (auto &ind: population) {
        ind = parameters;
        for (auto &[name, param]: ind) {
            param.value = param.min + (param.max - param.min) * uni01(rng);
        }
    }

    if (!settings.read_input_from_file.empty()) {
        try {
            ParametersToOptimize best_from_file = readParametersFromFile(settings.read_input_from_file);
            population[0] = best_from_file; // seed with known best
        } catch (const std::exception &e) {
            std::cerr << "Warning: " << e.what() << "\n";
        }
    }

    Result best = evaluatePopulation(population, simulation);

    for (size_t g = 0; g < settings.generations; g++) {
        std::vector<ParametersToOptimize> new_pop(popsize);

        // Elitism, top 5%
        size_t elitism_count = std::max<size_t>(1, popsize * settings.elitism_rate);
        std::partial_sort(population.begin(), population.begin() + elitism_count, population.end(), [&](const auto &a, const auto &b) { return simulation(a) < simulation(b); });
        for (size_t e = 0; e < elitism_count; ++e) {
            new_pop[e] = population[e];
        }


        std::uniform_int_distribution<size_t> dist(0, popsize - 1);

        for (size_t i = 1; i < popsize; i++) {
            // Selection, Tournament
            auto tournamentSelect = [&](int k = 3) -> const ParametersToOptimize & {
                const ParametersToOptimize *bestSelection = &population[dist(rng)];
                for (int j = 1; j < k; ++j) {
                    const ParametersToOptimize *challenger = &population[dist(rng)];
                    if (simulation(*challenger) < simulation(*bestSelection)) {
                        bestSelection = challenger;
                    }
                }
                return *bestSelection;
            };

            const auto &parent1 = tournamentSelect();
            const auto &parent2 = tournamentSelect();

            ParametersToOptimize child = parameters;

            for (auto &[name, param]: child) {
                const auto &p1 = parent1.at(name);
                const auto &p2 = parent2.at(name);

                float val = p1.value;

                // Crossover, Blended alpha
                if (uni01(rng) < settings.crossover_rate) {
                    float alpha = 0.5f;
                    float minval = std::min(p1.value, p2.value);
                    float maxval = std::max(p1.value, p2.value);
                    float range = maxval - minval;
                    val = minval - alpha * range + uni01(rng) * (1.f + 2.f * alpha) * range;
                }

                // Mutation
                if (uni01(rng) < settings.mutation_rate) {
                    std::normal_distribution<float> gauss(0.f, 0.1f * (param.max - param.min));
                    val += gauss(rng);
                }


                param.value = std::min(std::max(val, param.min), param.max);
            }
            new_pop[i] = child;
        }

        population = std::move(new_pop);
        Result generation_best = evaluatePopulation(population, simulation);
        if (generation_best.fitness <= best.fitness) {
            best = generation_best;
            fitness_over_time.push_back(best.fitness);
            if (settings.print_progress) {
                std::cout << "Generation " << g << " - Best Fitness: " << best.fitness << "\n";
                for (const auto &[name, param]: best.parameters) {
                    std::cout << "  " << name << ": " << param.value << "\n";
                }
            }
            if (!settings.write_progress_to_file.empty()) {
                if (outProgress.is_open()) {
                    outProgress << "generation,fitness,parameter,value,min,max\n";
                    for (const auto &[name, param]: best.parameters) {
                        outProgress << g << "," << best.fitness << "," << name << "," << param.value << "," << param.min << "," << param.max << "\n";
                    }
                    if (message_flag) {
                        std::cout << "Best parameters progress written to: " << settings.write_progress_to_file << "\n";
                    }
                } else {
                    std::cerr << "Failed to write to file: " << settings.write_progress_to_file << "\n";
                }
            }
        }
    }

    if (settings.print_progress) {
        printf("Generation Fitness\n");
        for (int i = 0; i < fitness_over_time.size(); i++) {
            printf("%d %f\n", i, fitness_over_time.at(i));
        }
    }

    if (!settings.write_result_to_file.empty()) {
        std::ofstream out(settings.write_result_to_file);
        if (out.is_open()) {
            out << "parameter,value,min,max\n";
            for (const auto &[name, param]: best.parameters) {
                out << name << "," << param.value << "," << param.min << "," << param.max << "\n";
            }
            out.close();
            if (message_flag) {
                std::cout << "Best parameters written to: " << settings.write_result_to_file << "\n";
            }
        } else {
            std::cerr << "Failed to write to file: " << settings.write_result_to_file << "\n";
        }
    }

    if (!settings.write_progress_to_file.empty()) {
        outProgress.close();
    }


    return best;
}
