/** \file "ParallelRobotSimulations.cpp" Many copies of robot task scenes, stepped together on worker threads.

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

#include "ParallelRobotSimulations.h"

#include <condition_variable>

using namespace helios;

// Threads kept waiting for work, which run a function over a range of indices together with the calling thread
class ParallelRobotSimulations::WorkerPool {
public:
    explicit WorkerPool(uint thread_count) {
        for (uint t = 1; t < thread_count; t++) {
            workers.emplace_back([this]() { workerLoop(); });
        }
    }

    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work_ready.notify_all();
        for (std::thread &worker: workers) {
            worker.join();
        }
    }

    [[nodiscard]] uint getThreadCount() const {
        return static_cast<uint>(workers.size() + 1);
    }

    void run(size_t count, const std::function<void(size_t)> &function) {
        if (running.exchange(true)) {
            helios_runtime_error("ERROR (ParallelRobotSimulations): A parallel run was started from within another, such as by calling forEachCopy() inside forEachCopy(). Do the inner work directly for the copy being handled.");
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            job = &function;
            job_count = count;
            next_index = 0;
            first_failure = nullptr;
            busy_workers = workers.size();
            generation++;
        }
        work_ready.notify_all();
        work();
        std::unique_lock<std::mutex> lock(mutex);
        work_done.wait(lock, [this]() { return busy_workers == 0; });
        job = nullptr;
        std::exception_ptr failure = first_failure;
        lock.unlock();
        running = false;
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

private:
    // Take indices until none are left. Exceptions cannot cross a thread boundary, so the first is kept and raised by run().
    void work() {
        for (size_t i = next_index++; i < job_count; i = next_index++) {
            try {
                (*job)(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!first_failure) {
                    first_failure = std::current_exception();
                }
            }
        }
    }

    void workerLoop() {
        size_t seen_generation = 0;
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            work_ready.wait(lock, [&]() { return stopping || generation != seen_generation; });
            if (stopping) {
                return;
            }
            seen_generation = generation;
            lock.unlock();
            work();
            lock.lock();
            if (--busy_workers == 0) {
                work_done.notify_all();
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable work_done;
    bool stopping = false;
    size_t generation = 0;
    size_t busy_workers = 0;
    const std::function<void(size_t)> *job = nullptr;
    size_t job_count = 0;
    std::atomic<size_t> next_index{0};
    std::atomic<bool> running{false};
    std::mutex failure_mutex;
    std::exception_ptr first_failure;
};

ParallelRobotSimulations::ParallelRobotSimulations(uint thread_count) {
    if (thread_count == 0) {
        thread_count = std::max(1u, std::thread::hardware_concurrency());
    }
    pool = std::make_unique<WorkerPool>(thread_count);
}

ParallelRobotSimulations::~ParallelRobotSimulations() = default;

uint ParallelRobotSimulations::addVariant(PhysicsEngine *engine, ControlLoop *control_loop, const std::vector<RobotAction *> &actions) {
    if (engine == nullptr || control_loop == nullptr) {
        helios_runtime_error("ERROR (ParallelRobotSimulations::addVariant): The PhysicsEngine and ControlLoop pointers must not be null.");
    }
    if (engine->getMuJoCoModel() == nullptr) {
        helios_runtime_error("ERROR (ParallelRobotSimulations::addVariant): The simulation has not been initialized. Initialize it and create its instances before adding it.");
    }
    uint variant_action_size = 0;
    for (RobotAction *action: actions) {
        if (action == nullptr) {
            helios_runtime_error("ERROR (ParallelRobotSimulations::addVariant): An action pointer is null.");
        }
        variant_action_size += action->getActionSize();
    }
    if (!variants.empty() && variant_action_size != action_size) {
        helios_runtime_error("ERROR (ParallelRobotSimulations::addVariant): The actions of this variant take " + std::to_string(variant_action_size) + " numbers, but those of the first variant take " + std::to_string(action_size) +
                             ". Every variant must take the same actions.");
    }
    action_size = variant_action_size;

    control_loop->prepareInstances();
    for (RobotAction *action: actions) {
        action->prepareInstances();
    }

    Variant variant{engine, control_loop, actions, static_cast<uint>(copy_variant.size()), engine->getInstanceCount()};
    for (uint instance = 0; instance < variant.copy_count; instance++) {
        copy_variant.push_back(static_cast<uint>(variants.size()));
        copy_instance.push_back(instance);
    }
    variants.push_back(variant);
    return static_cast<uint>(variants.size() - 1);
}

uint ParallelRobotSimulations::getCopyCount() const {
    return static_cast<uint>(copy_variant.size());
}

uint ParallelRobotSimulations::getVariantCount() const {
    return static_cast<uint>(variants.size());
}

uint ParallelRobotSimulations::getActionSize() const {
    return action_size;
}

uint ParallelRobotSimulations::getThreadCount() const {
    return pool->getThreadCount();
}

void ParallelRobotSimulations::requireCopy(uint copy, const std::string &method) const {
    if (copy >= copy_variant.size()) {
        helios_runtime_error("ERROR (ParallelRobotSimulations::" + method + "): Copy " + std::to_string(copy) + " does not exist. There are " + std::to_string(copy_variant.size()) + " copies.");
    }
}

uint ParallelRobotSimulations::getVariant(uint copy) const {
    requireCopy(copy, "getVariant");
    return copy_variant[copy];
}

uint ParallelRobotSimulations::getInstance(uint copy) const {
    requireCopy(copy, "getInstance");
    return copy_instance[copy];
}

PhysicsEngine *ParallelRobotSimulations::getPhysicsEngine(uint variant) const {
    if (variant >= variants.size()) {
        helios_runtime_error("ERROR (ParallelRobotSimulations::getPhysicsEngine): Variant " + std::to_string(variant) + " does not exist.");
    }
    return variants[variant].engine;
}

ControlLoop *ParallelRobotSimulations::getControlLoop(uint variant) const {
    if (variant >= variants.size()) {
        helios_runtime_error("ERROR (ParallelRobotSimulations::getControlLoop): Variant " + std::to_string(variant) + " does not exist.");
    }
    return variants[variant].control_loop;
}

void ParallelRobotSimulations::step(const float *actions, ControlStepResult *results) {
    if (actions == nullptr && action_size > 0) {
        helios_runtime_error("ERROR (ParallelRobotSimulations::step): The action array must not be null.");
    }
    forEachCopy([&](uint copy, uint variant, uint instance) {
        const Variant &scene = variants[variant];
        const float dt = scene.control_loop->getControlPeriod();
        const float *row = actions + size_t(copy) * action_size;
        for (RobotAction *action: scene.actions) {
            action->applyAction(row, dt, instance);
            row += action->getActionSize();
        }
        const ControlStepResult result = scene.control_loop->step(instance);
        if (results != nullptr) {
            results[copy] = result;
        }
    });
}

void ParallelRobotSimulations::resetCopy(uint copy) {
    requireCopy(copy, "resetCopy");
    const Variant &scene = variants[copy_variant[copy]];
    scene.control_loop->resetInstance(copy_instance[copy]);
    for (RobotAction *action: scene.actions) {
        action->resetInstance(copy_instance[copy]);
    }
}

void ParallelRobotSimulations::reset(const std::vector<uint> &copies) {
    for (uint copy: copies) {
        requireCopy(copy, "reset");
    }
    forEachIndex(copies.size(), [&](size_t i) { resetCopy(copies[i]); });
}

void ParallelRobotSimulations::forEachCopy(const std::function<void(uint copy, uint variant, uint instance)> &function) {
    forEachIndex(copy_variant.size(), [&](size_t copy) { function(static_cast<uint>(copy), copy_variant[copy], copy_instance[copy]); });
}

void ParallelRobotSimulations::forEachIndex(size_t count, const std::function<void(size_t index)> &function) {
    pool->run(count, function);
}

void ParallelRobotSimulations::gather(float *values, uint row_size, const std::function<void(uint copy, uint variant, uint instance, float *row)> &function) {
    forEachCopy([&](uint copy, uint variant, uint instance) { function(copy, variant, instance, values + size_t(copy) * row_size); });
}
