/** \file "ParallelRobotSimulations.h" Many copies of robot task scenes, stepped together on worker threads.

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

#ifndef HELIOS_PARALLEL_ROBOT_SIMULATIONS_H
#define HELIOS_PARALLEL_ROBOT_SIMULATIONS_H

#include "ControlLoop.h"

//! Many copies of the scenes of a robot task, commanded and stepped together, one control step at a time, on worker threads
/**
 * A task is trained on several scene variants, such as different trees, each a separate PhysicsEngine with its own ControlLoop and actions. Every simulation instance of a variant is a copy.
 * Copies are numbered in the order variants were added, the instances of each variant in order, so the copies of a variant are next to each other.
 *
 * step() takes the actions of every copy in one array, applies them, and runs a control step of every copy in parallel. Everything else a task does each control step, such as observations, rewards, and resets, can be run over all copies in parallel with forEachCopy().
 * The scenes are not owned: whatever built them must keep them alive. The threads are kept for the life of the object, so stepping does not start threads.
 */
class ParallelRobotSimulations {
public:
    //! Constructor
    /**
     * \param[in] thread_count Threads that run copies, counting the calling thread. Zero uses one per hardware thread.
     */
    explicit ParallelRobotSimulations(uint thread_count = 0);

    //! Destructor. Stops the worker threads.
    ~ParallelRobotSimulations();

    ParallelRobotSimulations(const ParallelRobotSimulations &) = delete;
    ParallelRobotSimulations &operator=(const ParallelRobotSimulations &) = delete;

    //! Add a scene variant, whose every simulation instance that exists now becomes a copy
    /**
     * The actions of a copy are read from its row of the action array in the order given here, each taking RobotAction::getActionSize() numbers. Every variant must take the same number in all.
     * The control loop's components are prepared for every instance (ControlLoop::prepareInstances()), and so are the actions.
     * \param[in] engine Initialized physics simulation of the variant, with its instances created.
     * \param[in] control_loop Control loop of the variant, stepping engine.
     * \param[in] actions Actions of the variant, in the order their numbers appear in a copy's row of the action array.
     * \return ID of the variant.
     */
    uint addVariant(PhysicsEngine *engine, ControlLoop *control_loop, const std::vector<RobotAction *> &actions);

    //! Get the number of copies, over all variants
    [[nodiscard]] uint getCopyCount() const;

    //! Get the number of variants
    [[nodiscard]] uint getVariantCount() const;

    //! Get the number of numbers in each copy's row of the action array
    [[nodiscard]] uint getActionSize() const;

    //! Get the number of threads that run copies, counting the calling thread
    [[nodiscard]] uint getThreadCount() const;

    //! Get the variant a copy belongs to
    /**
     * \param[in] copy Index of the copy.
     */
    [[nodiscard]] uint getVariant(uint copy) const;

    //! Get the simulation instance of its variant's PhysicsEngine that a copy is
    /**
     * \param[in] copy Index of the copy.
     */
    [[nodiscard]] uint getInstance(uint copy) const;

    //! Get the physics simulation of a variant
    /**
     * \param[in] variant ID of the variant.
     */
    [[nodiscard]] PhysicsEngine *getPhysicsEngine(uint variant) const;

    //! Get the control loop of a variant
    /**
     * \param[in] variant ID of the variant.
     */
    [[nodiscard]] ControlLoop *getControlLoop(uint variant) const;

    //! Apply the actions of every copy and run one control step of each, in parallel
    /**
     * \param[in] actions getCopyCount() rows of getActionSize() numbers.
     * \param[out] results What happened in the control step of each copy, getCopyCount() of them, or null if not needed.
     */
    void step(const float *actions, ControlStepResult *results = nullptr);

    //! Reset one copy with its control loop (ControlLoop::resetInstance()). Safe to call from within forEachCopy() for that copy.
    /**
     * \param[in] copy Index of the copy.
     */
    void resetCopy(uint copy);

    //! Reset several copies in parallel, as resetCopy() does
    /**
     * \param[in] copies Indices of the copies.
     */
    void reset(const std::vector<uint> &copies);

    //! Run a function for every copy, in parallel. Each copy is handled by one thread, so the function may change the state of the copy it is given, but not of any other.
    /**
     * If the function throws for any copy, the first error is raised once every copy has been handled.
     * \param[in] function Called with the index of the copy, its variant, and its instance.
     */
    void forEachCopy(const std::function<void(uint copy, uint variant, uint instance)> &function);

    //! Run a function for every index below a count, in parallel on the same threads, such as once per variant
    /**
     * \param[in] count Number of indices.
     * \param[in] function Called with each index.
     */
    void forEachIndex(size_t count, const std::function<void(size_t index)> &function);

    //! Fill an array with a row of numbers for every copy, computed in parallel
    /**
     * \param[out] values getCopyCount() rows of row_size numbers.
     * \param[in] row_size Numbers in each row.
     * \param[in] function Called with the index of the copy, its variant, its instance, and the start of its row to fill.
     */
    void gather(float *values, uint row_size, const std::function<void(uint copy, uint variant, uint instance, float *row)> &function);

private:
    struct Variant {
        PhysicsEngine *engine;
        ControlLoop *control_loop;
        std::vector<RobotAction *> actions;
        uint first_copy;
        uint copy_count;
    };

    void requireCopy(uint copy, const std::string &method) const;

    std::vector<Variant> variants;
    std::vector<uint> copy_variant; // per copy
    std::vector<uint> copy_instance; // per copy
    uint action_size = 0;

    class WorkerPool;
    std::unique_ptr<WorkerPool> pool;
};

#endif
