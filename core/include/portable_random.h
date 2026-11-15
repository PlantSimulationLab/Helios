/**
 * \file "portable_random.h" Random number distributions that give the same sequence on every platform
 *
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef HELIOS_PORTABLE_RANDOM
#define HELIOS_PORTABLE_RANDOM

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * The C++ standard fixes the output of its random number engines (std::minstd_rand0, std::mt19937, ...) but leaves the
 * distributions (std::uniform_real_distribution, std::normal_distribution, std::shuffle, ...) to each standard library. A model seeded
 * identically therefore produced a different plant, canopy or scan with libc++, libstdc++ and the Microsoft library.
 *
 * The distributions here take the place of the standard ones and are written out in full, so that the same engine with the same
 * seed gives the same sequence everywhere. They have the same interface as their std:: counterparts for the operations Helios uses.
 * The uniform distributions are exact on every platform. The others pass a uniform value through log, pow, sqrt or tan, evaluated
 * in double precision; a platform whose math library rounds one of those differently, or whose compiler fuses a multiplication and
 * an addition, can differ in the last bit of a double-precision result, which is below the resolution of the float results Helios uses.
 */
namespace helios {

    // Declared in global.h, which includes this file
    void helios_runtime_error(const std::string &error_message);

    namespace random_detail {

        //! Largest value a generator returns, less the smallest
        template<class Generator>
        constexpr uint64_t generatorSpan() {
            return uint64_t((Generator::max) ()) - uint64_t((Generator::min) ());
        }

        //! One draw of a generator, counted from zero
        template<class Generator>
        uint64_t draw(Generator &generator) {
            return uint64_t(generator()) - uint64_t((Generator::min) ());
        }

        //! Uniform integer in [0, range], by rejection so that every value is equally likely
        template<class Generator>
        uint64_t uniformInteger(Generator &generator, uint64_t range) {
            const uint64_t span = generatorSpan<Generator>();
            if (span > range) {
                // The generator gives more values than are wanted: divide them into equal groups and reject the remainder
                const uint64_t values = range + 1;
                const uint64_t group_size = span / values;
                const uint64_t limit = values * group_size;
                uint64_t value;
                do {
                    value = draw(generator);
                } while (value >= limit);
                return value / group_size;
            }
            // A generator that already returns every 64-bit value can never give fewer values than are wanted. The branch is left
            // out for such a generator at compile time, since the number of values per draw does not fit in 64 bits there.
            if constexpr (generatorSpan<Generator>() != (std::numeric_limits<uint64_t>::max) ()) {
                if (span < range) {
                    // The generator gives fewer values than are wanted: build the result from more than one draw
                    const uint64_t values_per_draw = span + 1;
                    uint64_t value;
                    uint64_t high_part;
                    do {
                        high_part = values_per_draw * uniformInteger(generator, range / values_per_draw);
                        value = high_part + draw(generator);
                    } while (value > range || value < high_part);
                    return value;
                }
            }
            return draw(generator);
        }

        //! Uniform value in [0,1) with as many random bits as the result type holds
        /**
         * Where one draw of the generator has enough bits, the draw is divided by the number of values the generator can return: a
         * single correctly rounded division, which is the same on every platform. Otherwise whole random bits are collected from
         * several draws into an integer, which is then scaled by a power of two. No step multiplies and adds floating-point numbers,
         * since a compiler may fuse those into one operation with different rounding.
         */
        template<class Real, class Generator>
        Real unitUniform(Generator &generator) {
            const uint64_t span = generatorSpan<Generator>();

            // Whole random bits in one draw: floor(log2(span + 1)), found with integer arithmetic so that it cannot differ by platform
            int bits_per_draw = 0;
            if (span == (std::numeric_limits<uint64_t>::max) ()) {
                bits_per_draw = 64;
            } else {
                for (uint64_t values = span + 1; values > 1; values >>= 1) {
                    bits_per_draw++;
                }
            }
            if (bits_per_draw == 0) {
                helios_runtime_error("ERROR (helios random distribution): The random number generator returns a single value, so nothing can be sampled from it.");
            }

            const int bits_needed = std::numeric_limits<Real>::digits;

            if (bits_per_draw >= bits_needed) {
                const double values_per_draw = double(span) + 1.0;
                Real result = Real(double(draw(generator)) / values_per_draw);
                if (result >= Real(1)) {
                    result = std::nextafter(Real(1), Real(0));
                }
                return result;
            }

            // The first draw supplies the most significant bits. The last draw supplies only as many of its high bits as are still needed.
            const uint64_t chunk_range = (uint64_t(1) << bits_per_draw) - 1;
            uint64_t bits = 0;
            int bits_collected = 0;
            while (bits_collected < bits_needed) {
                uint64_t chunk = uniformInteger(generator, chunk_range);
                int chunk_bits = bits_per_draw;
                if (bits_collected + chunk_bits > bits_needed) {
                    const int surplus = bits_collected + chunk_bits - bits_needed;
                    chunk >>= surplus;
                    chunk_bits -= surplus;
                }
                bits = (bits << chunk_bits) | chunk;
                bits_collected += chunk_bits;
            }
            // An integer below 2^bits_needed is held exactly, and scaling it by a power of two is exact, so the result is below 1
            return std::ldexp(Real(bits), -bits_needed);
        }

    } // namespace random_detail

    //! Uniformly distributed real number in [a,b). Takes the place of std::uniform_real_distribution.
    template<class Real = double>
    class uniform_real_distribution {
    public:
        typedef Real result_type;

        uniform_real_distribution() : lower(Real(0)), upper(Real(1)) {
        }
        explicit uniform_real_distribution(Real a, Real b = Real(1)) : lower(a), upper(b) {
        }

        template<class Generator>
        result_type operator()(Generator &generator) {
            return lower + (upper - lower) * random_detail::unitUniform<Real>(generator);
        }

        void reset() {
        }
        [[nodiscard]] result_type a() const {
            return lower;
        }
        [[nodiscard]] result_type b() const {
            return upper;
        }
        [[nodiscard]] result_type(min)() const {
            return lower;
        }
        [[nodiscard]] result_type(max)() const {
            return upper;
        }

    private:
        Real lower;
        Real upper;
    };

    //! Uniformly distributed integer in [a,b], both ends included. Takes the place of std::uniform_int_distribution.
    template<class Integer = int>
    class uniform_int_distribution {
    public:
        typedef Integer result_type;

        uniform_int_distribution() : lower(0), upper((std::numeric_limits<Integer>::max) ()) {
        }
        explicit uniform_int_distribution(Integer a, Integer b = (std::numeric_limits<Integer>::max) ()) : lower(a), upper(b) {
        }

        template<class Generator>
        result_type operator()(Generator &generator) {
            // Number of values wanted, less one. Unsigned arithmetic gives the right difference for a signed type as well.
            const uint64_t range = uint64_t(upper) - uint64_t(lower);
            const uint64_t offset = random_detail::uniformInteger(generator, range);
            return result_type(uint64_t(lower) + offset);
        }

        void reset() {
        }
        [[nodiscard]] result_type a() const {
            return lower;
        }
        [[nodiscard]] result_type b() const {
            return upper;
        }
        [[nodiscard]] result_type(min)() const {
            return lower;
        }
        [[nodiscard]] result_type(max)() const {
            return upper;
        }

    private:
        Integer lower;
        Integer upper;
    };

    //! Normally distributed real number. Takes the place of std::normal_distribution.
    /**
     * The Marsaglia polar method, which produces values in pairs; the second of each pair is kept for the next call.
     */
    template<class Real = double>
    class normal_distribution {
    public:
        typedef Real result_type;

        normal_distribution() : distribution_mean(Real(0)), distribution_stddev(Real(1)) {
        }
        explicit normal_distribution(Real mean, Real stddev = Real(1)) : distribution_mean(mean), distribution_stddev(stddev) {
        }

        template<class Generator>
        result_type operator()(Generator &generator) {
            double standard_normal;
            if (has_spare) {
                has_spare = false;
                standard_normal = spare;
            } else {
                double x, y, radius_squared;
                do {
                    x = 2.0 * random_detail::unitUniform<double>(generator) - 1.0;
                    y = 2.0 * random_detail::unitUniform<double>(generator) - 1.0;
                    radius_squared = x * x + y * y;
                } while (radius_squared > 1.0 || radius_squared == 0.0);
                const double multiplier = std::sqrt(-2.0 * std::log(radius_squared) / radius_squared);
                spare = x * multiplier;
                has_spare = true;
                standard_normal = y * multiplier;
            }
            return result_type(standard_normal * double(distribution_stddev) + double(distribution_mean));
        }

        //! Forget the value kept from the last pair, so that the next result depends only on the generator
        void reset() {
            has_spare = false;
        }
        [[nodiscard]] result_type mean() const {
            return distribution_mean;
        }
        [[nodiscard]] result_type stddev() const {
            return distribution_stddev;
        }

    private:
        Real distribution_mean;
        Real distribution_stddev;
        double spare = 0.0;
        bool has_spare = false;
    };

    //! Weibull-distributed real number with shape a and scale b. Takes the place of std::weibull_distribution.
    template<class Real = double>
    class weibull_distribution {
    public:
        typedef Real result_type;

        weibull_distribution() : shape(Real(1)), scale(Real(1)) {
        }
        explicit weibull_distribution(Real a, Real b = Real(1)) : shape(a), scale(b) {
        }

        template<class Generator>
        result_type operator()(Generator &generator) {
            const double uniform = random_detail::unitUniform<double>(generator);
            return result_type(double(scale) * std::pow(-std::log(1.0 - uniform), 1.0 / double(shape)));
        }

        void reset() {
        }
        [[nodiscard]] result_type a() const {
            return shape;
        }
        [[nodiscard]] result_type b() const {
            return scale;
        }

    private:
        Real shape;
        Real scale;
    };

    //! Gamma-distributed real number with shape alpha and scale beta. Takes the place of std::gamma_distribution.
    /**
     * The method of Marsaglia and Tsang (2000). A shape below one is sampled at shape + 1 and scaled back.
     */
    template<class Real = double>
    class gamma_distribution {
    public:
        typedef Real result_type;

        gamma_distribution() : shape(Real(1)), scale(Real(1)) {
        }
        explicit gamma_distribution(Real alpha, Real beta = Real(1)) : shape(alpha), scale(beta) {
        }

        template<class Generator>
        result_type operator()(Generator &generator) {
            const double alpha = double(shape);
            const double boosted_alpha = (alpha < 1.0) ? alpha + 1.0 : alpha;
            const double d = boosted_alpha - 1.0 / 3.0;
            const double c = 1.0 / std::sqrt(9.0 * d);

            double sample;
            while (true) {
                const double x = standard_normal(generator);
                double v = 1.0 + c * x;
                if (v <= 0.0) {
                    continue;
                }
                v = v * v * v;
                const double uniform = random_detail::unitUniform<double>(generator);
                const double x_squared = x * x;
                if (uniform < 1.0 - 0.0331 * x_squared * x_squared || (uniform > 0.0 && std::log(uniform) < 0.5 * x_squared + d * (1.0 - v + std::log(v)))) {
                    sample = d * v;
                    break;
                }
            }

            if (alpha < 1.0) {
                double uniform;
                do {
                    uniform = random_detail::unitUniform<double>(generator);
                } while (uniform == 0.0);
                sample *= std::pow(uniform, 1.0 / alpha);
            }
            return result_type(sample * double(scale));
        }

        void reset() {
            standard_normal.reset();
        }
        [[nodiscard]] result_type alpha() const {
            return shape;
        }
        [[nodiscard]] result_type beta() const {
            return scale;
        }

    private:
        Real shape;
        Real scale;
        normal_distribution<double> standard_normal;
    };

    //! Cauchy-distributed real number with location a and scale b. Takes the place of std::cauchy_distribution.
    template<class Real = double>
    class cauchy_distribution {
    public:
        typedef Real result_type;

        cauchy_distribution() : location(Real(0)), scale(Real(1)) {
        }
        explicit cauchy_distribution(Real a, Real b = Real(1)) : location(a), scale(b) {
        }

        template<class Generator>
        result_type operator()(Generator &generator) {
            // The tangent has a pole where the uniform value is exactly zero
            double uniform;
            do {
                uniform = random_detail::unitUniform<double>(generator);
            } while (uniform == 0.0);
            const double pi = 3.14159265358979323846;
            return result_type(double(location) + double(scale) * std::tan(pi * (uniform - 0.5)));
        }

        void reset() {
        }
        [[nodiscard]] result_type a() const {
            return location;
        }
        [[nodiscard]] result_type b() const {
            return scale;
        }

    private:
        Real location;
        Real scale;
    };

    //! Integer in [0,n) with the probability of each value in proportion to its weight. Takes the place of std::discrete_distribution.
    template<class Integer = int>
    class discrete_distribution {
    public:
        typedef Integer result_type;

        //! A single value, 0
        discrete_distribution() : cumulative_weights(1, 1.0) {
        }

        template<class InputIterator>
        discrete_distribution(InputIterator first, InputIterator last) {
            double total = 0.0;
            for (InputIterator weight = first; weight != last; ++weight) {
                if (!(double(*weight) >= 0.0)) {
                    helios_runtime_error("ERROR (helios::discrete_distribution): A weight is negative or not a number. Every weight must be zero or more.");
                }
                total += double(*weight);
                cumulative_weights.push_back(total);
            }
            if (cumulative_weights.empty()) {
                cumulative_weights.push_back(1.0);
            } else if (!(total > 0.0)) {
                helios_runtime_error("ERROR (helios::discrete_distribution): The weights sum to zero, so no value can be chosen. At least one weight must be positive.");
            }
        }

        template<class Generator>
        result_type operator()(Generator &generator) {
            const double target = random_detail::unitUniform<double>(generator) * cumulative_weights.back();
            // First value whose cumulative weight exceeds the target, so that a value of zero weight is never returned
            const auto position = std::upper_bound(cumulative_weights.begin(), cumulative_weights.end(), target);
            const size_t index = std::min(size_t(position - cumulative_weights.begin()), cumulative_weights.size() - 1);
            return result_type(index);
        }

        void reset() {
        }

    private:
        std::vector<double> cumulative_weights;
    };

    //! Put the elements of a range in random order. Takes the place of std::shuffle.
    /**
     * The Fisher-Yates shuffle, from the last element down.
     */
    template<class RandomIterator, class Generator>
    void shuffle(RandomIterator first, RandomIterator last, Generator &&generator) {
        typedef typename std::iterator_traits<RandomIterator>::difference_type difference_type;
        const difference_type count = last - first;
        for (difference_type i = count - 1; i > 0; i--) {
            uniform_int_distribution<uint64_t> pick(0, uint64_t(i));
            const difference_type j = difference_type(pick(generator));
            if (i != j) {
                using std::swap;
                swap(first[i], first[j]);
            }
        }
    }

} // namespace helios

#endif
