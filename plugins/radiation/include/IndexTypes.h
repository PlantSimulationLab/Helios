/** \file "IndexTypes.h" Type-safe indexing utilities for UUID ↔ Position conversion.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_INDEX_TYPES_H
#define HELIOS_INDEX_TYPES_H

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace helios {

    // ========== Strong Index Types ==========

    /**
     * @brief Primitive UUID - sparse identifier from Helios Context
     *
     * UUIDs are assigned by the Helios Context when primitives are created.
     * They are persistent, unique identifiers but may be non-sequential (e.g., 10, 42, 100, 5).
     *
     * **Cannot be implicitly converted to size_t**, preventing accidental use as array index.
     * Must explicitly convert via UUIDPositionMapper.
     *
     * @code
     * // Correct usage:
     * PrimitiveUUID uuid{42};
     * ArrayPosition pos = mapper.toPosition(uuid);
     * float value = buffer[pos];  // pos converts implicitly to size_t
     *
     * // Compile error:
     * float value = buffer[uuid];  // ERROR: PrimitiveUUID cannot convert to size_t
     * @endcode
     */
    struct PrimitiveUUID {
        uint value; ///< Raw UUID value from Helios Context

        /// Explicit construction from uint
        explicit constexpr PrimitiveUUID(uint v) : value(v) {
        }

        /// Explicit conversion to uint (prevents implicit array indexing)
        explicit operator uint() const {
            return value;
        }

        /// Equality comparison
        constexpr bool operator==(const PrimitiveUUID &other) const {
            return value == other.value;
        }
        constexpr bool operator!=(const PrimitiveUUID &other) const {
            return value != other.value;
        }
    };

    /**
     * @brief Array position - dense buffer index (0, 1, 2, ...)
     *
     * Array positions are sequential indices into GPU buffers. They are computed
     * during geometry building by reordering primitives by parent object for cache locality.
     *
     * **Allows implicit conversion to size_t** for convenient array indexing.
     *
     * @code
     * ArrayPosition pos{5};
     * float value = buffer[pos];  // Implicit conversion to size_t
     * @endcode
     */
    struct ArrayPosition {
        size_t value; ///< Array index value (0 to primitive_count-1)

        /// Explicit construction from size_t
        explicit constexpr ArrayPosition(size_t v) : value(v) {
        }

        /// Implicit conversion to size_t (enables array indexing)
        constexpr operator size_t() const {
            return value;
        }

        /// Equality comparison
        constexpr bool operator==(const ArrayPosition &other) const {
            return value == other.value;
        }
        constexpr bool operator!=(const ArrayPosition &other) const {
            return value != other.value;
        }
    };

    /// Special value indicating an invalid or not-found array position
    constexpr ArrayPosition INVALID_POSITION{SIZE_MAX};

    // ========== UUID ↔ Position Mapper ==========

    /**
     * @brief Bidirectional mapper between UUIDs and array positions
     *
     * Provides O(1) conversion in both directions:
     * - UUID → Position: Sparse array lookup `uuid_to_position_[UUID]`
     * - Position → UUID: Dense array lookup `position_to_uuid_[position]`
     *
     * The mapper is built once during geometry initialization and used throughout
     * the radiation model for all index conversions.
     *
     * **Thread-safe for read operations** after `build()` completes.
     *
     * @code
     * // Build mapper from ordered UUID list
     * std::vector<uint> primitive_uuids = {10, 42, 100, 5};  // Order defines positions
     * mapper.build(primitive_uuids);
     *
     * // Convert UUID → Position
     * PrimitiveUUID uuid{42};
     * ArrayPosition pos = mapper.toPosition(uuid);  // Returns ArrayPosition{1}
     *
     * // Convert Position → UUID
     * PrimitiveUUID uuid = mapper.toUUID(ArrayPosition{3});  // Returns PrimitiveUUID{5}
     * @endcode
     */
    class UUIDPositionMapper {
    private:
        std::vector<size_t> uuid_to_position_; ///< Sparse lookup: indexed by UUID, contains position
        std::vector<uint> position_to_uuid_; ///< Dense lookup: indexed by position, contains UUID

#ifndef NDEBUG
        bool debug_validation_enabled_ = true; ///< Enable runtime validation in debug builds
#endif

    public:
        /**
         * @brief Build mapping from ordered UUID list
         *
         * @param primitive_uuids Vector of UUIDs in array position order
         *                        (i.e., primitive_uuids[i] is the UUID at position i)
         *
         * This method:
         * 1. Stores the dense position→UUID mapping
         * 2. Builds a sparse UUID→position lookup table sized to max_UUID + 1
         * 3. Populates the sparse table with SIZE_MAX for non-existent UUIDs
         *
         * **Complexity**: O(n) where n = number of primitives
         */
        void build(const std::vector<uint> &primitive_uuids) {
            position_to_uuid_ = primitive_uuids;

            if (primitive_uuids.empty()) {
                uuid_to_position_.clear();
                return;
            }

            // Find max UUID to size sparse array
            uint max_uuid = *std::max_element(primitive_uuids.begin(), primitive_uuids.end());
            uuid_to_position_.assign(max_uuid + 1, SIZE_MAX);

            // Populate UUID → position mapping
            for (size_t pos = 0; pos < primitive_uuids.size(); ++pos) {
                uuid_to_position_[primitive_uuids[pos]] = pos;
            }
        }

        /**
         * @brief Convert UUID to array position
         *
         * @param uuid Primitive UUID
         * @return Array position, or INVALID_POSITION if UUID not found
         *
         * **Complexity**: O(1)
         *
         * @code
         * PrimitiveUUID uuid{42};
         * ArrayPosition pos = mapper.toPosition(uuid);
         * if (pos == INVALID_POSITION) {
         *     helios_runtime_error("UUID not found");
         * }
         * @endcode
         */
        ArrayPosition toPosition(PrimitiveUUID uuid) const {
            if (uuid.value >= uuid_to_position_.size()) {
                return INVALID_POSITION;
            }

            size_t pos = uuid_to_position_[uuid.value];
            if (pos == SIZE_MAX) {
                return INVALID_POSITION;
            }

            return ArrayPosition{pos};
        }

        /**
         * @brief Convert array position to UUID
         *
         * @param pos Array position
         * @return Primitive UUID
         * @throws std::out_of_range if position is out of bounds
         *
         * **Complexity**: O(1)
         *
         * @code
         * ArrayPosition pos{5};
         * PrimitiveUUID uuid = mapper.toUUID(pos);
         * @endcode
         */
        PrimitiveUUID toUUID(ArrayPosition pos) const {
            if (pos.value >= position_to_uuid_.size()) {
                throw std::out_of_range("ArrayPosition " + std::to_string(pos.value) + " out of range [0, " + std::to_string(position_to_uuid_.size()) + ")");
            }
            return PrimitiveUUID{position_to_uuid_[pos.value]};
        }

        /**
         * @brief Check if UUID exists in the mapping
         *
         * @param uuid Primitive UUID to check
         * @return true if UUID maps to a valid position
         *
         * @code
         * if (!mapper.isValidUUID(PrimitiveUUID{42})) {
         *     std::cerr << "UUID 42 not found in geometry\n";
         * }
         * @endcode
         */
        bool isValidUUID(PrimitiveUUID uuid) const {
            return toPosition(uuid) != INVALID_POSITION;
        }

        /**
         * @brief Get total number of primitives in the mapping
         *
         * @return Number of primitives (equivalent to position_to_uuid_.size())
         */
        size_t getPrimitiveCount() const {
            return position_to_uuid_.size();
        }

        /**
         * @brief Check if mapper is empty (no primitives)
         */
        bool empty() const {
            return position_to_uuid_.empty();
        }
    };

    // ========== Debug Validation (compiled out in release builds) ==========

#ifndef NDEBUG

    /**
     * @brief Runtime validation for indexing operations (debug builds only)
     *
     * Catches common indexing mistakes during development:
     * - Position out of bounds
     * - UUID used where position expected (heuristic warning)
     * - Invalid UUID
     *
     * **Compiled out in release builds** (zero runtime cost).
     *
     * Usage:
     * @code
     * IndexValidator validator(&mapper);
     * validator.validatePosition(pos.value, "myFunction");
     * @endcode
     */
    class IndexValidator {
    private:
        const UUIDPositionMapper *mapper_;

    public:
        /// Construct validator with reference to mapper
        explicit IndexValidator(const UUIDPositionMapper *mapper) : mapper_(mapper) {
        }

        /**
         * @brief Validate that a position is within bounds
         *
         * @param value Position value to validate
         * @param context Context string for error message (e.g., function name)
         * @throws std::runtime_error if position is out of range
         */
        void validatePosition(size_t value, const char *context) const {
            if (value >= mapper_->getPrimitiveCount()) {
                throw std::runtime_error(std::string(context) + ": Position " + std::to_string(value) + " out of range [0, " + std::to_string(mapper_->getPrimitiveCount()) + ")");
            }
        }

        /**
         * @brief Warn if value looks like UUID but used as position (heuristic)
         *
         * @param value Value to check
         * @param context Context string for warning message
         *
         * Heuristic: If value >= primitive_count but is a valid UUID, warn that
         * it might be a UUID/position mixup.
         */
        void warnPossibleMixup(size_t value, const char *context) const {
            if (value >= mapper_->getPrimitiveCount()) {
                PrimitiveUUID test_uuid{static_cast<uint>(value)};
                if (mapper_->isValidUUID(test_uuid)) {
                    std::cerr << "WARNING [" << context << "]: Value " << value << " looks like UUID but used as position\n";
                }
            }
        }
    };

/// Validate position is in bounds (debug only)
#define VALIDATE_POSITION(mapper, pos, ctx) IndexValidator(mapper).validatePosition(pos, ctx)

/// Warn if value might be UUID used as position (debug only)
#define WARN_UUID_MIXUP(mapper, val, ctx) IndexValidator(mapper).warnPossibleMixup(val, ctx)

#else
/// No-op in release builds
#define VALIDATE_POSITION(mapper, pos, ctx) ((void) 0)
/// No-op in release builds
#define WARN_UUID_MIXUP(mapper, val, ctx) ((void) 0)
#endif

} // namespace helios

#endif // HELIOS_INDEX_TYPES_H
