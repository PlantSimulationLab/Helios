#!/bin/bash

# Renders every plant model in the plant architecture library to an image, so that the whole library can be looked over
# after a change to the plant architecture plug-in. Each plant is shown at the last age in its life at which it is in full
# leaf and, if it bears them, carrying fruit or flowers. That is usually not its maximum age, at which a deciduous tree or
# vine is dormant and an annual may have shed its leaves.
#
# The program that does the rendering is utilities/render_plant_library.cpp. This script builds it in a temporary project
# under utilities/.scratch/, runs it, and removes the project again; only the images are left behind.

UTILPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

OUTPUT_DIR="${UTILPATH}/plant_library_renders"
IMAGE_WIDTH=1200
IMAGE_HEIGHT=1200
AGE_FRACTION=1
RANDOM_SEED=1
FINAL_AGE=0
TOP_VIEW=0
SPECIES=()
KEEP_PROJECT=0

usage() {
    echo "Usage: $0 [OPTIONS] [species ...]"
    echo ""
    echo "Renders plant models from the plant architecture library to one JPEG image per species. Each plant is grown through"
    echo "its whole life to find the last age at which it is in full leaf and bearing fruit (or flowers, if it never fruits),"
    echo "and is then grown again to that age and rendered. With no species named, every model in the library is rendered."
    echo "A table of each plant's age, size and organ counts is printed as it goes."
    echo ""
    echo "Options:"
    echo "  --output <dir>        Directory the images are written to (default: utilities/plant_library_renders)"
    echo "  --width <pixels>      Image width (default: ${IMAGE_WIDTH})"
    echo "  --height <pixels>     Image height (default: ${IMAGE_HEIGHT})"
    echo "  --age-fraction <f>    Consider only this fraction of each plant's maximum age, greater than 0 and at most 1"
    echo "                        (default: 1). Trees take minutes and several GB of memory at their maximum age."
    echo "  --final-age           Render each plant at the end of the life considered, whatever state it is in then,"
    echo "                        instead of looking for an age at which it is in leaf. Grows each plant once, not twice."
    echo "  --top-view            Also render each plant from directly above, to <species>_top.jpg"
    echo "  --seed <n>            Seed of the random number generator, so that a run can be repeated (default: ${RANDOM_SEED})"
    echo "  --keep-project        Do not remove the temporary build project when finished"
    echo "  --help                Show this message"
    echo ""
    echo "Examples:"
    echo "  $0                              Render the whole library"
    echo "  $0 bean cowpea tomato           Render three species"
    echo "  $0 --age-fraction 0.25 almond   Render a young almond tree"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --output)
            if [ "$#" -lt 2 ]; then echo "Error: --output needs a directory."; exit 1; fi
            OUTPUT_DIR="$2"; shift 2 ;;
        --width)
            if [ "$#" -lt 2 ]; then echo "Error: --width needs a number of pixels."; exit 1; fi
            IMAGE_WIDTH="$2"; shift 2 ;;
        --height)
            if [ "$#" -lt 2 ]; then echo "Error: --height needs a number of pixels."; exit 1; fi
            IMAGE_HEIGHT="$2"; shift 2 ;;
        --age-fraction)
            if [ "$#" -lt 2 ]; then echo "Error: --age-fraction needs a number."; exit 1; fi
            AGE_FRACTION="$2"; shift 2 ;;
        --final-age)
            FINAL_AGE=1; shift ;;
        --top-view)
            TOP_VIEW=1; shift ;;
        --seed)
            if [ "$#" -lt 2 ]; then echo "Error: --seed needs a number."; exit 1; fi
            RANDOM_SEED="$2"; shift 2 ;;
        --keep-project)
            KEEP_PROJECT=1; shift ;;
        --help|-h)
            usage; exit 0 ;;
        --*)
            echo "Error: Unknown option $1"; echo ""; usage; exit 1 ;;
        *)
            SPECIES+=("$1"); shift ;;
    esac
done

if ! mkdir -p "${OUTPUT_DIR}"; then
    echo "Error: Could not create the output directory ${OUTPUT_DIR}"
    exit 1
fi
OUTPUT_DIR="$( cd -- "${OUTPUT_DIR}" &> /dev/null && pwd )"

# A project of its own, named for this process so that two runs cannot collide
PROJECT_DIR="${UTILPATH}/.scratch/render_plant_library_$$"

cleanup() {
    if [ "${KEEP_PROJECT}" -eq 0 ]; then
        rm -rf "${PROJECT_DIR}"
    else
        echo "The build project was kept at ${PROJECT_DIR}"
    fi
}
trap cleanup EXIT

echo "Setting up the build project..."
# Each plug-in is its own argument: create_project.sh does not split a quoted list. It fails on a plug-in name it does not recognize.
if ! "${UTILPATH}/create_project.sh" "${PROJECT_DIR}" plantarchitecture visualizer > "${OUTPUT_DIR}/build.log" 2>&1; then
    echo "Error: Could not create the build project. See ${OUTPUT_DIR}/build.log"
    exit 1
fi
if ! cp "${UTILPATH}/render_plant_library.cpp" "${PROJECT_DIR}/main.cpp"; then
    echo "Error: Could not copy render_plant_library.cpp into the build project."
    exit 1
fi

echo "Building (the output is in ${OUTPUT_DIR}/build.log)..."
if ! cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" -DCMAKE_BUILD_TYPE=Release >> "${OUTPUT_DIR}/build.log" 2>&1; then
    echo "Error: CMake configuration failed. See ${OUTPUT_DIR}/build.log"
    exit 1
fi
if ! cmake --build "${PROJECT_DIR}/build" --config Release -j >> "${OUTPUT_DIR}/build.log" 2>&1; then
    echo "Error: The build failed. See ${OUTPUT_DIR}/build.log"
    exit 1
fi

# The executable takes its name from the project directory, and has to be run from its build directory to find its assets
EXECUTABLE_NAME="$(basename "${PROJECT_DIR}")"
if ! cd "${PROJECT_DIR}/build"; then
    echo "Error: Could not enter the build directory."
    exit 1
fi
EXECUTABLE=""
for CANDIDATE in "./${EXECUTABLE_NAME}" "./Release/${EXECUTABLE_NAME}.exe" "./${EXECUTABLE_NAME}.exe"; do
    if [ -x "${CANDIDATE}" ]; then
        EXECUTABLE="${CANDIDATE}"
        break
    fi
done
if [ -z "${EXECUTABLE}" ]; then
    echo "Error: The build did not produce an executable named ${EXECUTABLE_NAME}. See ${OUTPUT_DIR}/build.log"
    exit 1
fi

if ! "${EXECUTABLE}" "${OUTPUT_DIR}" "${IMAGE_WIDTH}" "${IMAGE_HEIGHT}" "${AGE_FRACTION}" "${RANDOM_SEED}" "${FINAL_AGE}" "${TOP_VIEW}" "${SPECIES[@]}"; then
    echo "Error: Rendering failed."
    exit 1
fi

echo "Images are in ${OUTPUT_DIR}"
