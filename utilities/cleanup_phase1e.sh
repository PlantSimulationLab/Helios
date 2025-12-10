#!/usr/bin/env bash
# Phase 1.E: Clean up RadiationModel.h - Remove RT* declarations now in OptiX6Backend

set -e

HEADER="/home/bnbailey/CLionProjects/Helios_gpu_migration/plugins/radiation/include/RadiationModel.h"
BACKUP="${HEADER}.before_phase1e_cleanup"

echo "Phase 1.E RadiationModel.h cleanup"
echo "==================================="
echo ""
echo "Creating backup..."
cp "$HEADER" "$BACKUP"
echo "✓ Backup saved to: $BACKUP"
echo ""
echo "Removing duplicate RT* declarations and old methods..."

# Apply deletions in reverse order (bottom-to-top) to preserve line numbers
sed -i \
  -e '2570,2593d' \
  -e '2224,2546d' \
  -e '2026,2194d' \
  -e '1980,1982d' \
  -e '1893,1919d' \
  -e '1849,1881d' \
  "$HEADER"

echo "✓ Removed 579 lines:"
echo "  - Lines 1849-1881: RT* source variables (protected section)"
echo "  - Lines 1893-1919: RT* camera variables (protected section)"
echo "  - Lines 1980-1982: initializeOptiX() declaration"
echo "  - Lines 2026-2194: Buffer helper methods (getOptiXbufferData, etc.)"
echo "  - Lines 2224-2546: All RT* members (private section)"
echo "  - Lines 2570-2593: Error handlers (sutilHandleError, RT_CHECK_ERROR macros)"
echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Review changes: diff $BACKUP $HEADER | less"
echo "  2. Fix OptiX_Context references in RadiationModel.cpp"
echo "  3. Test compilation"
echo ""
echo "To restore backup: cp $BACKUP $HEADER"
