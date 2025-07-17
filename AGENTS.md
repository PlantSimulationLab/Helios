## Code Style
- Code style should follow the style given in .clang-format.
- Standard C++ library include headers are listed in the file core/include/global.h. Check these includes to make sure you do not add unnecessary includes.

## Testing
- Do not try to run utilities/run_samples.sh. It will time out.
### Documentation
- If changes are made to docstrings or function signatures, build the documentation file `doc/Doxyfile` using doxygen. Some notes on this are given below:
- Run `doxygen doc/Doxyfile` to generate the documentation from the root directory not the `doc` directory.
- Check doxygen output for warnings. It is ok to ignore warnings of the form " warning: Member XXX is not documented." 
- The main concern when changing docstrings or function signatures is the potential for breaking \ref references, which produces a warning like:  warning: unable to resolve reference to XXX for \ref command".
    When fixing these, don't just remove the funciton signature or use markdown `` ticks to suppress the warning. It is important to keep the signature in the reference for overloaded functions/methods. Figure out why the function signature is not matching and thus causing Doxygen to treat it as plain text.
- You tend to flag hash symbols in code blocks as erroneous and propose to add a double hash (e.g., '##include "Context.h"'). This is not needed and ends up rendering the double hash.
### Test Coverage
- The script `utilitiesgenerate_coverage_report.sh` can be used to check test coverage. 
- It is recommended to use text-based output, which is achieved with the `-r text` option.