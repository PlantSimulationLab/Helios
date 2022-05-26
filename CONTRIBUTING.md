# Contributing to Helios

Thanks for contributing to [Helios](https://github.com/PlantSimulationLab/Helios)!

The following set of guidelines should be taken as suggestion, rather than rules.  These are currently a work-in-progress, and will evolve along with the project.

If you have comments or suggestions, please feel free to (open an issue)[https://github.com/PlantSimulationLab/Helios/issues/new] or [submit a pull-request] to this file.

## Submitting Changes

To submit a change, please [push local changes to a branch](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository), and [create a pull-request on github]((https://github.com/PlantSimulationLab/Helios/compare).  Pull requests should include a clear list of changes summarizing the overall contents of the change.

Each individual commits should be prefaced with a one-line summary of the intended change.  Large and complex commits may also include a longer description, separated by at least one line:

```
$ git commit -m "A brief summary of the commit.
> 
> A paragraph describing the change in greater detail."
```

## Style Guide

Code formatting for C++ sources is handled automatically by [clang-format](https://clang.llvm.org/docs/ClangFormat.html), which should be executed before pushing to `master`.  To do this, you can run the following from a Unix terminal:

```
git diff --name-only master -- '***.cpp' '***.cu' '***.h' | xargs clang-format -i
```

Alternatively, configure your CLion client to apply formatting on save or before commit.

In cases where the auto-formatter is inappropriate, you can disable formatting for a region of code by wrapping it with `// clang-format (off|on)` comments:

```
// clang-format off
... your_code_here ...
// clang-format on
```

Finally, note that formatting is disabled for external libraries contained in `./core/lib/...`.

