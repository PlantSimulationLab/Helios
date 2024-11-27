# Contributing to Helios

Thanks for contributing to [Helios](https://github.com/PlantSimulationLab/Helios)!

The following set of guidelines should be taken as suggestion, rather than rules.  These are currently a work-in-progress, and will evolve along with the project.

If you have comments or suggestions, please feel free to [open an issue](https://github.com/PlantSimulationLab/Helios/issues/new) or [submit a pull-request](https://github.com/PlantSimulationLab/Helios/compare) to this file.

## Editing Documentation

Documentation source files for the Helios core can be found in `doc/UserGuide.dox`, and for each plugin in the corresponding plug-in directory sub-folder `doc/*.dox`. These files are written in [Doxygen](https://www.doxygen.nl/index.html) format. Additionally, documentation for data structures, functions, methods, etc. is written directly in the corresponding header file, and also uses Doxygen syntax.

If you edit the documentation and want to view the changes, you can build the documentation HTML files. To do this, you need to have Doxygen, LaTeX and ghostscript installed on your system. For Linux:

```bash
sudo apt-get install doxygen texlive-full ghostscript
```

To build the documentation, navigate to the Helios root directory and run:

```bash
doxygen doc/Doxyfile
```

To view the built documentation, open the file `doc/html/index.html` in a web browser.

## Submitting Changes

To submit a change, please submit a pull request. There are two methods for doing this if you do not have a 'contributor' role in the Helios repo:

1. Fork the Helios repo, push changes to a branch, and [create a pull-request on github](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).  
2. You can use the GitHub web interface to [submit a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

Please make edits to the latest version of the master to the extent possible in order to allow for straight-forward merging of the changes into the master. Pull requests should include a clear list of changes summarizing the overall contents of the change.

Each individual commits should be prefaced with a one-line summary of the intended change.  Large and complex commits may also include a longer description, separated by at least one line:

```
$ git commit -m "A brief summary of the commit.
> 
> A paragraph describing the change in greater detail."
```

## Style Guide

Below are a few general guidelines to follow when contributing to Helios:
1. **Case**: Helios generally uses UpperCamelCase for class names, lowerCamelCase for method names, and snake_case for variable names.
2. **Const Ref**: Use const references (const &) for function/method arguments when passing non-scalar values.

