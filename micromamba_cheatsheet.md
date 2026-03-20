# Micromamba Cheat Sheet

Micromamba is a lightning-fast, C++ based, statically linked drop-in replacement for the Conda package manager.

## Environment Management

| Task | Command |
| :--- | :--- |
| **Create an environment** | `micromamba create -n myenv python=3.10` |
| **Create from a YAML file** | `micromamba create -f environment.yml` |
| **Activate environment** | `micromamba activate myenv` |
| **Deactivate environment** | `micromamba deactivate` |
| **List all environments** | `micromamba env list` |
| **Remove an environment** | `micromamba env remove -n myenv` |
| **Export to YAML** | `micromamba env export > environment.yml` |

---

## Package Management

| Task | Command |
| :--- | :--- |
| **Search for a package** | `micromamba search <package_name>` |
| **Install a package** | `micromamba install <package_name>` |
| **Install from specific channel** | `micromamba install -c conda-forge <package_name>` |
| **List installed packages** | `micromamba list` |
| **Update a specific package** | `micromamba update <package_name>` |
| **Update all packages** | `micromamba update --all` |
| **Remove a package** | `micromamba remove <package_name>` |

---

## System & Housekeeping

| Task | Command |
| :--- | :--- |
| **Initialize for your shell** | `micromamba shell init -s bash -p ~/micromamba` |
| **Run a command inside an env** | `micromamba run -n myenv <command>` |
| **Clean cache and tarballs** | `micromamba clean --all` |
| **View configuration info** | `micromamba info` |

> **Pro Tip:** Because Micromamba uses the exact same underlying architecture as Mamba and Conda, you can almost always swap out `conda` for `micromamba` in standard tutorials or documentation without breaking anything.