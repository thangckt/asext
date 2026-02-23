#!/bin/bash
set -e

#####ANCHOR Info
# Script to build docs locally using Sphinx + MyST-Parser
# This is for testing and previewing the docs before pushing to GitHub Pages
# Run this script from the folder containing this script (i.e. _docs/)
# Install dependencies: pip install mystmd myst-parser sphinx sphinx-ext-mystmd

#####ANCHOR Parameters
conda_env=py13doc #
python_ver=3.13   # use 3.11 for centos6 compatibility, all other OS use 3.13
reset_env=0       # If 1, reset conda env

#####ANCHOR Prepare conda_env
echo -e "\nTASK: Prepare conda env ${conda_env}"

conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"

if ! conda info --envs | grep "${conda_env} " >/dev/null 2>&1; then
    conda create -y -n "${conda_env}" -c conda-forge python="${python_ver}"
    rebuild_elpa=1
else
    echo "Conda env ${conda_env} already existed"
    if [ ${reset_env} -eq 1 ]; then
        conda env remove -y -n "${conda_env}"
        conda clean -y --all
        conda create -y -n "${conda_env}" python="${python_ver}"
        rebuild_elpa=1
    fi
fi

conda activate "${conda_env}"
# conda install --update-specs -y -c conda-forge python=${python_ver}
# conda config --set solver classic  # libmamba vs. classic

pip install -r requirement.txt

echo -e "\nTASK: Check conda env"
echo "Python: $(python -c 'import sys;print(sys.version)')"
echo "Python path: $(which python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"

#####ANCHOR Build steps
rundir=$(pwd)

### 1. Build API docs using Sphinx's autodoc extension
rm -rf sphinx_doc/_apidoc sphinx_doc/_summary
sphinx-build -b myst sphinx_doc sphinx_doc/_apidoc

### 2. Build the full docs using MyST
rm -rf _build
# myst build --html

myst start

# or
# myst start &
# sleep 2
# xdg-open http://localhost:3000

cd "${rundir}"
