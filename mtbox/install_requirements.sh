#!/bin/bash
WHICH_PYTHON=$(which python)

echo "[!!!IMPORTANT!!!] Do you wish to install the requirements for mtbox in the below Python environment?"
echo $WHICH_PYTHON
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit 0;;
    esac
done

# Install requirements from Anaconda repository
conda install --yes --copy --file requirements_conda.txt
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

# Install requirements from PyPi repository
# --trusted-host option is added to bypass SSL verfication error
pip install --trusted-host=pypi.python.org -r requirements_pypi.txt
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

# Install requirements from Github repository
# --egg option is added to bypass an error when installing multicoretsne package
pip install --egg -r requirements_github.txt
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

exit 0
