# Neural Horizon MPC with Acados

## Description

This is an open source project that implements neural horizon model predictive control with acados and compares it to the originally implemented CasADi versions, of the paper [Alsmeier, 2024](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10644452). Furthermore, it includes iterative node pruning with the Lottery Ticket Hypothesis and Finetuning. 

The advantage of this project is, that it improves the ability for real time model predictive control with acados and neural horizon. Furthermore, the resulting generated solver can be exported as C code.

The documentation on the results is in the file [Result_documentation.ipynb](Result_documentation.ipynb). However, to see the workflow of the neural horizon MPC generation with acados, one can also look atthe notebook [Tutorial_NH_AMPC.ipynb](Tutorial_NH_AMPC.ipynb). In addition there are different notebooks for seperatly creating multiple [datasets](Dataset_generation.ipynb), [networks](Network_generation.ipynb), [pruned networks](Pruned_Network_generation.ipynb), trajectories with [neural horizon MPC](Multi_NH.ipynb), [pruned neural horizon MPC](Multi_NH_prun.ipynb) and [plain acados MPC](Multi_AMPC.ipynb). Moreover, all *Show* or *Results_extraction* notebooks are are only there to create the figures. The Tutorial files [Tutorial_MPC_Only.ipynb](Tutorial_MPC_Only.ipynb) and [Tutorial_Neural_Horizon_CasADi.ipynb](Tutorial_Neural_Horizon_CasADi.ipynb) are there to generate the CasADi versions of the neural horizon. 

As is, the code does not provide the option to add valid reference states, only 0 is supported. Also, the FFNNs are restricted to a certain amount of parameters, because otherwise the acados implemented solvers crash. 


## Acados installation guide on Ubuntu or WSL

See  [Acados Installation](https://docs.acados.org/installation/index.html) or follow the steps bellow.

### Prerequisites

- Upgrade Ubuntu with:
    ```
    sudo apt-get update/upgrade
    ```

- Install c++ (CXX Compiler) with:
    ```
    sudo apt-get install g++
    ```

- Install git with:
    ```
    sudo apt install git-all
    ```

- Install cmake with:
    ```
    sudo apt install cmake
    ```


### Install acados

1. Clone acados and submodules:
    ```
    git clone https://github.com/acados/acados.git
    cd acados
    git submodule update --recursive --init
    ```


2. Change settings in CMakeLists.txt:
    ```
    code CMakeLists.txt
        > Change Line 71 (ACADOS_WITH_QPOASES) to ON
        > Change Line 72 (ACADOS_WITH_DAQP) to ON
        > Change Line 76 (ACADOS_WITH_QPDUNES) to ON
        > Change Line 77 (ACADOS_WITH_OSQP) to ON
        > Change Line 81 (ACADOS_PYTHON) to ON
    ```

3. Install acados with:
    ```
    mkdir -p build 
    cd build/ 
    cmake ..
    ```

    if this doesnt work run lines below and repeat: <br />
    ```
    cd ..
    sudo rm -r -f build/
    ```
    <br />
    and change in CMakeList.txt in Line 51 and 53 the “X64_AUTOMATIC” to your target found on <br />
    https://github.com/giaf/blasfeo/blob/master/README.md <br />
    under TARGET
<br />

4. Now make acados (in folder <acados_root>/build/, e.g. ~/acados/build/):
    ```
    make install -j4
    ```


### Install Python Environment

<acados_root> is the path where acados lies, e.g. ~/acados

1. Return to home with:
    ```
    cd
    ```

2. Make virtual environment acados_env:
    ```
    virtualenv <env_path> --python=/usr/bin/python3
    ```

3. Start env with:
    ```
    source <env_path>/bin/activate
    ```

4. Install acados via pip: 
    ```
    pip install -e <acados_root>/interfaces/acados_template
    ```


### Path adding terminal 

<full_acados_root> means e.g. /home/name/acados

1. add Paths:
    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<full_acados_root>/lib"
    export ACADOS_SOURCE_DIR="<full_acados_root>"
    ```

2. Restart Environment:
    ```
    source ~/.bashrc
    source <env_path>/bin/activate
    ```

Note: needs to be done every newly opened terminal


### Jupyter kernel:

1. activate venv:
    ```
    source <env_path>/bin/activate
    ```

2. install ipkernel:
    ```
    pip install --user ipykernel
    ```

3. install acados kernel:
    ```
    python -m ipykernel install --user --name=<kernel_name>
    ```

4. add paths to kernel:
    - go to printed directory and open kernel.json
    - add following line to the json
        ```
        "env": {"LD_LIBRARY_PATH": "<full_acados_root>/lib", "ACADOS_SOURCE_DIR": "<full_acados_root>"} 
        ```

## Acknowledgment
The Code in this repository was produced as part of the KI-Embedded project of the German Federal Ministry of Economic Affairs and Climate Action (BMWK).
The authors and maintainers acknowledge funding of the KI-Embedded project of the BMWK.

