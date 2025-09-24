# Jobman documentation

`jobman` is a job management package designed to submit and monitor jobs on remote machines. It is built on the top of the [dpdispatcher](https://docs.deepmodeling.com/projects/dpdispatcher) package.

`jobman` is designed for the big data era, where the number of remoted jobs is large that handling them manually is almost impossible. Imaging that you have more than 1000 jobs to run, you have access to 3 remote high-performance computing (HPC) serves with different computing environment, and you need to monitor the progress of each job, check the output files, and download the results. This is a tedious and time-consuming task. The `jobman` package is designed to automate such tasks. `jobman` will handle the input files, submit the jobs to remote machines, monitor the progress of each job, and download the results to the local machine whenever jobs finished.


## Case 1: Distribute jobs to a single remote machine

This is a general purpose usage, where `task_list` can be defined flexibly. In this case, each `Task` can have different `command_list`, `forward_files`, `backward_files`.

To use, just need to:

- Define the `task_list` as list of [Task](https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/task.html) objects.
- Use function [submit_job_chunk()](#submit_job_chunk) to submit jobs to remote machines.

```python
from thkit.jobman.submit import submit_job_chunk, Task
from thkit.jobman.helper import loadconfig_multi_machines

mdict_list = loadconfig_multi_machines("MACHINE.yml")  # load the remote machine config
mdict = mdict_list[0]  # use the first machine in the list

task_list = [Task(), Task(), ...]    # list of Task objects
submit_job_chunk(
    mdict=mdict,
    work_dir='./',
    task_list=task_list,
    forward_common_files=forward_common_files,
    backward_common_files=backward_common_files,
)
```
That's it! `jobman` will handle the rest.


1. Info 1: An example of defining `task_list`:

```python
task = Task.load_from_dict(
    {
        "command": f"mpirun -np lmp_serial -in {runfile}",
        "task_work_path": "./",
        "forward_files": ["all_input_files"],
        "backward_files": ["all_getback_files/*"],
        "outlog": "out.log",
        "errlog": "err.log",
    }
)
task_list = [task]
```

2. Info 2: Configure the remote machine in `MACHINE.yml` file, following the [remote machine schema](https://thangckt.github.io/thkit_doc/schema_doc/config_remotes/).

3. (Optional) Use a launcher script (e.g., `launcher.sh`) to run python code

```bash
#!/bin/bash

source /etc/profile.d/modules.sh
module use /home/tha/app/1modulefiles
module load miniforge
source activate py13

python above_py_script.py
```


## Case 2: Distribute jobs to multiple remote machines

This is used for specific purpose (e.g., `alff` package), where the jobs have the same forward_files, backward_files; but the command_list can be different based on computing environment on each remote machine. Just need to:

- Prepare the `task_dirs`, where all of them have the same forward_files, backward_files.
- Define a `prepare_command_list()` function to prepare the command_list for each remote machine.

```python
from thkit.jobman.submit import alff_submit_job_multi_remotes
from thkit.config import loadconfig
import asyncio

mdict = loadconfig("remote_machine.yml")  # load the remote machine config

### Prepare command_list on each machine
def prepare_command_list(machine: dict) -> list:
    command_list = []
    dft_cmd = machine.get("command", "python")
    dft_cmd = f"{dft_cmd} ../cli_gpaw_optimize.py ../{FILE_ARG_ASE}"  # `../` to run file in common directory
    command_list.append(dft_cmd)
    return command_list

### Submit to multiple machines
asyncio.run(
    alff_submit_job_multi_remotes(
        multi_mdict=mdict,
        prepare_command_list=prepare_command_list,
        work_dir=work_dir,
        task_dirs=task_dirs,
        forward_files=forward_files,
        backward_files=backward_files,
        forward_common_files=forward_common_files,
        mdict_prefix="dft",
        logger=logger,
    )
)
```

Note:
    - Setting remote machines follow the [remote machine schema](https://thangckt.github.io/thkit_doc/schema_doc/config_remote_machine/).
    - Can import from `jobman` these classes: [Task](https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/task.html), [Machine](https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/machine.html), [Resources](https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/resources.html), [Submission](https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/api/dpdispatcher.html#dpdispatcher.Submission).
    - To handle if some tasks is finished and some tasks are not finished, see the function [handle_submission()](https://github.com/deepmodeling/dpdispatcher/blob/d55b3c3435a6b4cb8e200682a39a3418fd04d922/dpdispatcher/entrypoints/submission.py#L9)


API Reference:

::: thkit.jobman
