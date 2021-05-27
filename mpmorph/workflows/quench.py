import numpy as np
from atomate.vasp.fireworks.core import OptimizeFW
from fireworks import Workflow
from mpmorph.fireworks import powerups
from mpmorph.fireworks.core import StaticFW, MDFW
from mpmorph.util import recursive_update

__author__ = 'Eric Sivonxay and Muratahan Aykol'
__maintainer__ = 'Eric Sivonxay'
__email__ = 'esivonxay@lbl.gov'


def get_quench_wf(structures, temperatures=None, priority=None, quench_type="slow_quench",
                  cool_args=None, hold_args=None, quench_args=None, descriptor=None, **kwargs):
    fw_list = []
    temp = [3000, 500, 500] if temperatures is None else temperatures
    cool_args = {"md_params": {"nsteps": 200}} if cool_args is None else cool_args
    hold_args = {"md_params": {"nsteps": 500}} if hold_args is None else hold_args
    quench_args = {} if quench_args is None else quench_args
    descriptor = descriptor if descriptor else ''

    for (i, structure) in enumerate(structures):
        _fw_list = []
        if quench_type == "slow_quench":
            for t in np.arange(temp[0], temp[1], -temp[2]):
                # get fw for cool step
                use_prev_structure = False
                if len(_fw_list) > 0:
                    use_prev_structure = True
                _fw = get_MDFW(structure, t, t - temp[2], name=f'snap_{i}_cool_{t - temp[2]}',
                               args=cool_args, parents=[_fw_list[-1]] if len(_fw_list) > 0 else [],
                               priority=priority, previous_structure=use_prev_structure,
                               insert_db=True, **kwargs)
                _fw_list.append(_fw)
                # get fw for hold step
                _fw = get_MDFW(structure, t - temp[2], t - temp[2],
                               name=f'snap_{i}_hold_{t - temp[2]}',
                               args=hold_args, parents=[_fw_list[-1]], priority=priority,
                               previous_structure=True, insert_db=True, **kwargs)
                _fw_list.append(_fw)

        if quench_type in ["slow_quench", "mp_quench"]:
            # Relax OptimizeFW and StaticFW
            run_args = {"run_specs": {"vasp_input_set": None, "vasp_cmd": ">>vasp_cmd<<",
                                      "db_file": ">>db_file<<",
                                      "spec": {"_priority": priority}
                                      },
                        "optional_fw_params": {"override_default_vasp_params": {
                            'user_incar_settings': {'ISIF': 2}}}
                        }
            run_args = recursive_update(run_args, quench_args)
            _name = "snap_" + str(i)

            fw1 = OptimizeFW(structure=structure, name=_name + descriptor + "_optimize",
                             parents=[_fw_list[-1]] if len(_fw_list) > 0 else [],
                             **run_args["run_specs"], **run_args["optional_fw_params"],
                             max_force_threshold=None)
            if len(_fw_list) > 0:
                fw1 = powerups.add_cont_structure(fw1)
            fw1 = powerups.add_pass_structure(fw1)

            fw2 = StaticFW(structure=structure, name=_name + descriptor + "_static",
                           parents=[fw1], **run_args["run_specs"],
                           **run_args["optional_fw_params"])
            fw2 = powerups.add_cont_structure(fw2)
            fw2 = powerups.add_pass_structure(fw2)

            _fw_list.extend([fw1, fw2])

        fw_list.extend(_fw_list)

    name = structure.composition.reduced_formula + descriptor + "_quench"
    wf = Workflow(fw_list, name=name)
    return wf


def get_MDFW(structure, start_temp, end_temp, name="molecular dynamics", priority=None,
             args=None, **kwargs):
    run_args = {"md_params": {"nsteps": 500, "start_temp": start_temp, "end_temp": end_temp},
                "run_specs": {"vasp_input_set": None, "vasp_cmd": ">>vasp_cmd<<",
                              "db_file": ">>db_file<<"},
                "optional_fw_params": {"override_default_vasp_params": {}, "spec": {}}}

    run_args["optional_fw_params"]["override_default_vasp_params"].update(
        {'user_incar_settings': {'ISIF': 1, 'LWAVE': False, 'PREC': 'Low'}})
    run_args = recursive_update(run_args, args)
    _mdfw = MDFW(structure=structure, name=name, **run_args["md_params"],
                 **run_args["run_specs"], **run_args["optional_fw_params"], **kwargs)
    return _mdfw
