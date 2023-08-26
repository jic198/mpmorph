import json
import os
import re
import zlib
import gridfs
import numpy as np
from atomate.common.firetasks.glue_tasks import get_calc_loc
from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.drones import VaspDrone
from fireworks import explicit_serialize, FiretaskBase, FWAction
from fireworks.utilities.fw_serializers import DATETIME_HANDLER
from pymatgen.core.trajectory import Trajectory
from mpmorph.database import VaspMDCalcDb, insert_gridfs
from mpmorph.util import get_diffusivity

__author__ = 'Eric Sivonxay and Jianli Cheng'

logger = get_logger(__name__)


@explicit_serialize
class VaspMDToDb(FiretaskBase):
    """
    Enter a VASP run into the database. Uses current directory unless you
    specify calc_dir or calc_loc.
    Optional params:
        calc_dir (str): path to dir (on current filesystem) that contains VASP
            output files. Default: use current working directory.
        calc_loc (str OR bool): if True will set most recent calc_loc. If str
            search for the most recent calc_loc with the matching name
        parse_dos (bool): whether to parse the DOS and store in GridFS.
            Defaults to False.
        bandstructure_mode (str): Set to "uniform" for uniform band structure.
            Set to "line" for line mode. If not set, band structure will not
            be parsed.
        additional_fields (dict): dict of additional fields to add
        db_file (str): path to file containing the database credentials.
            Supports env_chk. Default: write data to JSON file.
        fw_spec_field (str): if set, will update the task doc with the contents
            of this key in the fw_spec.
        defuse_unsuccessful (bool): Defuses children fireworks if VASP run state
            is not "successful"; i.e. both electronic and ionic convergence are reached.
            Defaults to True.
    """
    optional_params = ["calc_dir", "calc_loc", "parse_dos", "bandstructure_mode",
                       "additional_fields", "db_file", "fw_spec_field",
                       "md_structures", "defuse_unsuccessful"]

    def run_task(self, fw_spec):
        # get the directory that contains the VASP dir to parse
        calc_dir = os.getcwd()
        if "calc_dir" in self:
            calc_dir = self["calc_dir"]
        elif self.get("calc_loc"):
            calc_dir = get_calc_loc(self["calc_loc"], fw_spec["calc_locs"])["path"]

        # parse the VASP directory
        logger.info("PARSING DIRECTORY: {}".format(calc_dir))

        drone = VaspDrone(additional_fields=self.get("additional_fields"),
                          parse_dos=self.get("parse_dos", False),
                          bandstructure_mode=self.get("bandstructure_mode", False))

        # assimilate (i.e., parse)
        task_doc = drone.assimilate(calc_dir)

        # Check for additional keys to set based on the fw_spec
        if self.get("fw_spec_field"):
            task_doc.update(fw_spec[self.get("fw_spec_field")])

        # get the database connection
        db_file = env_chk(self.get('db_file'), fw_spec)

        # db insertion or taskdoc dump
        if not db_file:
            with open("task.json", "w") as f:
                f.write(json.dumps(task_doc, default=DATETIME_HANDLER))
        else:
            mmdb = VaspMDCalcDb.from_db_file(db_file, admin=True)
            # prevent duplicate insertion
            task = mmdb.db.tasks.find_one_and_delete({
                'formula_pretty': task_doc['formula_pretty'],
                'task_label': task_doc['task_label']})
            if task and task.get('trajectory'):
                fs_id = task['trajectory']['fs_id']
                mmdb.db.trajectories_fs.files.delete_one({'_id': fs_id})
                mmdb.db.trajectories_fs.chunks.delete_many({'files_id': fs_id})
            t_id = mmdb.insert_task(task_doc,
                                    parse_dos=self.get("parse_dos", False),
                                    parse_bs=bool(self.get("bandstructure_mode", False)),
                                    md_structures=self.get("md_structures", True))
            logger.info("Finished parsing with task_id: {}".format(t_id))

        if self.get("defuse_unsuccessful", True):
            defuse_children = (task_doc["state"] != "successful")
        else:
            defuse_children = False

        return FWAction(stored_data={"task_id": task_doc.get("task_id", None)},
                        defuse_children=defuse_children)


@explicit_serialize
class TrajectoryDBTask(FiretaskBase):
    """
    Obtain all production runs and insert them into the db. This is done by
    searching for a unique tag
    """
    required_params = ["tag_id", "db_file"]
    optional_params = ['notes']

    def run_task(self, fw_spec):
        notes = self.get('notes', None)
        tag_id = self['tag_id']

        # get the database connection
        db_file = env_chk(self.get('db_file'), fw_spec)
        mmdb = VaspMDCalcDb.from_db_file(db_file, admin=True)
        traj = mmdb.db.trajectories.find_one_and_delete({"runs_label": tag_id})
        if traj:
            fs_id = traj['fs_id']
            mmdb.db.trajectories_fs.files.delete_one({'_id': fs_id})
            mmdb.db.trajectories_fs.chunks.delete_many({'files_id': fs_id})
        runs = mmdb.db['tasks'].find(
            {"task_label": re.compile(f'\d+_run.*{tag_id}')})
        runs_sorted = sorted(runs, key=lambda x: int(re.findall('run[_-](\d+)', x['task_label'])[0]))
        trajectory_doc = runs_to_trajectory_doc(runs_sorted, mmdb, tag_id, notes)
        mmdb.db.trajectories.insert_one(trajectory_doc)


@explicit_serialize
class DiffusionAnalysisTask(FiretaskBase):
    """
    Calculate ionic diffusivity and conductivity and insert them into the db. This is done by
    searching for a unique tag
    """
    required_params = ['tag_id', 'db_file', 'step_skip', 't_range']

    def run_task(self, fw_spec):
        step_skip = self.get('step_skip')
        t_range = self.get('t_range')
        db_file = env_chk(self.get('db_file'), fw_spec)
        mmdb = VaspMDCalcDb.from_db_file(db_file, admin=True)
        traj_doc = mmdb.db.trajectories.find_one({'runs_label': self.get('tag_id')})
        fs_id = traj_doc['fs_id']
        fs = gridfs.GridFS(mmdb.db, 'trajectories_fs')
        ionic_steps_json = zlib.decompress(fs.get(fs_id).read())
        ionic_steps_dict = json.loads(ionic_steps_json.decode())
        traj = Trajectory.from_dict(ionic_steps_dict)
        diffs = get_diffusivity(traj[:], step_skip, traj.time_step, t_range)
        mmdb.db.trajectories.update_one({'_id': traj_doc['_id']},
                                        {'$set': {'diffusivity': diffs}})
        

def runs_to_trajectory_doc(runs, mmdb, runs_label, notes=None):
    """
    Takes a list of task_documents, aggregates the trajectories from the ionics_steps gridfs storage, then dumps
    the pymatgen.core.Trajectory object into the 'trajectories_fs' collection and makes a dictionary doc to track
    the entry.

    :param runs: list of MD runs
    :param mmdb:
    :param runs_label: unique identifier to the runs
    :param notes: (optional) any notes or comments on the specific run
    :return:
    """
    trajectory = runs_to_trajectory_from_gfs(runs, mmdb)
    gfs_id, compression_type = insert_gridfs(trajectory.as_dict(), mmdb.db,
                                             "trajectories_fs")

    traj_doc = {
        'formula_pretty': trajectory[0].composition.reduced_formula,
        'formula': trajectory[0].composition.formula.replace(' ', ''),
        'temperature': int(runs[0]["input"]["incar"]["TEBEG"]),
        'runs_label': runs_label,
        'compression': compression_type,
        'fs_id': gfs_id,
        'fs': 'trajectories_fs',
        'step_fs_ids': [i["trajectory"]['fs_id'] for i in runs],
        'structure': trajectory[0].as_dict(),
        'dimension': list(np.shape(trajectory.frac_coords)),
        'time_step': runs[0]["input"]["incar"]["POTIM"] * 1e-3,
        'frame_properties': list(trajectory.frame_properties[0].keys()),
        'notes': notes
    }
    return traj_doc


def runs_to_trajectory_from_gfs(runs, mmdb):
    gfs_keys = [[run['trajectory']['fs_id'], 'trajectories_fs'] for run in runs]
    trajectory = None
    for i, (fs_id, fs) in enumerate(gfs_keys):
        # Load stored Trajectory
        print(fs_id, 'is stored in trajectories_fs')
        _trajectory = load_trajectory_gfs(fs_id=fs_id, db=mmdb.db, fs=fs)
        if trajectory is None:
            trajectory = _trajectory
        else:
            # Eliminate duplicate structure at the start of each trajectory
            # (since vasp will output the input structure)
            trajectory.extend(_trajectory[1:])
    return trajectory


def load_trajectory_gfs(fs_id, db, fs=None):
    if not fs:
        # Default to trajectories_fs
        fs = gridfs.GridFS(db, 'trajectories_fs')
    elif not isinstance(fs, gridfs.GridFS):
        # Handle fs supplied as str
        fs = gridfs.GridFS(db, fs)

    trajectories_json = zlib.decompress(fs.get(fs_id).read())
    trajectories_dict = json.loads(trajectories_json.decode())
    if isinstance(trajectories_dict, str):
        # Previous bug in mpmorph resulted in double serialization of dictionary.
        trajectories_dict = json.loads(trajectories_dict)
    try:
        trajectory = Trajectory.from_dict(trajectories_dict)
    except AssertionError:
        frame_properties = []
        for i in range(len(trajectories_dict['frac_coords'])):
            _properties = {}
            # forces were stored as an numpy object which serialized as a dict
            for key in trajectories_dict['frame_properties'].keys():
                if isinstance(trajectories_dict['frame_properties'][key], dict):
                    _properties[key] = trajectories_dict['frame_properties'][key]['data'][i]
                else:
                    _properties[key] = trajectories_dict['frame_properties'][key][i]
            frame_properties.append(_properties)
        trajectories_dict['frame_properties'] = frame_properties
        trajectory = Trajectory.from_dict(trajectories_dict)
    except TypeError:
        trajectories_dict['coords'] = trajectories_dict.pop('frac_coords')
        trajectory = Trajectory.from_dict(trajectories_dict)
    return trajectory
