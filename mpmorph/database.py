from __future__ import division, print_function, unicode_literals, absolute_import

import json
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb
from monty.json import MontyEncoder
from pymatgen.core.trajectory import Trajectory
import numpy as np

logger = get_logger(__name__)

__author__ = 'Eric Sivonxay <esivonxay@lbl.gov>'


class VaspMDCalcDb(VaspCalcDb):
    """
    Adapted from atomate.vasp.database

    Class to help manage database insertions of Vasp drones
    """

    def __init__(self, host="localhost", port=27017, database="vasp", collection="tasks", user=None,
                 password=None, **kwargs):
        super(VaspMDCalcDb, self).__init__(host, port, database, collection, user, password, **kwargs)

    def insert_task(self, task_doc, parse_dos=False, parse_bs=False, md_structures=False):
        """
        Inserts a task document (e.g., as returned by Drone.assimilate()) into the database.
        Handles putting DOS and band structure into GridFS as needed.
        Args:
            task_doc: (dict) the task document
            parse_dos: (bool) attempt to parse dos in task_doc and insert into Gridfs
            parse_bs: (bool) attempt to parse bandstructure in task_doc and insert into Gridfs
        Returns:
            (int) - task_id of inserted document
        """

        # insert dos into GridFS
        if parse_dos and "calcs_reversed" in task_doc:
            if "dos" in task_doc["calcs_reversed"][0]:  # only store idx=0 DOS
                dos = json.dumps(task_doc["calcs_reversed"][0]["dos"], cls=MontyEncoder)
                gfs_id, compression_type = self.insert_gridfs(dos, "dos_fs")
                task_doc["calcs_reversed"][0]["dos_compression"] = compression_type
                task_doc["calcs_reversed"][0]["dos_fs_id"] = gfs_id
                del task_doc["calcs_reversed"][0]["dos"]

        # insert band structure into GridFS
        if parse_bs and "calcs_reversed" in task_doc:
            if "bandstructure" in task_doc["calcs_reversed"][0]:  # only store idx=0 BS
                bs = json.dumps(task_doc["calcs_reversed"][0]["bandstructure"], cls=MontyEncoder)
                gfs_id, compression_type = self.insert_gridfs(bs, "bandstructure_fs")
                task_doc["calcs_reversed"][0]["bandstructure_compression"] = compression_type
                task_doc["calcs_reversed"][0]["bandstructure_fs_id"] = gfs_id
                del task_doc["calcs_reversed"][0]["bandstructure"]

        # insert structures  at each ionic step into GridFS
        if md_structures and "calcs_reversed" in task_doc:
            # insert ionic steps into gridfs
            ## Convert from a list of dictionaries to a dictionary of lists
            ionic_steps_dict = task_doc["calcs_reversed"][0]['output']['ionic_steps']
            time_step = task_doc['input']['incar']['POTIM']
            trajectory = convert_ionic_steps_to_trajectory((ionic_steps_dict), time_step)
            del task_doc["calcs_reversed"][0]['output']['ionic_steps']

            traj_dict = json.dumps(trajectory, cls=MontyEncoder)
            gfs_id, compression_type = self.insert_gridfs(traj_dict, "trajectories_fs")

            task_doc['trajectory'] = {
                'formula_pretty': trajectory[0].composition.reduced_formula,
                'formula': trajectory[0].composition.formula.replace(' ', ''),
                'temperature': int(task_doc["input"]["incar"]["TEBEG"]),
                'compression': compression_type,
                'fs_id': gfs_id,
                'fs': 'trajectories_fs',
                'dimension': list(np.shape(trajectory.frac_coords)),
                'time_step': task_doc["input"]["incar"]["POTIM"],
                'frame_properties': list(trajectory.frame_properties[0].keys())
            }

        # insert the task document and return task_id
        return self.insert(task_doc)


def convert_ionic_steps_to_trajectory(ionic_steps_dict, time_step):
    read_site_props = False
    if ionic_steps_dict[0]['structure']['sites'][0].get('properties'):
        read_site_props = True
    lattice = ionic_steps_dict[0]['structure']['lattice']['matrix']
    species = [site['species'][0]['element'] for site in ionic_steps_dict[0]['structure']['sites']]

    frac_coords = []
    site_properties = []
    for ionic_step in ionic_steps_dict:
        frac_coords.append([site['abc'] for site in ionic_step['structure']['sites']])

        if read_site_props:
            _site_properties = {}
            for key in ionic_step['structure']['sites'][0]['properties']:
                _prop = [site['properties'][key] for site in ionic_step['structure']['sites']]
                _site_properties[key] = _prop
            site_properties.append(_site_properties)
        del ionic_step['structure']

    return Trajectory(lattice, species, frac_coords, site_properties=site_properties,
                      constant_lattice=True, frame_properties=ionic_steps_dict,
                      time_step=time_step)
