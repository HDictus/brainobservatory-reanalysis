from pathlib import Path
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#OUT_DIR = Path(__file__).parent.parent / 'analysis_neuro/analyses/data/'

# TODO: is there a way to safely run allensdk functions alongside analysis-neuro?
data_directory = Path("/work/lnmc/visual_cortex/")  / 'data_store' / 'ecephys' # must be updated to a valid directory in your filesystem
data_directory.parent.mkdir(exist_ok=True)
data_directory.mkdir(exist_ok=True)
manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')

metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')
strict = metrics['isi_violations'] == 0
metrics = metrics[strict]

FS = metrics['waveform_duration'] < 0.5
metrics = metrics.assign(spiking_class=['FS' if at else 'RS' for at in FS])
metrics = metrics[metrics['ecephys_structure_acronym'] == 'VISp']


















