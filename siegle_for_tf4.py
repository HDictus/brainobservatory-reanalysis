import pandas as pd
from analysis_neuro import terminology as terms
from analysis_neuro.analyses.data import stimuli
import os
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# TODO: is there a way to safely run allensdk functions alongside analysis-neuro?
data_directory = Path("/work/lnmc/visual_cortex/")  / 'data_store' / 'ecephys' # must be updated to a valid directory in your filesystem
data_directory.parent.mkdir(exist_ok=True)
data_directory.mkdir(exist_ok=True)
manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')
strict = metrics['isi_violations'] == 0
metrics = metrics[strict]

FS = metrics['waveform_duration'] < 0.5
metrics = metrics.assign(spiking_class=['FS' if at else 'RS' for at in FS])


units = cache.get_units()

xyz = units[[
    'anterior_posterior_ccf_coordinate', 'dorsal_ventral_ccf_coordinate',
    'left_right_ccf_coordinate']
].values

from voxcell.nexus.voxelbrain import Atlas
import voxcell
atlas = Atlas.open("/work/lnmc/visual_cortex/VISp/.atlas")
rmap = atlas.load_region_map()
region_ids = atlas.load_data("brain_regions").lookup(xyz, outer_value=-1)
def try_get_acronym(region_id):
    try:
        return rmap.get(region_id, 'acronym')
    except voxcell.exceptions.VoxcellError:
        return '???'

acronyms = [try_get_acronym(region_id) for region_id in region_ids]

def _get_layer(acronym):
    if acronym[-1] in '123456':
        return f'L{acronym[-1]}'
    if acronym[-2] == '6':
        return f'L{acronym[-2:]}'
    return None


layers = [_get_layer(acronym) for acronym in acronyms]
units['layer'] = layers

experiment = metrics[['ecephys_structure_acronym', 'spiking_class', 'g_osi_dg', 'firing_rate', 'firing_rate_dg', 'firing_rate_ns']]
experiment['layer'] = units['layer']
units = units[units['isi_violations'] == 0]
units = units[units['snr'] > 2]
sessions = cache.get_session_table()
rate_data = []
shfrate_data = []
sponts_data = []
for session_id in tqdm(sessions[sessions['session_type'] == 'brain_observatory_1.1'].index):
    session = cache.get_session_data(session_id)

    session_units = session.units.index.intersection(units.index)
    gratings = session.get_stimulus_table(['drifting_gratings'])

    gratings_tf4 = gratings[gratings['temporal_frequency'] != 'null'] #[gratings['temporal_frequency'] == 4.0]
    nspikes = session.presentationwise_spike_counts([0., 2.0], gratings_tf4.index, session_units).to_dataframe()
    nspikes['rate'] = nspikes['spike_counts'] / 2.0
    nspikes['orientation'] = gratings_tf4.loc[nspikes.index.get_level_values(0), 'orientation'].values
    nspikes['tf'] = gratings_tf4.loc[nspikes.index.get_level_values(0), 'temporal_frequency'].values
    #meanrates = nspikes.reset_index().groupby(['unit_id', 'orientation'])['rate'].mean()
    rate_data.append(nspikes.reset_index()[['unit_id', 'orientation', 'tf', 'rate']])
    #rate_data.append(meanrates.reset_index())
    spont = session.presentationwise_spike_counts([0, 2.5], session.get_stimulus_table(['spontaneous']).index, session_units).to_dataframe()
    spont['rate'] = spont['spike_counts'] / 2.5
    sponts_data.append(spont.reset_index())
    meanshf = nspikes.reset_index().groupby('unit_id').apply(lambda df: df.assign(**{'rate': df.loc[np.random.permutation(df.index), 'rate'].values})).reset_index(drop=True).groupby(['unit_id', 'orientation'])['rate'].mean()
    shfrate_data.append(meanshf)

sponts_data = pd.concat(sponts_data)
pd.DataFrame({
    terms.FIRING_RATE: sponts_data['rate'],
    terms.CELL_ID: sponts_data['unit_id'],
    terms.LAYER: experiment['layer'].loc[sponts_data['unit_id']].values,
    terms.REGION: experiment['ecephys_structure_acronym'].loc[sponts_data['unit_id']].values,
    terms.SPIKING_CLASS: experiment['spiking_class'].loc[sponts_data['unit_id']].values,
    terms.DATASET: 'Siegle2021'
}).to_parquet('siegle_spont.parquet')
rate_data = pd.concat(rate_data)

pd.DataFrame({
    terms.FIRING_RATE: rate_data['rate'],
    terms.STIM_ORIENTATION: rate_data['orientation'],
    terms.CELL_ID: rate_data['unit_id'],
    terms.LAYER: experiment['layer'].loc[rate_data['unit_id']].values,
    terms.REGION: experiment['ecephys_structure_acronym'].loc[rate_data['unit_id']].values,
    terms.SPIKING_CLASS: experiment['spiking_class'].loc[rate_data['unit_id']].values,
    terms.DATASET: 'Siegle2021'
}).to_parquet('siegle_tf4_rates.parquet')

def _shuffle(data):
    return data.groupby(['unit_id', 'tf']).apply(
        lambda df: df.assign(rate=df['rate'][np.random.permutation(df.index)].values)).reset_index(drop=True)

#shfrate_data = pd.concat(shfrate_data).reset_index()
shfrate_data = _shuffle(rate_data)

def g_OSI_signal(df, resp_col, ori_col, groupby, null_shuffle=False):
    df['scaling'] = np.exp(2 * 1j * np.deg2rad(df[ori_col]))
    df['scaled_resp'] = df[resp_col] * df['scaling']
    grouped = df.groupby(groupby)
    return (grouped['scaled_resp'].sum() / grouped[resp_col].sum()).abs()


def _osi(rate_data):
    rate_data = rate_data.groupby(['unit_id', 'orientation', 'tf'])['rate'].mean().reset_index()
    pref_tf = rate_data.set_index('tf').groupby(['unit_id'])['rate'].idxmax()
    rate_data = rate_data[rate_data['tf'] == pref_tf[rate_data['unit_id']].values]
    minrates = rate_data.groupby('unit_id')['rate'].min()
    rate_data['relrate'] = rate_data['rate'] #- minrates[rate_data['unit_id']].values
    import analysis_neuro as an
    from analysis_neuro import features
    osi = g_OSI_signal(rate_data, 'relrate', 'orientation', ['unit_id'])
    return osi

osi = _osi(rate_data)
shosi = _osi(shfrate_data)
import matplotlib.pyplot as plt 
plt.clf()
plt.hist(osi.values)
plt.savefig('osi_tf4.png')
celldata = experiment.loc[osi.index]


outdata = pd.DataFrame({
    terms.CELL_ID: osi.index,
    terms.ORIENTATION_SELECTIVITY: osi.values,
    terms.LAYER: celldata['layer'].values,
    terms.SPIKING_CLASS: celldata['spiking_class'].values,
    terms.REGION: celldata['ecephys_structure_acronym'].values,
    terms.DATASET: 'Siegle2021'
})
outdata.to_parquet('siegle_tf4.parquet')

