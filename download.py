from pathlib import Path
import pandas as pd 

output_dir = '.'
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.drifting_gratings import DriftingGratings
import allensdk.brain_observatory.stimulus_info as stim_info


# This class uses a 'manifest' to keep track of downloaded data and metadata.  
# All downloaded files will be stored relative to the directory holding the manifest
# file.  If 'manifest_file' is a relative path (as it is below), it will be 
# saved relative to your working directory.  It can also be an absolute path.
boc =  BrainObservatoryCache(
    manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))

# Download a list of all targeted areas
targeted_structures = boc.get_all_targeted_structures()
print("all targeted structures: " + str(targeted_structures))

data_set = boc.get_ophys_experiment_data(502376461)
dg = DriftingGratings(data_set)


outdfs = []
from tqdm import tqdm
if False:
    for container in tqdm(boc.get_ophys_experiments(stimuli=[stim_info.DRIFTING_GRATINGS])):
        #try:
        data_set = boc.get_ophys_experiment_data(container['id'])
        #except Exception:
        #    continue
        dg = DriftingGratings(data_set)
        meanswp = dg.mean_sweep_response
        for col in meanswp:
            if col == 'dx':
                continue
            outdfs.append(dg.stim_table.assign(
                cell_id=dg.cell_id[int(col)], 
                response=meanswp[col],
                cre_line=container['cre_line'].split('-')[0])
            )
    outdfs = pd.concat(outdfs)
    outdfs.to_parquet('all_responses.parquet')
outdfs = pd.read_parquet('all_responses.parquet')


def _shuffle(data):
    return data.groupby(['cell_id', 'temporal_frequency']).apply(
        lambda df: df.assign(response=df['response'][np.random.permutation(df.index)].values)).reset_index(drop=True)

shuffled = _shuffle(outdfs)
outdfs = outdfs.groupby(['cre_line', 'cell_id', 'temporal_frequency', 'orientation'])['response'].mean().reset_index()
shuffled = shuffled.groupby(['cre_line', 'cell_id', 'temporal_frequency', 'orientation'])['response'].mean().reset_index()

def _add_normalized(outdfs):
    minrates = outdfs.groupby('cell_id')['response'].min()
    #pref_tfs = outdfs.set_index('temporal_frequency').groupby('cell_id')['response'].idxmax()
    outdfs['response_normed'] = outdfs['response'] - minrates[outdfs['cell_id']].values
    return minrates

_add_normalized(outdfs)
_add_normalized(shuffled)


#opt = outdfs['temporal_frequency'] == pref_tfs.loc[outdfs['cell_id']].values

pref_tfs = outdfs.set_index('temporal_frequency').groupby('cell_id')['response'].idxmax()

import numpy as np
def g_OSI_signal(df, resp_col, ori_col, groupby, null_shuffle=False):
    df['scaling'] = np.exp(2 * 1j * np.deg2rad(df[ori_col]))
    df['scaled_resp'] = df[resp_col] * df['scaling']
    grouped = df.groupby(groupby)
    return (grouped['scaled_resp'].sum() / grouped[resp_col].sum()).abs()

def _osi(outdfs):
    osi_by_tf = g_OSI_signal(outdfs, 'response_normed', 'orientation', ['cell_id', 'cre_line', 'temporal_frequency']).reset_index()
    osi_by_tf['osi'] = osi_by_tf[0]
    del osi_by_tf[0]
    osi_by_tf = osi_by_tf[osi_by_tf['temporal_frequency'] > 0]
    return osi_by_tf.reset_index()

osi_by_tf = _osi(outdfs)
shosi_by_tf = _osi(shuffled).set_index(['cell_id', 'temporal_frequency'])['osi']


preferred = osi_by_tf[osi_by_tf['temporal_frequency'] == pref_tfs[osi_by_tf['cell_id']].values]

plt.clf()
plt.figure(figsize=(6.4, 4.8))
sns.lineplot(
    osi_by_tf[np.isin(osi_by_tf['cre_line'], ['Pvalb', 'Sst', 'Vip'])],
    x='temporal_frequency', 
    y='osi', 
    hue='cre_line', 
    errorbar=('ci', 95)
)
plt.xlabel('temporal frequency (Hz)')
plt.tight_layout()
plt.savefig('cre-osi-by-tf')
cosi_by_tf = osi_by_tf.set_index(['cell_id', 'temporal_frequency'])
cosi_by_tf['osi (shuffle-corrected)'] = cosi_by_tf['osi'] - shosi_by_tf
cosi_by_tf = cosi_by_tf.reset_index()
plt.clf()
plt.figure(figsize=(6.4, 4.8))
sns.lineplot(
    cosi_by_tf[np.isin(osi_by_tf['cre_line'], ['Pvalb', 'Sst', 'Vip'])],
    x='temporal_frequency', 
    y='osi (shuffle-corrected)', 
    hue='cre_line', 
    errorbar=('ci', 95)
)
plt.xlabel('temporal frequency (Hz)')
plt.tight_layout()
plt.savefig('cre-cosi-by-tf')

###############################################################
## Repeat this procedure without the minimum-rate normalization
# This will more closely resemble their original computation
################################################################

def _add_normalized(outdfs):
    minrates = outdfs.groupby('cell_id')['response'].min()
    #pref_tfs = outdfs.set_index('temporal_frequency').groupby('cell_id')['response'].idxmax()
    outdfs['response_normed'] = outdfs['response']
    outdfs.loc[outdfs['response_normed'] < 0, 'response_normed'] = 0
    return minrates

_add_normalized(outdfs)
_add_normalized(shuffled)


#opt = outdfs['temporal_frequency'] == pref_tfs.loc[outdfs['cell_id']].values



import numpy as np
def g_OSI_signal(df, resp_col, ori_col, groupby, null_shuffle=False):
    df['scaling'] = np.exp(2 * 1j * np.deg2rad(df[ori_col]))
    df['scaled_resp'] = df[resp_col] * df['scaling']
    grouped = df.groupby(groupby)
    return (grouped['scaled_resp'].sum() / grouped[resp_col].sum()).abs()

def _osi(outdfs):
    osi_by_tf = g_OSI_signal(outdfs, 'response_normed', 'orientation', ['cell_id', 'cre_line', 'temporal_frequency']).reset_index()
    osi_by_tf['osi'] = osi_by_tf[0]
    del osi_by_tf[0]
    osi_by_tf = osi_by_tf[osi_by_tf['temporal_frequency'] > 0]
    return osi_by_tf.reset_index()

osi_by_tf = _osi(outdfs)
shosi_by_tf = _osi(shuffled).set_index(['cell_id', 'temporal_frequency'])['osi']


preferred = osi_by_tf[osi_by_tf['temporal_frequency'] == pref_tfs[osi_by_tf['cell_id']].values]
df = pd.DataFrame(boc.get_cell_specimens())
df = df[~np.isnan(df['g_osi_dg'])]
df['tld1_name'] = [n.split('-')[0] for n in df['tld1_name']]
groups = df.groupby(['tld1_name'])
means = groups['g_osi_dg'].mean()
assert False
plt.clf()
plt.figure(figsize=(6.4, 4.8))
sns.lineplot(
    osi_by_tf[np.isin(osi_by_tf['cre_line'], ['Pvalb', 'Sst', 'Vip'])],
    x='temporal_frequency', 
    y='osi', 
    hue='cre_line', 
    errorbar=('ci', 95)
)
plt.xlabel('temporal frequency (Hz)')
plt.tight_layout()
plt.savefig('cre-osi-by-tf-rectified.png')
cosi_by_tf = osi_by_tf.set_index(['cell_id', 'temporal_frequency'])
cosi_by_tf['osi (shuffle-corrected)'] = cosi_by_tf['osi'] - shosi_by_tf
cosi_by_tf = cosi_by_tf.reset_index()
plt.clf()
plt.figure(figsize=(6.4, 4.8))
sns.lineplot(
    cosi_by_tf[np.isin(osi_by_tf['cre_line'], ['Pvalb', 'Sst', 'Vip'])],
    x='temporal_frequency', 
    y='osi (shuffle-corrected)', 
    hue='cre_line', 
    errorbar=('ci', 95),
    legend=False
)
plt.xlabel('temporal frequency (Hz)')
plt.tight_layout()
plt.savefig('cre-cosi-by-tf-rectified.png')

import numpy as np
df = pd.DataFrame(boc.get_cell_specimens())
df = df[~np.isnan(df['g_osi_dg'])]
groups = df.groupby(['tld1_name'])
from scipy import stats

diffs = []
for name1, data1 in tqdm(groups, total=groups.ngroups):
    for name2, data2 in groups:
        osi_sig = stats.ttest_ind(data1['g_osi_dg'], data2['g_osi_dg'])
        dsi_sig = stats.ttest_ind(data1['g_dsi_dg'], data2['g_dsi_dg'])
        diffs.append({
            'osi_p': osi_sig.pvalue,
            'dsi_p': dsi_sig.pvalue,
            'osi': data1['g_osi_dg'].mean() - data2['g_osi_dg'].mean(),
            'dsi': data1['g_dsi_dg'].mean() - data2['g_dsi_dg'].mean(),
            'first': name1.split('-')[0],
            'second': name2.split('-')[0]
        })
diffs = pd.DataFrame(diffs)

pvalues_osi = diffs.pivot_table(index='first', columns='second', values='osi_p')
pvalues_dsi = diffs.pivot_table(index='first', columns='second', values='dsi_p')
osi = diffs.pivot_table(index='first', columns='second', values='osi')
dsi = diffs.pivot_table(index='first', columns='second', values='dsi')

total_comparisons = len(np.tril(pvalues_osi.values).nonzero()[0])
# technically twice that, but it's a bit arbitrary anyway
threshold = 0.05 / total_comparisons

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

# 2fold
plt.figure(figsize=(4,4))
significant = pvalues_osi < threshold
hm = osi.copy()
#hm = np.log(pvalues_osi.copy())
hm[~significant] = np.nan
#hm *= np.sign(osi)
sns.heatmap(hm, cmap='bwr', cbar=False)
plt.tight_layout()
plt.savefig('osi_heatmap.png')
plt.clf()


plt.figure(figsize=(4,4))
significant = pvalues_dsi < threshold
hm = dsi.copy()
#hm = np.log(pvalues_dsi.copy())
hm[~significant] = np.nan
#hm *= np.sign(dsi)
sns.heatmap(hm, cmap='bwr', cbar=False)
plt.tight_layout()
plt.savefig('dsi_heatmap.png')
plt.clf()


df['cre_line'] = [name.split('-')[0] for name in df['tld1_name']]

#plt.figure()
f, a = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True, width_ratios=(3, 10))
inh = np.isin(df['cre_line'], ['Vip', 'Sst', 'Pvalb'])
sns.boxplot(df[inh], x='cre_line', y='g_osi_dg', ax=a[0], color='lightgray')
sns.boxplot(df[~inh], x='cre_line', y='g_osi_dg', ax=a[1], color='gray')
a[0].set_ylabel('OSI')
a[1].set_ylabel('')
a[0].set_xlabel('')
a[1].set_xlabel('')
#df['inh'] = inh
#sns.boxplot(df, x='cre_line', y='g_osi_dg', hue='inh')
plt.tight_layout()
plt.savefig('osi-boxplot.png')



#plt.figure()
f, a = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True, width_ratios=(3, 10))
inh = np.isin(df['cre_line'], ['Vip', 'Sst', 'Pvalb'])
sns.boxplot(df[inh], x='cre_line', y='g_dsi_dg', ax=a[0], color='lightgray')
sns.boxplot(df[~inh], x='cre_line', y='g_dsi_dg', ax=a[1], color='gray')
a[0].set_ylabel('DSI')
a[1].set_ylabel('')
a[0].set_xlabel('')
a[1].set_xlabel('')
#df['inh'] = inh
#sns.boxplot(df, x='cre_line', y='g_osi_dg', hue='inh')
plt.tight_layout()
plt.savefig('dsi-boxplot.png')