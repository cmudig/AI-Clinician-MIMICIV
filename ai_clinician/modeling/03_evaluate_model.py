import numpy as np
import pandas as pd
import tqdm
import argparse
import os
import shutil
import pickle
from ai_clinician.modeling.models.komorowski_model import *
from ai_clinician.modeling.models.common import *
from ai_clinician.modeling.columns import C_OUTCOME
from ai_clinician.preprocessing.utils import load_csv
from ai_clinician.preprocessing.columns import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tqdm.tqdm.pandas()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Evaluates an AI Clinician model on the MIMIC-IV test set.'))
    parser.add_argument('data', type=str,
                        help='Model data directory (should contain train and test directories)')
    parser.add_argument('model', type=str,
                        help='Path to pickle file containing the model')
    parser.add_argument('--out', dest='out_path', type=str, default=None,
                        help='Path to pickle file at which to write out results (optional)')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                        help='Decay for reward values (default 0.99)')
    parser.add_argument('--soften-factor', dest='soften_factor', type=float, default=0.01,
                        help='Amount by which to soften factors (random actions will be chosen this proportion of the time)')
    parser.add_argument('--num-iter-ql', dest='num_iter_ql', type=int, default=6,
                        help='Number of bootstrappings to use for TD learning (physician policy)')
    parser.add_argument('--num-iter-wis', dest='num_iter_wis', type=int, default=750,
                        help='Number of bootstrappings to use for WIS estimation (AI policy)')
    args = parser.parse_args()
    
    data_dir = args.data
    model = AIClinicianModel.load(args.model)
    assert model.metadata is not None, "Model missing metadata needed to generate actions"

    n_cluster_states = model.n_cluster_states
    n_actions = model.n_actions
    action_bins = model.metadata['actions']['action_bins']

    MIMICraw = load_csv(os.path.join(data_dir, "test", "MIMICraw.csv"))
    MIMICzs = load_csv(os.path.join(data_dir, "test", "MIMICzs.csv"))
    metadata = load_csv(os.path.join(data_dir, "test", "metadata.csv"))
    unique_icu_stays = metadata[C_ICUSTAYID].unique()
    
    # Bin vasopressor and fluid actions
    print("Create actions")    
    actions = transform_actions(
        MIMICraw[C_INPUT_STEP],
        MIMICraw[C_MAX_DOSE_VASO],
        action_bins
    )
    
    np.seterr(divide='ignore', invalid='ignore')
    
    blocs = metadata[C_BLOC].values
    stay_ids = metadata[C_ICUSTAYID].values
    outcomes = metadata[C_OUTCOME].values

    print("Evaluate on MIMIC test set")
    states = model.compute_states(MIMICzs.values)
    
    records = build_complete_record_sequences(
        metadata,
        states,
        actions,
        model.absorbing_states,
        model.rewards
    )
    
    test_bootql = evaluate_physician_policy_td(
        records,
        model.physician_policy,
        args.gamma,
        args.num_iter_ql,
        model.n_cluster_states
    )
    
    phys_probs = model.compute_physician_probabilities(states=states, actions=actions)
    model_probs = model.compute_probabilities(states=states, actions=actions)
    test_bootwis, _,  _ = evaluate_policy_wis(
        metadata,
        phys_probs,
        model_probs,
        model.rewards,
        args.gamma,
        args.num_iter_wis
    )

    model_stats = {}
    
    model_stats['test_bootql_0.95'] = np.quantile(test_bootql, 0.95)   #PHYSICIANS' 95# UB
    model_stats['test_bootql_mean'] = np.nanmean(test_bootql)
    model_stats['test_bootql_0.99'] = np.quantile(test_bootql, 0.99)
    model_stats['test_bootwis_mean'] = np.nanmean(test_bootwis)    
    model_stats['test_bootwis_0.01'] = np.quantile(test_bootwis, 0.01)  
    wis_95lb = np.quantile(test_bootwis, 0.05)  #AI 95# LB, we want this as high as possible
    model_stats['test_bootwis_0.05'] = wis_95lb
    model_stats['test_bootwis_0.95'] = np.quantile(test_bootwis, 0.95)
    print("Results:", model_stats)

    #region fig-test

    print('Drawing ql-wis histogram...')
    #fig1
    fig = plt.figure(figsize=(20, 10))
    test_bootql_tile = np.tile(test_bootql, [int(np.floor(len(test_bootwis)/ len(test_bootql))), 1]).ravel()

    counts, _, _ = np.histogram2d(test_bootql_tile, test_bootwis, bins=(np.arange(-105, 103, 2.5), np.arange(-105, 103, 2.5)))
    counts = counts.T
    counts = np.flipud(counts)

    offpolicy_eval_plot = fig.add_subplot(231)
    logcounts = np.log10(counts + 1)
    im = offpolicy_eval_plot.imshow(logcounts, cmap=plt.get_cmap('jet'))
    fig.colorbar(im, ax=offpolicy_eval_plot)
    offpolicy_eval_plot.set_xticks(np.arange(1, 90, 10))
    offpolicy_eval_plot.set_xticklabels(['-100','-75','-50','-25','0','25','50','75','100'])
    offpolicy_eval_plot.set_yticks(np.arange(1, 90, 10))
    offpolicy_eval_plot.set_yticklabels(['100', '75','50','25','0','-25','-50','-75','-100'])
    xmin, xmax, ymin, ymax = offpolicy_eval_plot.axis()
    offpolicy_eval_plot.plot([xmin, ymin], [xmax, ymax], 'r', linewidth=2)
    offpolicy_eval_plot.set_xlabel('Clinicans\' policy value')
    offpolicy_eval_plot.set_ylabel('AI policy value')

    print('Drawing clinician policy bar...')
    #fig2
    counts, _, _ = np.histogram2d(actions / len(action_bins[1]) + 1, actions % len(action_bins[1]) + 1, bins=(np.arange(1, 7), np.arange(1, 7)))
    counts /=  len(actions)
    # counts = np.flipud(counts)

    actual_action_plot = fig.add_subplot(232, projection='3d')
    _x = np.arange(5)
    _y = np.arange(5)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = counts.ravel()
    bottom = np.zeros_like(top)
    width = depth = 1
    actual_action_plot.bar3d(x, y, bottom, width, depth, top, shade=True)
    actual_action_plot.set_xticks(range(5))
    actual_action_plot.set_yticks(range(5))
    actual_action_plot.set_yticklabels(['>475', '173-475','44-173','0-44','0'])
    actual_action_plot.set_xticklabels(['0', '0.003-0.08','0.08-0.20','0.22-0.363','>0.363'])

    actual_action_plot.set_xlabel('Vasopressor dose')
    actual_action_plot.set_ylabel('     IV fluids dose')
    actual_action_plot.set_title('Clinicians\' policy')

    print('Drawing AI policy bar...')
    #fig3
    ai_action_plot = fig.add_subplot(233, projection='3d')
    optimal_action = model.Q.argmax(axis=1)
    OA1 = optimal_action + 1
    a = np.array([OA1, np.floor((OA1 - 0.0001) / 5) + 1, OA1 - np.floor(OA1 / 5) * 5])
    a[2, a[2, :] == 0] = 5
    med = a[[1, 2], :]

    counts, _, _ = np.histogram2d(med[0], med[1], bins=(np.arange(1, 7), np.arange(1, 7)))
    counts /=  len(med[0])
    # counts = np.flipud(counts)
    top = counts.ravel()
    ai_action_plot.bar3d(x, y, bottom, width, depth, top, shade=True)
    ai_action_plot.set_xticks(range(5))
    ai_action_plot.set_yticks(range(5))
    ai_action_plot.set_yticklabels(['>475', '173-475','44-173','0-44','0'])
    ai_action_plot.set_xticklabels(['0', '0.003-0.08','0.08-0.20','0.22-0.363','>0.363'])
    ai_action_plot.set_xlabel('Vasopressor dose')
    ai_action_plot.set_ylabel('     IV fluids dose')
    ai_action_plot.set_title('AI policy')

    #fig4

    iv_ticks = np.arange(-1250, 1251, 100)
    vp_ticks = np.arange(-1.05, 1.06, 0.1)

    nr_reps = 200
    p = np.unique(records.loc[:, C_ICUSTAYID])
    prop = min(10000 / len(p), 0.75)

    action_medians = model.metadata['actions']['action_medians']

    qldata = np.zeros([len(records.loc[records.loc[:, C_ACTION] != -1, :]), 4]) # TODO

    records_no_hidden = records.loc[records.loc[:, C_ACTION] != -1, :]

    qldata[:, :2] = records_no_hidden[[C_ICUSTAYID, C_OUTCOME]]
    qldata[:, 3] = MIMICraw[C_INPUT_STEP] - np.array(action_medians[0])[optimal_action[records_no_hidden[C_STATE].to_numpy()] // len(action_bins[1])]
    qldata[:, 2] = MIMICraw[C_MAX_DOSE_VASO] - np.array(action_medians[1])[optimal_action[records_no_hidden[C_STATE].to_numpy()] % len(action_bins[1])]
    
    r = pd.DataFrame(qldata)
    r.columns = ['id','morta','vaso','ivf']
    d = r.groupby('id').agg(['mean','median','sum'])

    groupCount = r.groupby('id').count()['morta'].to_numpy()
    d3 = np.array([d.morta['mean'].to_numpy(), d.vaso['mean'].to_numpy(), d.ivf['mean'].to_numpy(), d.vaso['median'].to_numpy(), d.ivf['median'].to_numpy(), d.ivf['sum'].to_numpy(), groupCount]).T

    iv_sample = np.zeros([len(iv_ticks) - 1, nr_reps, 2])
    vp_sample = np.zeros([len(vp_ticks) - 1, nr_reps, 2])

    print('Sampling and computing mean mortality for Intravenous Fluids...')
    for rep in tqdm.tqdm(range(nr_reps)):
        ii = np.floor(np.random.rand(len(p)) + prop)
        d4 = d3[ii == 1, :]

        iv_mean_sum = []
        for i in range(len(iv_ticks) - 1):
            ii = (d4[:, 4] >= iv_ticks[i]) & (d4[:, 4] <= iv_ticks[i + 1])
            if len(d4[ii, 0]) > 0:
                iv_mean_sum.append([iv_ticks[i], iv_ticks[i + 1], sum(ii), np.nanmean(d4[ii, 0]), np.nanstd(d4[ii, 0])])
            else:
                iv_mean_sum.append([iv_ticks[i], iv_ticks[i + 1], sum(ii), 0, 0])
        
        iv_mean_sum = np.array(iv_mean_sum)
        iv_sample[:, rep, 0] = iv_mean_sum[:, 3]
        iv_sample[:, rep, 1] = iv_mean_sum[:, 2]

    print('Sampling and computing mean mortality for Vasopressors...')
    for rep in tqdm.tqdm(range(nr_reps)):
        vp_mean_sum = []
        for i in range(len(vp_ticks) - 1):
            ii = (d4[:, 3] >= vp_ticks[i]) & (d4[:, 3] <= vp_ticks[i + 1])
            if len(d4[ii, 0]) > 0:
                vp_mean_sum.append([vp_ticks[i], vp_ticks[i + 1], sum(ii), np.nanmean(d4[ii, 0]), np.nanstd(d4[ii, 0])])
            else:
                vp_mean_sum.append([vp_ticks[i], vp_ticks[i + 1], sum(ii), 0, 0])

        vp_mean_sum = np.array(vp_mean_sum)
        
        vp_sample[:, rep, 0] = vp_mean_sum[:, 3]
        vp_sample[:, rep, 1] = vp_mean_sum[:, 2]
        
    iv_mean = np.nanmean(iv_sample[:, :, 0], axis=1)
    vp_mean = np.nanmean(vp_sample[:, :, 0], axis=1)

    def get_s(sample, ticks):
        sample_std = np.zeros(len(ticks) - 1)
        for i in tqdm.tqdm(range(len(ticks) - 1)):

            dies = np.ones(int(np.nansum(sample[i, :, 0] * sample[i, :, 1])))
            lives = np.zeros(int(np.nansum((1 - sample[i, :, 0]) * sample[i, :, 1])))
            sum = np.sqrt(np.nansum(sample[i, :, 1]))
            sample_std[i] = np.nanstd(np.concatenate((dies, lives)))

            sample_std[i] = sample_std[i] / sum
        return sample_std
    print('Computing std of mortality for Intravenous Fluids...')
    iv_sample_std = get_s(iv_sample, iv_ticks)
    print('Computing std of mortality for Vasopressor...')
    vp_sample_std = get_s(vp_sample, vp_ticks)

    print('Drawing intravenous fliuds (mortality-dose excess) line chart...')
    IV_dose_plot = fig.add_subplot(234)

    f = 10
    
    h = IV_dose_plot.plot(iv_mean, 'b', linewidth=1)
    IV_dose_plot.plot(iv_mean+f*iv_sample_std,'b:', linewidth=1)
    IV_dose_plot.plot(iv_mean-f*iv_sample_std,'b:', linewidth=1)

    
    IV_dose_plot.plot([len(iv_mean)/2, len(iv_mean)/2], [0, 1], 'k:')

    

    IV_dose_plot.set_xlim([1, len(iv_mean)])
    IV_dose_plot.set_ylim([0, 1])

    iv_ticks = iv_ticks.astype('float64')

    iv_ticks -= (iv_ticks[-1] - iv_ticks[-2]) / 2
    iv_ticks = np.round(iv_ticks,2)
    iv_ticks = iv_ticks[1:-1:2]

    IV_dose_plot.set_xticks(np.arange(0, 2*len(iv_ticks), 2))
    IV_dose_plot.set_xticklabels(iv_ticks.tolist())
    for tick in IV_dose_plot.get_xticklabels():
        tick.set_rotation(90)
    IV_dose_plot.set_xlabel('Average dose excess per patient')
    IV_dose_plot.set_ylabel('Mortality')

    IV_dose_plot.set_title('Intravenous fluids')

    print('Drawing vasopressor (mortality-dose excess) line chart...')
    vaso_dose_plot = fig.add_subplot(235)

    f = 10
    
    h = vaso_dose_plot.plot(vp_mean, 'b', linewidth=1)
    vaso_dose_plot.plot(vp_mean+f*vp_sample_std,'b:', linewidth=1)
    vaso_dose_plot.plot(vp_mean-f*vp_sample_std,'b:', linewidth=1)

    
    vaso_dose_plot.plot([len(vp_mean)/2, len(vp_mean)/2], [0, 1], 'k:')

    
    vaso_dose_plot.set_xlim([1, len(vp_mean)])
    vaso_dose_plot.set_ylim([0, 1])

    vp_ticks -= (vp_ticks[-1] - vp_ticks[-2]) / 2
    vp_ticks = np.round(vp_ticks,2)
    vp_ticks = vp_ticks[1:-1:2]

    vaso_dose_plot.set_xticks(np.arange(0, 2*len(vp_ticks), 2))
    vaso_dose_plot.set_xticklabels(vp_ticks.tolist())

    for tick in vaso_dose_plot.get_xticklabels():
        tick.set_rotation(90)
    vaso_dose_plot.set_xlabel('Average dose excess per patient')
    vaso_dose_plot.set_ylabel('Mortality')

    vaso_dose_plot.set_title('Vasopressors')

    plt.show()
    #endregion

    #TODO: test on other datset (besides MIMIC)
    
    if args.out_path is not None:
        with open(args.out_path, "wb") as file:
            pickle.dump(model_stats, file)
    print('Done.')