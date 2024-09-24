import os 
import logging 
from collections.abc import Iterator
import numpy as np
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd
import dask_ml.model_selection as dcv
from matplotlib import pyplot as plt
import sklearn.metrics as skm
import sklearn.utils.random as skr
import scipy.stats as sps
from optuna.trial import TrialState
import torch
import torch.nn as nn
logger = logging.getLogger(__name__) 



def setup_datafiles():
    pass
    return 
    
    
def setup_logging(level="INFO"):
    logLevel = os.environ.get('LOG_LEVEL', 'INFO').upper()
    FORMAT = '%(asctime)s - %(levelname)s: - %(message)s'
    logging.basicConfig(level="INFO", format= FORMAT)
    logging.getLogger("imported_module").setLevel(logging.CRITICAL)

##-------------------------------------------------------------------------------------
## Load Profiles Program (3)
##-------------------------------------------------------------------------------------

def read_profiles_from_s3(sample_set, s3_path):
    for i, row in sample_set.iterrows():
        # print(f'row : \n{row}')
        print(f"Metadata {i}  Source:{row['Metadata_Source']}  Batch:{row['Metadata_Batch']}   Plate:{row['Metadata_Plate']}   Plate Type:{row['Metadata_PlateType']}")
        s3_path = profile_formatter.format(**row.to_dict())
        print(f"Path: {s3_path}")
        dframes.append(
            pd.read_parquet(s3_path, storage_options={"anon": True}, columns=columns)
        )
        temp_dframes = pd.concat(dframes)
        stats = get_stats(stats, temp_dframes)
        # break
    dframes = pd.concat(dframes) 


def read_profiles_from_local(root_folder, keys, source, columns):
    read_frames = []
    Metadata_Source, Metadata_Batch, Metadata_Plate = keys
    source_path = source.format(root_folder, Metadata_Source, Metadata_Batch, Metadata_Plate)
    df = pd.read_parquet(source_path, columns=columns)
    
    # print(len(df.columns.to_list()))
    return df

def cat_columns(column_list = None):
    from collections import defaultdict
    if isinstance(column_list, pd.DataFrame):
        print(f" input is dataframe shape: {column_list.shape}")
        column_list = column_list.columns.to_list()
    elif isinstance(column_list, list):
        print(f" input is list length {len(column_list)}")
    # elif isinstance(column_list, dict):
    #     column_list = list(column_list.keys())
    # len(column_list)
    
    keys = ["Cells", "Cytoplasm", "Nuclei", "Metadata", "Image_Granularity", 
            "Image_Intensity", "Image_Texture", "Image_Threshold", "Image_ImageQuality", "Texture",
            "Channel", "Count", "ExecutionTime", "FileName", "Frame", "Granularity", "MD5Digest", "Width", "Height",
            "Series", "ModuleError", "PathName", "Scaling", "Threshold", "URL"]
    
    key_len = [len(x) for x in keys]
    ttls = defaultdict(int)  ## {'Cells': 0, 'Cytoplasm': 0 , 'Nuclei' : 0, 'Metadata': 0, 'Other': 0}
    cols = defaultdict(list)
    
    for i in column_list:
        for j  in range(len(keys)):
            if i[:key_len[j]] == keys[j]:
                # print(f" found   {j}  - {keys[j]}  {key_len[j]}    {i[:key_len[j]]}")
                ttls[keys[j]] += 1
                cols[keys[j]].append(i)
                break
        else:
            cols['Other'].append(i)
            ttls['Other']+= 1
            # print(f" not found :   {i}")
    ttl_count = 0
    for k,v in ttls.items():
        print(f"    type: {k:20s}    count: {v:5d}")
        ttl_count += v
    print(f"\t\t\t\t  total: {ttl_count:5d}")
    return cols

def get_cols_like(col_list, pattern = None):
    import re
    if not isinstance(pattern, list):
        pattern = [pattern]
    result = []
    for pat in pattern:
        p = re.compile(pat)    
        result += [ i for i in col_list if p.search(i) is not None]
    return result 

def disp_stats(df, cols,header = True):
    if header:
        print(f"{' ':4s} {' ':60s}: {'min':>10s}   {'max':>10s}   {'std':>10s}   {'var':>10s}   {'mean':>10s}   {'median':>10s}    {'uniq rto':>10s}    {'freq rto':>10s}   {'quantile':^50s}")
    n_rows = df.shape[0]
    for ctr, i in enumerate(cols):
        print(f"{ctr:4d} {i[:60]:60s}: {df[i].min():10.2f}   {df[i].max():10.2f}   {df[i].std():10.2f}   {df[i].var():10.2f}"\
              f"   {df[i].mean():10.2f}   {df[i].median():10.2f}    {df[i].nunique()/n_rows:10.2f}    {calculate_frequency(df[i]):10.2f}     {np.quantile(df[i], [0.0,0.25, 0.50, 0.75,1.00] )}")

def check_values(df, cols):
    # if df is not None:
    #     print(f" dataframe shape: {df.shape}")
    #     cols = df.columns.to_list()
    # else:
    #     cols = column_list
    # len(col_list) 
    inv_columns = []
    n_rows = df.shape[0]
    print(f"\n Columns with invalid data from {len(cols)} columns " \
          f" {'min':>10s}    {'max':>10s}    {'std':>10s}    {'mean':>10s}    {'median':>10s}    {'quantile':^50s}")    
    for ctr, i in enumerate(cols):
        if (pd.isna(df[i].min()) or pd.isna(df[i].max()) or pd.isna(df[i].std()) 
           or df[i].max() == np.inf ): 
            inv_columns.append(i)
            print(f"{ctr:4d} {i[:45]:45s}: {df[i].min():10.2f}    {df[i].max():10.2f}    {df[i].std():10.2f}"\
                  f"    {df[i].mean():10.2f}    {df[i].median():10.2f}    {np.quantile(df[i], [0.0,0.25, 0.50, 0.75,1.00] )}")  
    if len(inv_columns) == 0:
        print(f" **** No Invalid Numeric Columns Found ****")
    return inv_columns

def check_df_for_nans(batch, group_id, rows_read, na_rows, na_columns, df_Nans, verbose = False): 

    for (row_id, row) in batch.iterrows():

        row_check = pd.isna(row)
        bad_columns_count = row_check.astype(int).sum()
        if  bad_columns_count == 0:
                if verbose:
                    print(f" VALID   |{group_id:4d} |{rows_read:6d} | {row_id:3d} |{na_rows:4d} | {row.Metadata_Source:9s} {row.Metadata_Batch[:20]:20s} {row.Metadata_Plate:20s}| {row.Metadata_JCP2022} |{row_check.sum():4d} |")
        else:
            na_rows += 1
            if  bad_columns_count < 1477:
                print(f" INVALID |{group_id:4d} |{rows_read:6d} | {row_id:3d} |{na_rows:4d} | {row.Metadata_Source:9s} {row.Metadata_Batch[:20]:20s} {row.Metadata_Plate:20s}| {row.Metadata_JCP2022} |{row_check.sum():4d} |{row.index[row_check].to_list()[:4]}")
                # print(f" {row.index[row_check].to_list()}")
                na_columns |= set(row.index[pd.isna(row)])
                df_Nans = pd.concat([df_Nans, row], axis = 1)
                if verbose:
                    print(row[check_cols].transpose())
            else:
                print(f" INVALID |{group_id:4d} |{rows_read:6d} | {row_id:3d} |{na_rows:4d} | {row.Metadata_Source:9s} {row.Metadata_Batch[:20]:20s} {row.Metadata_Plate:20s}| {row.Metadata_JCP2022} |{row_check.sum():4d} | all numeric columns ")
                df_Nans = pd.concat([df_Nans, row], axis = 1)
        rows_read += 1

    return rows_read, na_rows, na_columns, df_Nans


def reduce_col_sizes(df, cols):
    flt_cnt, int_cnt = 0, 0 
    for i in  cols:
        if df[i].dtype == 'float64':
            flt_cnt +=1
            df[i] = df[i].astype('float32')
            # print(f" {i} converted from float64 to {dframe[i].dtype}")
        elif df[i].dtype ==  np.dtype(object): 
            df[i] = df[i].astype(np.string_)
            print(f" {i} converted from object to np.string")
        elif df[i].dtype == 'int64':
            int_cnt += 1
            df[i] = df[i].astype('int32')
            print(f" {i} converted to int64 {dframe[i].dtype}")
    return df, flt_cnt, int_cnt

def calculate_frequency(feature_column):
    """Calculate frequency of second most common to most common feature.
    
    Parameters
    ----------
    feature_column : Pandas series of the specific feature in the population_df

    Returns
    -------
    Feature name if it passes threshold, "NA" otherwise
    """
    
    val_count = feature_column.value_counts()
    
    try:
        max_count = val_count.iloc[0]
    except IndexError:
        return np.nan
    
    try:
        second_max_count = val_count.iloc[1]
    except IndexError:
        return np.nan

    freq = second_max_count / max_count
    return freq

def disp_metadata_file(metadata, COMPOUND_PROFILE_COLUMNS):

    print("-"*80)
    print(" all_profile_columns & COMPOUND_PROFILE_COLUMNS")
    print("-"*80)
    print(f"  Len all_profile_columns        : {len(metadata['all_profile_columns'])}")
    print(f"  Len COMPOUND_PROFILE_COLUMNS   : {len(COMPOUND_PROFILE_COLUMNS)}\n")
    print(f"  all_profile_columns[:10]       : {metadata['all_profile_columns'][:10]}")
    print(f"  COMPOUND_PROFILE_COLUMNS[:10]  : {COMPOUND_PROFILE_COLUMNS[:10]}")
    print()
    print(f"  all_profile_columns[10:15]     : {metadata['all_profile_columns'][10:15]}")
    print(f"  COMPOUND_PROFILE_COLUMNS[10:15]: {COMPOUND_PROFILE_COLUMNS[10:15]}")
    print()
    print(f"  all_profile_columns[15:20]     : {metadata['all_profile_columns'][15:20]}")
    print(f"  COMPOUND_PROFILE_COLUMNS[15:20]: {COMPOUND_PROFILE_COLUMNS[15:20]}")
    print()
    print(f"  all_profile_columns[-4:]       : {metadata['all_profile_columns'][-4:]}")
    print(f"  COMPOUND_PROFILE_COLUMNS[-4:]  : {COMPOUND_PROFILE_COLUMNS[-4:]}")
    
    for k in metadata['metadata_columns'].keys():
        print("-"*80)
        print(f" {k}  - length({len(metadata['metadata_columns'][k])} )")
        print("-"*80)
        if isinstance(metadata['metadata_columns'][k], list):
            for v in metadata['metadata_columns'][k]:
                print(f" \t : list item : {v}")
    
        elif isinstance(metadata['metadata_columns'][k], dict):    
            for i,v in metadata['metadata_columns'][k].items():
                print(f" \t : key :  {i:25s}     item: {v}")
        print()
    
    for key in ["all_profile_columns", "metadata_columns", "selected_columns", "parquet_columns"]:
        print("\n"+"-"*80)
        print(f" {key}   {type(metadata[key])} {len(metadata[key]):5d} entries")
        print("-"*80)
        if isinstance(metadata[key], dict):
            ttl = 0 
            for j in metadata[key].keys():
                ttl += len(metadata[key][j])
                print(f" {'['+j+']':30s}   {len(metadata[key][j]):4d}       {type(metadata[key][j])}")  
                # print(f" key: {i:30s}/   {len(metadata[key][i]):5d}")
            print(f" \t\t\t  {'total:'} {ttl:5d}")     
        # else:
            # print(f" {key:20s}   {type(metadata[key])} {len(pickle_data[key]):5d} entries")
##-------------------------------------------------------------------------------------
## Training programs (4.1, 4.2, ...)
##-------------------------------------------------------------------------------------


def compute_classification_metrics(model, y_true, y_pred, top_k =3, mode = 'train'):
    metrics = {}
    if mode == 'train':
        metrics['train_auc'] = model['history']['train']['auc'][-1]
        metrics['train_logloss'] = model['history']['train']['logloss'][-1]
        metrics['val_auc']     = model['history']['test']['auc'][-1]
        metrics['val_logloss'] = model['history']['test']['logloss'][-1]
    else:
        if hasattr(model, "evals_result"):
            ev_result = model.evals_result()
            metrics['train_auc'] = ev_result['validation_0']['auc'][-1]
            metrics['train_logloss'] = ev_result['validation_0']['logloss'][-1]
            metrics['val_auc']  = ev_result['validation_1']['auc'][-1]
            metrics['val_logloss'] = ev_result['validation_0']['logloss'][-1]
        else:
            metrics['train_auc'] = 0.0
            metrics['train_logloss'] = 0.0
            metrics['val_auc']  = 0.0
            metrics['val_logloss'] = 0.0
            
    metrics['roc_auc']   = skm.roc_auc_score(y_true, y_score = y_pred)
    metrics['logloss']   = skm.log_loss(y_true, y_pred= y_pred)
    
    metrics['accuracy']  = skm.accuracy_score(y_true, y_pred= (y_pred >= 0.5))
    metrics['bal_acc']   = skm.balanced_accuracy_score(y_true, y_pred= (y_pred >= 0.5), adjusted = True)
    
    metrics['top_k_acc'] = skm.top_k_accuracy_score(y_true, y_score = y_pred, k=top_k)
    metrics['F1_score']  = skm.f1_score(y_true, y_pred = (y_pred >= 0.5))
 
    
    metrics['map']       = skm.average_precision_score(y_true, y_pred)
    metrics['pearson_corr'], pearson_p = sps.pearsonr(y_true, y_pred)
    
    return metrics    
    
def print_metric_hist(metrics):
    print("-" * 80)
    for key in metrics.keys():
        _metric_array = np.array(metrics[key])
        print(f" {key:20s}    {_metric_array.mean():9.5f} +/- {_metric_array.std():.5f}")
    print("-" * 80)    
 

def split_Xy(input, y_col = ["Metadata_log10TPSA"] ):
    if not isinstance(y_col,list):
        y_col = list(y_col)
    y_output = input[y_col]
    X_output = input.drop(columns=y_col)        
    return X_output, y_output


# def make_cv_splits(df_profiles, n_folds: int = 5, y_columns = None) -> Iterator[tuple[dd.DataFrame, dd.DataFrame]]:
#     if not isinstance(y_columns,list):
#         y_columns = list(y_columns)

#     frac = [1 / n_folds] * n_folds
#     # print(frac, n_folds)
#     splits = df_profiles.random_split(frac, shuffle=True)   
    
#     for i in range(n_folds):
#         train = [splits[j] for j in range(n_folds) if j != i]
#         train = dd.concat(train)
#         test = splits[i] 
#         X_train, y_train = split_Xy(train, y_columns)
#         X_test , y_test  = split_Xy(test, y_columns)
#         yield (X_train, y_train), (X_test, y_test)
        

def make_cv_splits_2(df_profiles, n_folds: int = 10, y_columns = None) -> Iterator[tuple[dd.DataFrame, dd.DataFrame]]:
    from itertools import chain
    if not isinstance(y_columns,list):
        y_columns = list(y_columns)
    
    num_files = len(df_profiles)
    assert num_files % n_folds == 0, f" num of # bin files {num_files} must be divisible by #folds {n_folds} "
    step_size = num_files // n_folds
    idx_list = list(range(0, num_files, step_size))
    print(f" # bin files: {num_files}   # of folds {n_folds} -  (groups of {num_files // n_folds} file tuples)")
    # print(list(idx_list))
    
    for i in idx_list :    
        trn_list = [np.arange(j, j+step_size) for j in idx_list if j != i]
        val_list = list(np.arange(i, i+step_size))
        trn_list = list(chain.from_iterable(trn_list))
        print(f"CV Split {i//step_size} -  Training files: {trn_list}   Validation files: {val_list}  ")
        train = dd.concat([df_profiles[j] for j in trn_list])
        X_train, y_train = split_Xy(train, y_columns)
        if len(val_list) == 0:
            X_test, y_test = None, None
        else:
            validation = dd.concat([df_profiles[j] for j in val_list])
            X_test , y_test  = split_Xy(validation, y_columns)
        yield (X_train, y_train), (X_test, y_test)
 

def get_dd_subset(df_ps, skiprows = 0, nrows=10, ss = None, verbose = False):
    if ss is None: 
        ss = df_ps.map_partitions(len).compute()
    ss_cumsum = ss.cumsum()
    ss_floorsum = ss.cumsum() - ss
    last_partition = ss_cumsum.index[-1]  
    _start_row = skiprows
    _end_row   = _start_row + nrows 

    if verbose:
        print(f" Skip {skiprows} rows then read {nrows} rows : from  row# {_start_row} to {_end_row}")
    assert skiprows < ss_cumsum[last_partition], f"Row skip ({skiprows}) is equal or larger than dataframe ({ss_cumsum[last_partition]})" 
    
    _start_row = _start_row if skiprows >0 else -1 
    st_idx = ss_floorsum[ss_floorsum.gt(_start_row)].index
    if verbose:
        print(f" st idx : {st_idx}")
 
    if len(st_idx) == 0 :
        print(f" No partitions satisfy skiprows = {skiprows}. Last partition begins at row {ss_floorsum.tail(1).item()}")
        return -1,-1
     
    st = st_idx.min() if len(st_idx) > 0 else 0       
    counter = 0
    en = st
 
    while counter < nrows  and en <= last_partition:
        counter += ss[en]
        print(f" Partition {en} (starting row: {ss_floorsum[en]}   ending row: {ss_cumsum[en]})  rows: {ss[en]}   count: {counter}")
        en +=1
    
    if verbose:
        print()
        print(f" Partition range: [{st}   {en}] ---- total rows included {ss[st:en].sum()}")
 
    en = en + 1 if en == st else en
    
    if verbose:
        print(f" ***** output (st,en) is ({st} , {en})")
        for i in range(st, en):
            print(f" {ss_floorsum[i]}   {ss_cumsum[i]}")
        print("\n\n")
    return st, en 


def read_binned_profile_files(file_idxs : set, filename : str = None  ,
                              names : list = [], usecols : list = [], 
                              dtype : dict = {}, **kwparms) -> list :
    file_list = []
    filenames = [filename.format(x) for x in file_idxs]
    logging.info(f" Read profiles file ...")
    
    for f in filenames:
        bin_file = read_cell_profiles( f , names = names, usecols = usecols, dtype = dtype, **kwparms)
        file_list.append(bin_file)
    
    # df_iter = map(call_read_cell_profiles, training_files)
    # input_file_list = [x for x in df_iter]
    logging.info(f" Read profiles file ... complete")
    return file_list


def read_cell_profiles(profile_file, rows = None, header=0, names = None, usecols=None, dtype =None, **kwparms ):
    print(f" Reading cell profiles file :  {profile_file}    {kwparms}")
    df_ps = dd.read_csv(profile_file, header=header, names = names, usecols = usecols, dtype = dtype)   
    # print(f" Number of partitions:  {df_ps.npartitions}   partition(1) shape: {df_ps.get_partition(0).shape}")
    
    # if skiprows is not None:
    #     print(f" Skipping {skiprows} rows")
    #     df_ps = df_ps.loc[skiprows:]
        
    if rows is not None:
        print(f" Limiting output to {rows} rows")
        df_ps = df_ps.head(npartitions = df_ps.npartitions, n=rows)        
        df_ps = dd.from_pandas(df_ps, npartitions = 100)       
        rows_str = f"{rows}"
    else:
        rows_str = "ALL"
    
    print(f" Reading {rows_str}  rows - Number of partitions:  {df_ps.npartitions}   ")
    print()    
    return df_ps

def read_cell_profiles_old(profile_file, rows = None, skiprows = None, index_cols = None):
    print(f" Reading cell profiles file :  {profile_file}")
    df_ps = dd.read_csv(profile_file, usecols=Xy_columns, dtype= Xy_columns_dtype)   
    print(f" Number of partitions:  {df_ps.npartitions}   partition(1) shape: {df_ps.get_partition(0).shape}")
    
    if skiprows is not None:
        print(f" skipping {skiprows} rows")
        df_ps = df_ps[skiprows:]
        
    if rows is not None:
        print(f" limiting output to {rows} rows")
        df_ps = df_ps.head(npartitions = df_ps.npartitions, n=rows)        
        df_ps = dd.from_pandas(df_ps, npartitions = 100)       
        rows_str = f"{rows}"
    else:
        rows_str = "ALL"
    print()    
    print(f" Reading {rows_str} rows into {type(df_ps)} shape: {df_ps.shape }")
    print(f" Number of partitions:  {df_ps.npartitions}   partition(1) shape: {df_ps.get_partition(0).compute().shape}")    
    return df_ps


def disp_trial_info(trial):
    print(f" Best trial:  {trial.number}   Trial state: {trial.state} ")
    print(f" start: {trial.datetime_start} , end:  {trial.datetime_complete}  duration: {trial.duration}")

    print(f" Intermediate values: {trial.intermediate_values}")
    print(f" Trial last step    : {trial.last_step} ")
    print()
    print(" Parameters: ")
    print("-------------")
    for key in trial.params:
        print(f"    {key:30s} {trial.params[key]:.7f}    {trial.distributions.get(key, 'n/a')}     ")
    print()
    print(f" Trial results: {trial.values}")
    print('\n')


def disp_study_history(study, best_only = False):
    print(f" {study.study_name}  study history\n")
    print(f" Total trials in study: {len(study.trials)}")
    best_trials = [x.number for x in study.best_trials]
    print(f" Best trials: {best_trials}" )    
    print(f"                start     -   completion      status        validation metrics")
    print(f" trial#         time      -      time          code      {study.metric_names[0]}        {study.metric_names[1]}")
    print("-"*80)
    for st in study.trials:
        if  (not best_only) or (best_only and st.number in best_trials):
            dt_start = st.datetime_start.strftime('%Y-%m-%d   %H:%M:%S') if st.datetime_start is not None else '-- n/a --' 
            dt_end   = st.datetime_complete.strftime('%H:%M:%S') if st.datetime_complete is not None else ' -n/a- ' 
            print(f"Trial #: {st.number:<4d} {dt_start:^21s} - {dt_end:^8s}  {st.state:3d}  ", end="")
            if st.state == TrialState.COMPLETE:
                print(f" {st.values[0]:10.5f}   {st.values[1]:12.5f}    {st.user_attrs.get('memo', '')}")
            elif st.state == TrialState.RUNNING:
                print(f"        *** RUNNING ***       {st.user_attrs.get('memo', '')}")            
            elif st.state == TrialState.PRUNED:
                print(f"        *** PRUNED ***        {st.user_attrs.get('memo', '')}")
            elif st.state == TrialState.FAIL:
                print(f"        *** FAILED ***        {st.user_attrs.get('memo', '')}")
            elif st.state == TrialState.WAITING:
                print(f"        *** WAITING ***       {st.user_attrs.get('memo', '')}")            
            else:
                print("\n")
    print(" *** end of trials *** ")
    
def propose_parameters(trial, objective, eval_metric):
 
    _params = {

        ## --------------------------------------------------------------
        ## General Parameters
        ## --------------------------------------------------------------            
        "verbosity"          : 0,
        # "objective"          :  "reg:squarederror",
        "objective"          :  objective,
        "eval_metric"        :  eval_metric,
        
        "booster"            :  "gbtree",   ## trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),

        ## Device     choices: 'cpu' . . . .
        # "device"          : "gpu"
        # "validate_parameters" [default: True]
        ## nthread [default to maximum number of threads available if not set]
        ##                   Number of parallel threads used to run XGBoost. When choosing it, please keep thread contention
        ##                  and hyperthreading in mind.
        
        ## disable_default_eval_metric [default= false]  Flag to disable default metric. Set to 1 or true to disable.
       
        # "n_estimators"     : trial.suggest_int("n_estimators", 75, 125),
        
        ## --------------------------------------------------------------
        ## Parameters for Tree Booster
        ## --------------------------------------------------------------
        
        ## tree_method:      [default="auto"] The tree construction algorithm used in XGBoost. 
        ##                   Choices: "auto", "exact", "approx", "hist", this is a combination of commonly used updaters. For other updaters like refresh, set the parameter updater directly.
        ##                             auto: Same as the hist tree method.
        ##                             exact: Exact greedy algorithm. Enumerates all split candidates.
        ##                             approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
        ##                             hist: Faster histogram optimized approximate greedy algorithm.        
        "tree_method"        : "auto",
        
        ## eta/learning_rate default =0.3 Step size shrinkage used in update to prevents overfitting. 
        ## After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.

        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 10, log=True, step = None),
        # "learning_rate"    : trial.suggest_float("learning_rate", 0.0000001, 1, log=True, step = None),

        
        ## GAMMA / min_split_loss: Default=0. Minimum loss reduction required to make a further partition on a leaf node of the tree. 
        ##                    The larger gamma is, the more conservative the algorithm will be.
        ##                    range: [0, Inf) 
        # "min_split_loss"   : trial.suggest_float("min_split_loss", 0, 10),
        "gamma"            : trial.suggest_float("min_split_loss", 0, 10),

        ## max_depth:  [Default=6] Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 
        ##             0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. 
        ##             exact tree method requires non-zero value. 
        "max_depth"        : trial.suggest_int("max_depth", 1, 15),
        
        ## min_child_weight:  [default=1] Minimum sum of instance weight (hessian) needed in a child. 
        ##                    If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, 
        ##                    then the building process will give up further partitioning. In linear regression task, this simply corresponds
        ##                    to minimum number of instances needed to be in each node. 
        ##                    The larger min_child_weight is, the more conservative the algorithm will be.
        ##                    range: [0,∞]        
        "min_child_weight"   : trial.suggest_float("min_child_weight", 0, 10),
        
        ## max_delta_step:    [default=0] Maximum delta step we allow each leaf output to be. 
        ##                    If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making 
        ##                    the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression 
        ##                    when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
        ##                    range: [0,∞]        
        "max_delta_step"   : trial.suggest_float("max_delta_step", 0, 10),

        ## subsample [default=1]: Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half
        ##                       of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once 
        ##                      in every boosting iteration.
        ##                      range: (0,1]
        "subsample"         : trial.suggest_float("subsample", 0.4, 1.0),

        ## sampling_method:  [default= uniform] The method to use to sample the training instances.
        ##              "uniform": each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
        ##              "gradient_based": the selection probability for each training instance is proportional to the regularized absolute value of gradients 
        ##              (more specifically, SQRT(g^2 + lambda*h^2). subsample may be set to as low as 0.1 without loss of model accuracy. 
        ##              Note that this sampling method is only supported when tree_method is set to hist and the device is cuda; 
        ##              other tree methods only support uniform sampling.        
        "sampling_method"   : "uniform",
        
        ## NOTE:  All colsample_by* parameters have a range of (0, 1], the default value of 1
        ##        and specify the fraction of columns to be subsampled.
        
        ## colsample_bytree;  [default=1] is the subsample ratio of columns when constructing each tree. 
        ##                    Subsampling occurs once for every tree constructed.
        "colsample_bytree" : 1.0, ## trial.suggest_float("colsample_bytree", 0.5, 1),

        ## colsample_bylevel: [default=1] is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. 
        ##                    Columns are subsampled from the set of columns chosen for the current tree.
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1),
        
        ## colsample_bynode:  [default=1] the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. 
        ##                    Columns are subsampled from the set of columns chosen for the current level.
        "colsample_bynode" : trial.suggest_float("colsample_bynode", 0.5, 1),

        # lambda [default=1, alias: reg_lambda]
        # L2 regularization term on weights. Increasing this value will make model more conservative.
        # range: [0, ∞]
        "lambda"           : 1,

        # alpha [default=0, alias: reg_alpha]
        # L1 regularization term on weights. Increasing this value will make model more conservative.        
        # range: [0, ∞]
        # "alpha"           : 0 

        ## scale_pos_weight [default=1] Control the balance of positive and negative weights, useful for unbalanced classes. 
        ##                 A typical value to consider: sum(negative instances) / sum(positive instances)        
        "scale_pos_weight" : 1,
                
        # tree_method string [default= auto] The tree construction algorithm used in XGBoost. See description in the reference paper and Tree Methods.
        #                  Choices: auto, exact, approx, hist, this is a combination of commonly used updaters. 
        #                           For other updaters like refresh, set the parameter updater directly.
        #                  auto:   Same as the hist tree method.
        #                  exact:  Exact greedy algorithm. Enumerates all split candidates`.
        #                  approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
        #                  hist:   Faster histogram optimized approximate greedy algorithm.
        
        # updater
        # refresh_leaf [default=1]
        # process_type [default= default]
        
        # grow_policy [default= depthwise]
        
        # max_leaves [default=0] Maximum number of nodes to be added. Not used by exact tree method.
        # "max_leaves"       : trial.suggest_int("max_leaves", 0, 2),
        "max_leaves"       : 0, 

        # max_bin, [default=256]
        
        ## num_parallel_tree, [default=1]: Number of parallel trees constructed during each iteration. 
        ##                    This option is used to support boosted random forest.
        
        ## multi_strategy, [default = one_output_per_tree]
        ## max_cached_hist_node, [default = 65536]

        ## --------------------------------------------------------------
        ## Parameters for Categorical Features
        ## --------------------------------------------------------------
        ## These parameters are only used for training with categorical data. See Categorical Data for more information.
        ## Note: These parameters are experimental. exact tree method is not yet supported.
        ##
        ## max_cat_to_onehot:   A threshold for deciding whether XGBoost should use one-hot encoding based split for categorical data. 
        ##                      When number of categories is lesser than the threshold then one-hot encoding is chosen, otherwise the 
        ##                      categories will be partitioned into children nodes.
        ##                      New in version 1.6.0.
        # "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 10),
        
        ## max_cat_threshold:  Maximum number of categories considered for each split. Used only by partition-based splits for preventing over-fitting.        
        ##                     New in version 1.7.0.
        

        ## --------------------------------------------------------------
        ## Learning Task Parameters
        ## --------------------------------------------------------------        

        

        ## SET STATIC ###############################################################################
        
        ## lambda [default=1, alias: reg_lambda]  L2 regularization term on weights. Increasing this value will make model more conservative.
        ##                    range: [0, Inf) 
        # "reg_lambda"       : trial.suggest_float("reg_lambda", 0, 10),
        "reg_lambda"       : 3.2267,   

    }    
    return _params

def rerun_objective(trial, disp_params = True, save = True):
    """
    Run objective without parameter sampling
    """
    metric_keys = ["train_auc","train_logloss", "val_auc", "val_logloss", "roc_auc", "logloss",
                    "accuracy","bal_acc","top_k_acc","F1_score","map","pearson_corr"]    
    study_params = {"booster"      : "gbtree",
                    "device"       : "gpu",
                    "objective"    :  "binary:logistic",
                    "eval_metric"  :  ["auc", "logloss"],
                    "verbosity"    : 0, 
                    "disable_default_eval_metric" : False,
                    ** trial.params}
    print('-'*80)
    print(f" Training model (trial #{trial.number}) ")
    print('-'*80)

    if disp_params:
        print(f" Parameters:")
        for k, v in study_params.items():
            print(f"  {k:30s} {v}")
            
    iter_files = make_cv_splits_2(input_file_list, n_folds=5, y_columns=y_columns)
    model, metrics =  train_model(iter_files, metric_keys = metric_keys, ** study_params)

    print_metric_hist(metrics)
    print(f" model best score    :  {model['booster'].best_score}")
    print(f" model best iteration:  {model['booster'].best_iteration}")
    
    if save:
        # save_as_filename = ".\saved_models\{1}_trial_{0:03d}.json".format(study.study_name,trial.number)
        save_as_filename = ".\saved_models\{0}_rerun_{1:03d}.json".format(study.study_name,trial.number)
        print(f" Save model to : {save_as_filename}")
        model['booster'].save_model(save_as_filename)
    
    # return np.array(metrics['val_auc']).mean(), np.array(metrics['val_logloss']).mean()


def balance_datasets(X,y, ratio = 2, verbose = False):
    y = y.astype(np.uint8)
    neg_idxs = np.nonzero([y == 0])[1]
    pos_idxs = np.nonzero([y == 1])[1]
    pos_idxs = np.array(pos_idxs)
    # print(f"\n # Pos rows: {len(pos_idxs)}   # Neg rows: {len(neg_idxs)}  Total: {len(pos_idxs)+ len(neg_idxs)}")
    # print(f" # Pos compounds: {len(pos_idxs)//3}   # Neg compounds: {len(neg_idxs)//3}  total: {(len(pos_idxs) + len(neg_idxs))//3}")
    # print(f"\n pos indexes - len: {len(pos_idxs)}")
    # print( pos_idxs[:50])
    # print( pos_idxs[-50:])
    # print(f"\n neg indexes - len: {len(neg_idxs)}")
    # print(neg_idxs[:50])
    # print(neg_idxs[-50:])
    
    stepped_pos_idxs = [x for x in pos_idxs if x % 3 == 0]
    stepped_neg_idxs = [x for x in neg_idxs if x % 3 == 0]
    stepped_neg_idxs = np.array(stepped_neg_idxs)
    pos_counts = len(stepped_pos_idxs)
    neg_counts = len(stepped_neg_idxs)
    
    if verbose:
        print(f"\n # Pos counts: {pos_counts}    # Neg counts: {neg_counts}   Total: {pos_counts+neg_counts}")
        print(f"\n pos indexes - len: {pos_counts}")
        print(stepped_pos_idxs[:25])
        print(stepped_pos_idxs[-25:])
        print(f"\n neg indexes - len: {neg_counts}")
        print(stepped_neg_idxs[:25])
        print(stepped_neg_idxs[-25:])
        print()
    num_neg_samples = ratio * pos_counts
    
    print(f"\n Take {pos_counts} samples from total of {pos_counts} postive training samples")
    print(f" Take {num_neg_samples} samples from total of {neg_counts} negative training samples")
    sample_idxs = skr.sample_without_replacement(n_population=neg_counts, n_samples= num_neg_samples, )
    sample_idxs.sort()
    neg_sample_idxs = stepped_neg_idxs[sample_idxs]
    if verbose:
        print(f"\n Sample indxs - len: {len(sample_idxs)}")
        print(f" {sample_idxs[:20]}")
        print(f" {sample_idxs[-20:]}")
        print(f"\n neg_sample_idxs: {len(neg_sample_idxs)}")
        print(f"{neg_sample_idxs[:20]}")
        print(f"{neg_sample_idxs[-20:]}")
        print()
        print(neg_sample_idxs[:20])
        print(neg_sample_idxs[:20]+1)
        print(neg_sample_idxs[:20]+2)
    
    neg_sample_idxs_3 = np.concatenate((neg_sample_idxs, neg_sample_idxs+1, neg_sample_idxs+2))
    neg_sample_idxs_3.sort()
    if verbose:
        print(f"\n pos_sample_idxs_3: {len(pos_idxs)}")
        print(f" [:20] :{pos_idxs[:20]}")
        print(f" [-20:] {pos_idxs[-20:]}")
        print(f"\n neg_sample_idxs_3: {len(neg_sample_idxs_3)}")
        print(f" [:20] :{neg_sample_idxs_3[:20]}")
        print(f" [-20:] {neg_sample_idxs_3[-20:]}")
    
    balanced_ds_idxs = np.concatenate((pos_idxs, neg_sample_idxs_3))

    bal_X = X[balanced_ds_idxs]
    bal_y = y[balanced_ds_idxs]

    print(f"\n Balanced Dataset: # pos samples: {len(pos_idxs)}    # Neg samples: {len(neg_sample_idxs_3)}  Total len: {len(balanced_ds_idxs)}")   
    print(f"\n X :  Min: {bal_X.min():.4f}    Max: {bal_X.max():.4f}   Mean: {bal_X.mean():.4f}  Std: {bal_X.std():.4f}")
    print(f" y :  Min: {bal_y.min():.4f}    Max: {bal_y.max():.4f}   Mean: {bal_y.mean():.4f}  Std: {bal_y.std():.4f}")
    return bal_X,bal_y



def label_counts(input_list = None, title = None, label = None):
    if input_list is None:
        input_list = [(title, label)]

    for (ttl, lbl) in input_list:
        bcnt = np.bincount(lbl.astype(np.int64))
        bcnt_sum = bcnt.sum()
        print(f" {ttl}")
        print('','-'*(len(ttl)+1))
        print(f" Total samples: {bcnt_sum}  - compounds: {bcnt_sum//3}")
        print(f" Label 0: {bcnt[0]:7,d}      % {bcnt[0]*100/bcnt_sum:2.2f} ")
        print(f" Label 1: {bcnt[1]:7,d}      % {bcnt[1]*100/bcnt_sum:2.2f} ")
        print("")
        
def compute_metrics(true, pred, title = 'Classification Metrics'):
    test_accuracy = skm.accuracy_score(true, pred)
    # precision : tp / (tp+fp)
    precision, recall, f1, support = skm.precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
    label_count = len(true)
    print(f" {title}")
    print('-'*(len(title)+2))
    print(f" Accuracy: {test_accuracy:.5f}     Precision: {precision:.5f}     Recall: {recall:.5f}     F1: {f1:.5f} \n"
          f"\n True + labels:        {true.sum():6.0f}     ratio to total:  {true.sum()/label_count:.5f}"
          f"\n Predicted + labels:   {pred.sum():6d}     ratio to total:  {pred.sum()/label_count:.5f}"
          f"\n True/Predicted Match: {(pred == true).sum():6d}     ratio to total:  {(pred==true).sum()/label_count:.5f}" )

def plots_from_estimator(estim, X, y):
    rows = 1
    cols = 3
    fig, axs = plt.subplots(1, cols, sharey=False, tight_layout=True, figsize=(cols *5,5))
    _ = skm.PrecisionRecallDisplay.from_estimator(estim, X, y, plot_chance_level = True, ax = axs[0])
    _ = skm.RocCurveDisplay.from_estimator(estim, X, y, plot_chance_level= True, ax = axs[1])
    _ = skm.ConfusionMatrixDisplay.from_estimator(estim, X, y, ax = axs[2])
    _ = axs[0].set_title(" Precision / Recall ")
    _ = axs[1].set_title(" ROC Curve ")
    plt.show()

def plots_from_predictions(true, pred):
    rows = 1
    cols = 3
    fig, axs = plt.subplots(1, cols, sharey=False, tight_layout=True, figsize=(cols *5,5) )
    _ = skm.PrecisionRecallDisplay.from_predictions(true, pred, plot_chance_level = True, ax = axs[0])
    _ = skm.RocCurveDisplay.from_predictions(true, pred, plot_chance_level= True, ax = axs[1])
    _ = skm.ConfusionMatrixDisplay.from_predictions(true, pred, ax = axs[2], values_format='d')
    _ = axs[0].set_title(" Precision / Recall ")
    _ = axs[1].set_title(" ROC Curve ")
    plt.show()

def plot_cls_metrics(y_true, y_prob, y_pred, epochs = None ):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    msg_sfx = f"- epoch:{epochs} " if epochs is not None else ""
    roc_display = skm.RocCurveDisplay.from_predictions(
        y_true.squeeze(),
        y_prob.squeeze(), 
        name=f"ROC Curve",
        color="darkorange",
        plot_chance_level=True,
        ax = axes[0])

    _ = roc_display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title = f" ROC curve {msg_sfx}")
        # title=f"ROC curve - TPSA Classification  \n LogLoss: {metrics['logloss'] :0.3f} AUC: {metrics['roc_auc']:0.3f} ",
    _ = roc_display.ax_.legend(fontsize=8)

    pr_display = skm.PrecisionRecallDisplay.from_predictions(
        y_true.squeeze(),
        y_prob.squeeze(),
        name="Precision/Recall",
        pos_label= 1,
        plot_chance_level=True,
        ax = axes[1])

    _ = pr_display.ax_.set_title(f" Precision-Recall curve {msg_sfx}")
    _ = pr_display.ax_.legend(fontsize=8)


    cm_display = skm.ConfusionMatrixDisplay.from_predictions(
        y_true.squeeze(),
        y_pred.squeeze(),
        values_format="5d",
        ax = axes[2])

    _ = cm_display.ax_.set_title(f"Confusion Matrix {msg_sfx}")

def save_checkpoint(epoch, model, optimizer = None, scheduler = None, 
                    filename = None, ckpt_path = "ckpts",
                    update_latest=False, update_best=False, 
                    verbose = False):
    """simplified version of save_checkpoint_v5 from utils_ptsnnl"""
    import torch.nn as nn
    from types import NoneType
    model_checkpoints_folder = os.path.join(ckpt_path)
    if not os.path.exists(model_checkpoints_folder):
        print(f"path {model_checkpoints_folder} doesn't exist")
        raise Exception(f"path {model_checkpoints_folder} doesn't exist") 

    checkpoint = {'epoch'      : epoch,
                  'params'     : dict()
                 }
    if isinstance(model, nn.modules.container.Sequential):
        checkpoint['model'] = model
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()

    ## save model attributes 
    model_attributes = model.__dict__
    for key, value in model_attributes.items():
        if key not in checkpoint:
            if key[0] == '_' :
                pass
                # if verbose:
                #     print(f"{key:40s}, {str(type(value)):60s} -- {key} in ignore_attributes - will not be added")
            else:
                if verbose:
                    print(f"{key:40s}, {str(type(value)):60s} -- add to checkpoint dict")
                checkpoint['params'][key] = value
        else:
            if verbose:
                print(f"{key:40s}, {str(type(value)):60s} -- {key} already in checkpoint dict")
    if verbose:
        print(checkpoint.keys())

    # if filename is None: 
    #     filename = f"{model.name}_{args.runmode[:4]}_{args.exp_title}_{args.exp_date}"
    #     if update_latest:
    #         s_filename = f"LAST_ep_{epoch:03d}"
    #     elif update_best:
    #         s_filename = f"BEST"
    #     else:
    #         s_filename = f"ep_{epoch:03d}"
    #     filename = f"{filename}_{s_filename}"
    if filename[-3:] != ".pt":
        filename += ".pt"

    torch.save(checkpoint, os.path.join(model_checkpoints_folder, filename)) 
    logger.info(f" Model exported to {filename} - epoch: {epoch}")


def load_checkpoint(model, optimizer = None, scheduler = None,
                    filename = None, ckpt_path = "ckpts",
                    dryrun = False,
                    verbose=False):

    if filename[-3:] != '.pt':
        filename += '.pt'
    if verbose:
        logging.info(f" Load model checkpoint from  {filename}")
    ckpt_file = os.path.join(ckpt_path, filename)

    try:
        checkpoint = torch.load(ckpt_file)
    except FileNotFoundError:
        Exception("Previous state checkpoint not found.")
        print(f"FileNotFound Exception - {filename}")
    except Exception as e :
        print("Other Exception")
        print(sys.exc_info())

    epoch = checkpoint.get('epoch',-1)
    # logger.info(f" ==> Loading from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")

    if dryrun:
        for key, value in checkpoint['params'].items():
            logging.info(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
        # model.load_state_dict(checkpoint['state_dict'])
    else:
        model = checkpoint['model']
        if optimizer is not None:
            # print(checkpoint['optimizer'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            # print(checkpoint['scheduler'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    if verbose:
        for key, value in checkpoint.items():
            logging.info(f"{key:40s}, {str(type(value)):60s} ")
        if hasattr(model, 'training_history'):
            for k in ['trn', 'val']:
                print(k)
                for kk,vv in model.training_history[k].items():
                    print(f" {k}-{kk}   checkpoint len: {len(vv)} ")
        print(f"model :\n {checkpoint['model']}\n")

        # for k,v in model.named_parameters():
        #     if v.requires_grad == False:
        #         v.requires_grad_()
        #         print(f" set {k} to requires_grad = True {v.requires_grad}")
        # display_model_hyperparameters(model, ' Loaded hyperparameters ')
        # display_model_parameters(model, 'loaded named parameters')

        # for k,v in checkpoint['optimizer'].items():
            # optimizer.load_state_dict(v)
            # if verbose:
                # print(f"optimizer state dict:\n {v['param_groups']}")

        # for k,v in checkpoint['scheduler'].items():
        # scheduler.load_state_dict(v)
        #     if verbose:
        #         print(f"scheduler state dict:\n  {v}")
    if verbose:
        logger.info(f" ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
    if hasattr(model,' trn_best_metric'):
        logger.info(f" Model best training metric   : {model.trn_best_metric:6f} - epoch: {model.trn_best_epoch}") 
    if hasattr(model, 'val_best_metric'):
        logger.info(f" Model best validation metric : {model.val_best_metric:6f} - epoch: {model.val_best_epoch}") 

    return model, optimizer, scheduler, epoch

