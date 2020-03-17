"""
INTENSE QC Component 3 - Rulebase

This component of the INTENSE QC package reads flagged files and applies a rulebase
to determine which rainfall data should be excluded. 
Required packages: 
    intense
    pandas
    numpy
    datetime
    os
    
Developed by: 
    Elizabeth Lewis, PhD
    SB, RV, others...

Publication to be cited:
    Paper

June 2019
"""

import intense as ex
import pandas as pd
import numpy as np
import os
import sys
# from multiprocessing import Pool, Manager
import multiprocessing as mp


# def apply_rulebase(filePath,directory):
def apply_rulebase(filePath, outPath, q=None):
    # try:
    s = ex.read_intense_qc(filePath)
    print(s.station_id)
    ''' #hashing these out because redundant
    # Helper function 1: station metadata summary
    statInfo = [s.station_id,
                s.country,
                s.original_station_number,
                s.original_station_name,
                s.latitude,
                s.longitude,
                s.elevation,
                datetime.datetime.strftime(s.start_datetime, "%Y%m%d%H"),
                datetime.datetime.strftime(s.end_datetime, "%Y%m%d%H"),
                s.number_of_records,
                s.percent_missing_data,
                s.original_timestep,
                s.original_units,
                s.resolution]


    #Helper function 2: single flag summary
    singleFlags= [s.QC_offset,# -1, 1
                  s.QC_preQC_affinity_index,# flaot
                  s.QC_preQC_pearson_coefficient,# float
                  s.QC_factor_daily,
                  s.QC_breakpoint,
                  s.QC_days_of_week,
                  s.QC_hours_of_day]

    # Helper function 3: list of years flagged by annual checks
    yearsFlags = [len(s.QC_change_min_value[1]) - s.QC_change_min_value[1].count(np.nan),# 1, list
                  len(s.QC_intermittency) - s.QC_intermittency.count(np.nan),
                  len(s.QC_percentiles[0]) - s.QC_percentiles[0].count(np.nan),
                  len(s.QC_percentiles[1]) - s.QC_percentiles[1].count(np.nan),
                  len(s.QC_k_largest[0]) - s.QC_k_largest[0].count(np.nan),
                  len(s.QC_k_largest[1]) - s.QC_k_largest[1].count(np.nan),
                  len(s.QC_k_largest[2]) - s.QC_k_largest[2].count(np.nan)]

    # Helper function 4: list of indicative R99pTOT and PCRPTOT flags
    # These are currently unused to exclude rainfall data
    annFlags = []
    for i in range(9):
        annFlags.append(s.QC_R99pTOT.count(i)) # 1, 2, 3, 4, 5, 6, 7, 8
    for i in range(9):
        annFlags.append(s.QC_PRCPTOT.count(i))# 1, 2, 3, 4, 5, 6, 7, 8
'''
    """
    ----------------------------------- Rulebase -----------------------------------
    """

    # hourlyFlags = s.data.drop("vals", axis=1)
    # summaryFlags = hourlyFlags.apply(pd.value_counts)

    # Copy original data 
    # origData = s.data
    # origData = origData.rename(columns={"vals":"orig_vals"})

    # 08/10/2019 (DP) - just copy original data to another column?
    s.data["orig_vals"] = s.data["vals"].copy()

    # List of possible flags:

    """
    possibleFlags = [["QC_hourly_neighbours", [0, 1, 2, 3]],
                    ["QC_hourly_neighbours_dry", [0, 1, 2, 3]],
                    ["QC_daily_neighbours", [0, 1, 2, 3]],
                    ["QC_daily_neighbours_dry", [0, 1, 2, 3]],
                    ["QC_monthly_neighbours", [-3, -2, -1, 0, 1, 2, 3]],
                    ["QC_world_record", [0, 1, 2, 3, 4]],
                    ["QC_Rx1day", [0, 1, 2, 3, 4, 5, 6, 7, 8]],
                    ["QC_CWD", [0, 1, 2, 3, 4, 5, 6, 7, 8]],
                    ["QC_CDD", [0, 1, 2, 3, 4, 5, 6, 7, 8]],
                    ["QC_daily_accumualtions", [0, 1, 2]],
                    ["QC_monthly_accumulations", [0, 1, 2]],
                    ["QC_streaks", [0, 1, 2]],
                    ["QC_factor_monthly", [-3, -2, -1, 0, 1, 2, 3]]]
    """

    # Calculate mean wet hour
    mwh = s.data.loc[s.data['vals'] > 0, 'vals'].mean()

    # R1: Exclude years where K largest = 0
    s.data["R1"] = 0  # for rulebase flags
    for i in range(3):
        for year in s.QC_k_largest[i]:
            try:
                year = int(year)
                # s.data.set_value(str(year), 'vals', np.nan)
                # s.data.set_value(str(year), "R1", 1)
                ##s.data.loc[s.data.index.year == year, 'vals'] = np.nan
                s.data.loc[s.data.index.year == year, 'R1'] = 1
            except:
                pass

    # R2: Exclude years where Q99/95 = 0
    # for i in range(2):
    # Now we only want to run it for Q99
    s.data["R2"] = 0

    # Start RB09 comment block
    '''
    
    for year in s.QC_percentiles[1]:
        if str(year) != 'nan': 
            #s.data.set_value(str(year), 'vals', np.nan)  
            #s.data.set_value(str(year), "R2", 1)
            ##s.data.loc[s.data.index.year == year, 'vals'] = np.nan
            s.data.loc[s.data.index.year == year, 'R2'] = 1
    
    '''
    # End RB09 comment block

    # R3: Exclude years where Intermitancy test failed
    # Not any more!
    # for year in s.QC_intermittency:
    # if str(year) != 'nan':
    # s.data.set_value(str(year), 'vals', np.nan)

    '''
    #R4: Exclude runs of >2 daily accumulations
    indexesToChange = []
    ds = s.data.QC_daily_accumualtions.values
    if np.max(ds)== 1:
        ds = [x * 2 for x in ds]
    if len(ds) > 48:
        for i in range(len(ds) - 48):
            if sum(ds[i+1:i+49]) >= 48: # NB THIS WORKS DIFFERENTLY IF THE FLAG HAS BEEN A '2' OR A '1' 
                for j in range(i+1,i+25):
                    indexesToChange.append(j)
    
    indexesToChange = np.unique(indexesToChange)
    for i in indexesToChange:
        s.data.iloc[i].vals = np.nan
    '''

    # R4: Exclude runs of >2 daily accumulations
    s.data["R4"] = 0
    s.data['prev_qc_daily_acc'] = s.data.shift(1)['QC_daily_accumualtions']
    s.data['next_qc_daily_acc'] = s.data.shift(-1)['QC_daily_accumualtions']
    s.data['prev_qc_daily_acc'] = np.where(s.data['prev_qc_daily_acc'].isnull(), 0, s.data['prev_qc_daily_acc'])
    s.data['next_qc_daily_acc'] = np.where(s.data['next_qc_daily_acc'].isnull(), 0, s.data['next_qc_daily_acc'])
    df = pd.DataFrame(
        dict(start=np.flatnonzero(
            (s.data.prev_qc_daily_acc == 0) & (s.data.QC_daily_accumualtions >= 1) & (s.data.next_qc_daily_acc >= 1)),
             end=np.flatnonzero((s.data.prev_qc_daily_acc >= 1) & (s.data.QC_daily_accumualtions >= 1) & (
                         s.data.next_qc_daily_acc == 0))))
    df['diff'] = df['end'] - df['start'] + 1
    df = df.loc[df['diff'] >= 48]
    idx = []
    for row in df.iterrows():
        r = range(row[1].start, row[1].end + 1)
        idx.extend(r)
    ##s.data.iloc[idx, s.data.columns.get_loc('vals')] = np.nan
    s.data.iloc[idx, s.data.columns.get_loc("R4")] = 1
    s.data.drop(['prev_qc_daily_acc', 'next_qc_daily_acc'], axis=1, inplace=True)

    # s.data["R5"] = 0
    # #R5: Exclude runs of >2 monthly accumulations
    # #s.data.loc[s.data['QC_monthly_accumulations'] > 0] = np.nan
    # s.data.loc[s.data['QC_monthly_accumulations'] > 0, 'vals'] = np.nan
    # s.data.loc[s.data['QC_monthly_accumulations'] > 0, 'R5'] = 1

    # Updated R5 because flags 3 and 4 now introduced to flag months that could be accumulations
    # apart from the fact that the 24 hours following the wet hour are all dry

    # *** IS THE INTENTION REALLY TO EXCLUDE >2 MONTH RUNS OF MONTHLY ACCUMULATIONS (I.E. SIMILAR TO DAILY)?
    # IF SO THIS NEEDS TO BE UPDATED, AS AT THE MOMENT IT APPLIES TO RUNS OF 1 MONTH ***
    # R5: Exclude runs of >2 monthly accumulations
    s.data["R5"] = 0
    # s.data.loc[s.data['QC_monthly_accumulations'] > 0] = np.nan
    ##s.data.loc[(s.data['QC_monthly_accumulations'] >= 1) & 
    ##    (s.data['QC_monthly_accumulations'] <= 2), 'vals'] = np.nan
    s.data.loc[(s.data['QC_monthly_accumulations'] >= 1) &
               (s.data['QC_monthly_accumulations'] <= 2), 'R5'] = 1

    # R6: Exclude streaks
    s.data["R6"] = 0
    ##s.data.loc[s.data['QC_streaks'] >0, 'vals'] = np.nan
    s.data.loc[s.data['QC_streaks'] > 0, "R6"] = 1

    # R7: Exclude world record any level
    s.data["R7"] = 0
    # For a more lenient option change '>0' with '>1' and manually check flagged and retained values
    ##s.data.loc[s.data['QC_world_record'] >0, 'vals'] = np.nan
    s.data.loc[s.data['QC_world_record'] > 0, 'R7'] = 1

    # R8: Exclude Rx1day any level <- changed to exclude it and previous 23 hours as well (remember this checks if hour exceeds ETCCDI daily max)
    s.data["R8"] = 0
    rx1_to_exclude = s.data['QC_Rx1day'] > 0
    for i in range(1, s.data.shape[0] - 1):
        if rx1_to_exclude.iloc[i] == True and rx1_to_exclude.iloc[
            i - 1] == False:  # Will only change previous flags if state changes from F -> T
            rx1_to_exclude.iloc[range(i - 23, i)] = True

    ##s.data.loc[rx1_to_exclude, 'vals'] = np.nan
    s.data.loc[rx1_to_exclude, 'R8'] = 1

    # R9: Exclude CWD any level
    s.data["R9"] = 0
    ##s.data.loc[s.data['QC_CWD'] >0, 'vals'] = np.nan
    s.data.loc[s.data['QC_CWD'] > 0, 'R9'] = 1

    '''
    # R10 flags
    def f2(x):
        if x['QC_hourly_neighbours'] ==3  and x['orig_vals'] > 2*mwh: return 1
        else: return 0
    s.data["R10"] = s.data.apply(f2, axis=1)
    #R10 Exclude hourly neighbours > 2x mean wet hour
    def f(x):
        if x['QC_hourly_neighbours'] ==3  and x['vals'] > 2*mwh: return np.nan
        else: return x['vals']
    s.data['vals'] = s.data.apply(f, axis=1)
    '''

    # R10 Exclude hourly neighbours > 2 x mean wet hour
    ##s.data['vals'] = np.where(
    ##    (s.data['QC_hourly_neighbours'] == 3) & (s.data['orig_vals'] > (2.0 * mwh)),
    ##    np.nan, s.data['vals'])
    s.data['R10'] = np.where(
        (s.data['QC_hourly_neighbours'] == 3) & (s.data['orig_vals'] > (2.0 * mwh)),
        1, 0)

    '''
    # R11 flags
    def f2(x):
        if x['QC_daily_neighbours'] ==3 and x['orig_vals'] > 2*mwh: return 1
        else: return 0
    s.data["R11"] = s.data.apply(f2, axis=1)
    #R11 Exclude daily neighbours > 2x mean wet hour
    def f(x):
        if x['QC_daily_neighbours'] ==3 and x['vals'] > 2*mwh: return np.nan
        else: return x['vals']
    s.data['vals'] = s.data.apply(f, axis=1)
    '''

    # R11 Exclude daily neighbours > 2 x mean wet hour
    ##s.data['vals'] = np.where(
    ##    (s.data['QC_daily_neighbours'] == 3) & (s.data['orig_vals'] > (2.0 * mwh)),
    ##    np.nan, s.data['vals'])
    s.data['R11'] = np.where(
        (s.data['QC_daily_neighbours'] == 3) & (s.data['orig_vals'] > (2.0 * mwh)),
        1, 0)

    '''
    # R12 flags
    def f2(x):
        if x['QC_hourly_neighbours_dry'] ==3 and x['QC_CDD'] > 0: return 1
        elif np.isnan(x['QC_hourly_neighbours_dry']) and x['QC_CDD'] > 0: return 1
        else: return 0
    s.data["R12"] = s.data.apply(f2, axis=1)
    #R12 Exclude hourly neighbours dry where CDD
    def f(x):
        if x['QC_hourly_neighbours_dry'] ==3 and x['QC_CDD'] > 0: return np.nan
        elif np.isnan(x['QC_hourly_neighbours_dry']) and x['QC_CDD'] > 0: return np.nan
        else: return x['vals']
    s.data['vals'] = s.data.apply(f, axis=1)
    '''

    # R12 Exclude hourly neighbours dry where CDD
    ##s.data['vals'] = np.where(
    ##    ((s.data['QC_hourly_neighbours_dry'] == 3) & (s.data['QC_CDD'] > 0)) |
    ##    ((np.isnan(s.data['QC_hourly_neighbours_dry'])) & (s.data['QC_CDD'] > 0)),
    ##    np.nan, s.data['vals'])
    s.data['R12'] = np.where(
        ((s.data['QC_hourly_neighbours_dry'] == 3) & (s.data['QC_CDD'] > 0)) |
        ((np.isnan(s.data['QC_hourly_neighbours_dry'])) & (s.data['QC_CDD'] > 0)),
        1, 0)

    '''
    # R13 flags
    def f2(x):
        if x['QC_daily_neighbours_dry'] ==3 and x['QC_CDD'] > 0: return 1
        elif np.isnan(x['QC_daily_neighbours_dry']) and x['QC_CDD'] > 0: return 1
        else: return 0
    s.data["R13"] = s.data.apply(f2, axis=1)
    #R13 Exclude daily neighbours dry where CDD
    def f(x):
        if x['QC_daily_neighbours_dry'] ==3 and x['QC_CDD'] > 0: return np.nan
        elif np.isnan(x['QC_daily_neighbours_dry']) and x['QC_CDD'] > 0: return np.nan
        else: return x['vals']
    s.data['vals'] = s.data.apply(f, axis=1)
    '''

    # R13 Exclude daily neighbours dry where CDD
    ##s.data['vals'] = np.where(
    ##    ((s.data['QC_daily_neighbours_dry'] == 3) & (s.data['QC_CDD'] > 0)) |
    ##    ((np.isnan(s.data['QC_daily_neighbours_dry'])) & (s.data['QC_CDD'] > 0)),
    ##    np.nan, s.data['vals'])
    s.data['R13'] = np.where(
        ((s.data['QC_daily_neighbours_dry'] == 3) & (s.data['QC_CDD'] > 0)) |
        ((np.isnan(s.data['QC_daily_neighbours_dry'])) & (s.data['QC_CDD'] > 0)),
        1, 0)

    '''
    # R14 flags
    s.data['R14'] = np.where(
        (s.data['QC_monthly_neighbours'] == 3) | (s.data['QC_monthly_neighbours'] == -3),
        1, 0)
    # R14 Exclude where 3 or more monthly neighbours are all >|100|% different to gauge
    s.data['vals'] = np.where(
        (s.data['QC_monthly_neighbours'] == 3) | (s.data['QC_monthly_neighbours'] == -3),
        np.nan, s.data['vals'])
    '''

    # R14 Exclude where 3 or more monthly neighbours are all >|100|% different 
    # to gauge and value outside of climatological max based on all neighbours
    # (with + 25% margin)
    # Also exclude if <3 neighbours online but greater than (2 * max), with min/max
    # again defined using all neighbours and data
    ##s.data['vals'] = np.where(
    ##    (np.absolute(s.data['QC_monthly_neighbours']) == 4) | 
    ##    (s.data['QC_monthly_neighbours'] == 5),
    ##    np.nan, s.data['vals'])
    s.data['R14'] = np.where(
        (np.absolute(s.data['QC_monthly_neighbours']) == 4) |
        (s.data['QC_monthly_neighbours'] == 5),
        1, 0)

    # Update values series based on rules
    rulebase_columns = ["R" + str(x) for x in range(1, 14 + 1) if x != 3]
    s.data['RemoveFlag'] = s.data[rulebase_columns].max(axis=1)
    s.data['vals'] = np.where(s.data['RemoveFlag'] == 0, s.data['vals'], np.nan)

    """
    ------------------------------------ Output ------------------------------------
    """
    # Update time series
    # s.data = pd.concat([origData, s.data], axis=1, join_axes=[origData.index])

    # Update percentage missing data 
    s.percent_missing_data = s.data.vals.isnull().sum() * 100 / len(s.data.vals.values)

    # Write file in INTENSE format
    # s.write(outPath)
    output_folder = outPath + "/QCd_Data/"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    s.write(output_folder)
    # old --> #s.write(os.path.join(directory))

    # 08/10/2019 (DP)
    # Write out station file as csv
    if not os.path.exists(outPath + "/RuleFlags/"):
        os.mkdir(outPath + "/RuleFlags/")
    # output_path = outPath + "/RuleFlags/" + s.station_id + "_v7a_ruleflags.csv"
    output_path = outPath + "/RuleFlags/" + s.station_id + ".csv"
    if not os.path.exists(output_path):
        s.data.to_csv(output_path, index_label="DateTime", na_rep="nan")

    # **********
    # FOR MULTI-PROCESSING WITH RULE FLAG SUMMARY

    # Summarise rulebase and amount of data removed

    # - Get percent missing from original and final - calculate difference
    percent_missing_original = s.data.orig_vals.isnull().sum() * 100 / len(s.data.orig_vals.values)
    percent_missing_qcd = s.data.vals.isnull().sum() * 100 / len(s.data.vals.values)
    percent_removed = percent_missing_qcd - percent_missing_original

    # - For removed hours, median number of rulebase flags
    rulebase_columns = ["R" + str(x) for x in range(1, 14 + 1) if x != 3]
    s.data["NumRulebaseFlags"] = s.data[rulebase_columns].sum(axis=1)
    median_rulebase_flags = s.data.loc[s.data["NumRulebaseFlags"] > 0, "NumRulebaseFlags"].median()
    ##s.data["RemovedFlag"] = np.where(s.data["NumRulebaseFlags"] > 0, 1, 0) # useful?

    '''
    # - Percentage of the time that R10 and R11 are the same
    s.data["R10_R11"] = np.where((s.data["R10"] == 1) & (s.data["R11"] == 1), 1, 0)
    a = s.data.loc[(s.data["R10"] == 1) | (s.data["R11"] == 1), ["R10", "R11", "R10_R11"]]
    if len(a["R10"]) > 0:
        percent_r10r11 = a["R10_R11"].sum() / float(len(a["R10"])) * 100.0
    else:
        percent_r10r11 = np.nan
    
    # - Percentage of the time that R12 and R13 are the same
    s.data["R12_R13"] = np.where((s.data["R12"] == 1) & (s.data["R13"] == 1), 1, 0)
    a = s.data.loc[(s.data["R12"] == 1) | (s.data["R13"] == 1), ["R12", "R13", "R12_R13"]]
    if len(a["R12"]) > 0:
        percent_r12r13 = a["R12_R13"].sum() / float(len(a["R12"])) * 100.0
    else:
        percent_r12r13 = np.nan
    '''

    # - Sum rulebase flags
    df1 = s.data.aggregate("sum")
    df1.index.name = "Quantity"
    df1 = df1.to_frame()
    df1.columns = ["Value"]
    df1 = df1.loc[rulebase_columns]
    df1 = df1 / len(s.data["vals"]) * 100.0

    # - Append other quantities
    df1.loc["percent_missing_original", "Value"] = percent_missing_original
    df1.loc["percent_missing_qcd", "Value"] = percent_missing_qcd
    df1.loc["percent_removed", "Value"] = percent_removed
    df1.loc["median_rulebase_flags", "Value"] = median_rulebase_flags
    ##df1.loc["percent_r10r11", "Value"] = percent_r10r11
    ##df1.loc["percent_r12r13", "Value"] = percent_r12r13

    # For multiprocessing
    output_list = df1["Value"].tolist()
    output_list.extend([s.station_id, s.latitude, s.longitude, s.number_of_records,
                        filePath, s.start_datetime, s.end_datetime])
    output_line = ",".join(str(x) for x in output_list)
    q.put(output_line)
    return output_line

    # **********

    # except:
    #    print('Passed')
    #    pass


# -----------------------------------------------------------------------------
# FOR MULTI-PROCESSING WITH RULE FLAG SUMMARY

def find_files():
    folders_to_check = sorted(os.listdir(root_folder))
    folders_to_check = [f for f in folders_to_check if f not in ['qcDebug', 'Superseded']]

    # RB09 - added to ensure only looking at folders, not e.g. RB summary files
    folders_to_check = [f for f in folders_to_check if os.path.isdir(root_folder + '/' + f)]

    file_paths = []

    for folder in folders_to_check:

        # List of QC flag files
        flag_folder = root_folder + "/" + folder + "/Flags/"
        file_names = sorted(os.listdir(flag_folder))

        # Equivalent list of paths
        for f in file_names:
            input_path = flag_folder + f
            output_folder = root_folder + "/" + folder
            qcd_data_path = output_folder + "/QCd_Data/" + f.replace("_QC.txt", ".txt")

            if not os.path.exists(qcd_data_path):
                tmp = [input_path, output_folder]
                file_paths.append(tmp)

    return file_paths


def listener(q):
    '''listens for messages on the q, writes to file. '''

    with open(summary_path, 'w') as f:

        headers = ["R" + str(x) for x in range(1, 14 + 1) if x != 3]
        ##headers.extend(["percent_missing_original", "percent_missing_qcd",
        ##    "percent_removed", "median_rulebase_flags", "percent_r10r11", 
        ##    "percent_r12r13", "station_id", "latitude", "longitude"])
        headers.extend(["percent_missing_original", "percent_missing_qcd",
                        "percent_removed", "median_rulebase_flags", "station_id",
                        "latitude", "longitude", "number_of_records",
                        "file_path", "start_date", "end_date"])
        headers = ",".join(headers)
        f.write(headers + "\n")

        while True:
            m = q.get()
            if m == 'kill':
                # f.write('killed')
                break
            f.write(str(m) + '\n')
            f.flush()


def main():
    # must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()
    # pool = mp.Pool(mp.cpu_count() + 2)
    pool = mp.Pool(num_processes)

    # put listener to work first
    watcher = pool.apply_async(listener, (q,))

    # get list of files to process
    file_paths = find_files()

    # fire off workers
    # input_files = sorted(os.listdir(inFolder)) #[:6]
    # input_files = ["MY_selangor_3416002_QC.txt"]
    jobs = []
    # for fn in input_files:
    for fn in file_paths:
        # job = pool.apply_async(apply_rulebase, (os.path.join(inFolder, fn), outFolder, q))
        job = pool.apply_async(apply_rulebase, (fn[0], fn[1], q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


# -----------------------------------------------------------------------------

'''
#countryList = ["Australia1min", "Australia5min", "Belgium", "Bermuda", "Brazil", "California", "Canada", "Catalonia", "CostaRica", "India", "France", "Finland",  "CostaRica", "Singapore",  "Sweden", "Netherlands", "Portugal", "Ireland", "Malaysia", "Italy", "Switzerland", "Norway", "Japan", "Germany", "Panama"]
#countryList = os.listdir("B:/QualityControlledData_v7/Flags")
countryList = ["Malaysia"]

def processFolder(country, q=None):

    #inFolder = "B:/QualityControlledData_v7/Flags/" + country
    inFolder = "U:/INTENSE data/QualityControlledData_v7/Flags/" + country
    #outFolder = "B:/QualityControlledData_v7/QCd_Data/"+ country 
    outFolder = "U:/INTENSE data/DP_To_Process/QC/v7_Checks/"
        
    for file in os.listdir(inFolder):
        if file == "MY_selangor_3416002_QC.txt":
            #print(file)
            #print(os.path.join(outFolder, file))
            apply_rulebase(os.path.join(inFolder, file), outFolder)

##processFolder(countryList[0])

num_processes = 4

if __name__ == '__main__':
    pool = Pool(processes=num_processes)
    m = Manager()
    q = m.Queue()
    for folderToCheck in countryList:
        pool.apply_async(processFolder, [folderToCheck, q])
    pool.close()
    pool.join()

    results = []
    while not q.empty():
        try:
            results.append(q.get())
        except:
            pass
            
'''

flags_version = "10"  # sys.argv[1] # '01c'
rulebase_version = "10"  # 'v7e' # sys.argv[2]
# country = "Malaysia"
# inFolder = "U:/INTENSE data/QualityControlledData_v7/Flags/" + country
# inFolder = '/media/nas/x21971/DP/QC_Checks/Malaysia/' + flags_version + '/Flags/'
root_folder = '/media/nas/x21971/QC_10'
# outFolder = "U:/INTENSE data/DP_To_Process/QC/v7_Checks/" + country
# outFolder = '/media/nas/x21971/DP/QC_Checks/Malaysia/' + flags_version + '/Rulebase_' + rulebase_version + '/'
# summary_path = "U:/INTENSE data/DP_To_Process/QC/v7_Checks/" + country + "/Rulebase_Summary_" + country + "_v7a_01.csv"
# summary_path = ('/media/nas/x21971/DP/QC_Checks/Malaysia/' + flags_version +
#    '/Rulebase_' + rulebase_version + '/Rulebase_Summary_Malaysia_FL' + flags_version + 
#    '_RB' + rulebase_version + '_01.csv')
summary_path = ('/media/nas/x21971/QC_10/Rulebase_Summary_FL' + flags_version +
                '_RB' + rulebase_version + '_01.csv')
num_processes = 5  # 4

# if not os.path.exists(outFolder):
#    os.mkdir(outFolder)

# input_files = sorted(os.listdir(inFolder)) #[:6]
# fn = input_files[0]
# q = None
# apply_rulebase(os.path.join(inFolder, fn), outFolder, q)
# sys.exit()

# Test
# file_paths = find_files()
# apply_rulebase(file_paths[0][0], file_paths[0][1])
# sys.exit()

# '''
if os.path.exists(summary_path):
    print('summary file already exists -', flags_version)
    sys.exit()

if __name__ == "__main__":
    main()
# '''
