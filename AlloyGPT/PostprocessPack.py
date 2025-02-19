import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from matplotlib.offsetbox import AnchoredText
import os, sys
# 
from pandas.api.types import infer_dtype
# from hmGPT.TestPack import covert_one_line_head_and_body
# from hmGPT.TrainPack import generate_single_example_AR_LM_w_token_tokenizer
from AlloyGPT.TestPack import covert_one_line_head_and_body
from AlloyGPT.TrainPack import generate_single_example_AR_LM_w_token_tokenizer

ref_alloy_SE_resu = {
    'Bulk_DiffResi_FCC_L12': 1.2381041848008516e+23, 
    'Bulk_DiffReed_FCC': 6.164630203464284e+19, 
    'Misfit_Amp_btw_L12_FCC': 0.6781516262637011, 
    'Misfit_btw_L12_FCC': -0.006781516262637011, 
    'Coarsening_Metric': 0.5477338939556056, 
    'VoluFrac_PerCent_L12': 2.5294852704908504, 
    'R_Opti_At_MaxStrength': 8.272727272727273, 
    'Threshold_Stress': 73.19051690645342, 
    'Orowan_Stress': 179.77296762720937, 
    'MF_L12': 0.02360115061858477, 
    'Elem_Cost': 4.2181161, 
    'All_Phase_Name': 'AL3M_D023#1@AL3M_L12#1@AL3NI1#1@AL3Y_D019#1@FCC_A1#1', 
    'All_Phase_MF': '6.716766E-03@2.360115E-02@4.082811E-02@7.855777E-04@9.280684E-01'
}

ref_alloy_SC_resu = {
    'Freezing_Range_From_fccAl': 14.115624999999909, 
    'Freezing_T_From_fccAl': 924.4945002, 
    'MF_AllSolid_at_Freezing_From_fccAl': 0.01589400082649642, 
    'MF_L12_at_Freezing_From_fccAl': 0.002384521408811688, 
    'MF_Al3Ni_at_Freezing_From_fccAl': 0.0, 
    'MF_L3_at_Freezing_From_fccAl': 0.0, 
    'MF_Al3Zr_at_Freezing_From_fccAl': 0.012631515013883798, 
    'T_at_FullSolid': 910.3788752, 
    'MF_L12_at_FullSolid': 0.013109623626322705, 
    'MF_Al3Ni_at_FullSolid': 0.02989654275533658, 
    'MF_L3_at_FullSolid': 0.010357840503645573, 
    'MF_Al3Zr_at_FullSolid': 0.012631515013883798, 
    'Freezing_Range_From_AllSolid_w_cutoff': 14.11562, 
    'CSC_0': -0.0, 
    'HCS_0': -0.0, 
    'CSC_1': 0.19238, 
    'HCS_1': 0.027155629755999996, 
    'Find_L12': 1, 
    'Freezing_T_From_L12': 975.6220002, 
    'MF_NS_at_Freezing_From_L12': 0.012632991084696113, 
    'Find_Al3Ni': 1, 
    'Freezing_T_From_Al3Ni': 911.1120002, 
    'MF_NS_at_Freezing_From_Al3Ni': 0.6820072048111355, 
    'Find_L3': 1, 
    'Freezing_T_From_L3': 911.1757502, 
    'MF_NS_at_Freezing_From_L3': 0.6743420079723822, 
    'Find_Al3Zr': 1, 
    'Freezing_T_From_Al3Zr': 1201.355125, 
    'MF_NS_at_Freezing_From_Al3Zr': 6.467871305003354e-07, 
    'All_Stable_Phase': 'AL3M_D023@AL3M_L12@FCC_A1@AL23NI6M4@AL3NI1', 
    'All_Stable_Phase_MF': '1.263152E-02@1.310962E-02@9.327753E-01@1.035784E-02@2.989654E-02'
}

# 1. get mole % 
# ref: https://iupac.qmul.ac.uk/AtWt/
mole_wei={}
mole_wei['Ni'] = 58.6934
mole_wei['Er'] = 167.259
mole_wei['Zr'] = 91.224
mole_wei['Y']  = 88.905838
mole_wei['Yb'] = 173.045
mole_wei['Al'] = 26.9815384

def convert_wei_PC_to_mole_PC(elem):
    # assume element list: 
    # ["Ni", "Er", "Zr", "Y", "Yb"]
    # 
    # 
    Al_wc = 1.-elem['Ni_wt']-elem['Er_wt'] \
    -elem['Zr_wt']-elem['Y_wt']-elem['Yb_wt']
    # 
    Ni_mole = elem['Ni_wt']/mole_wei['Ni']
    Er_mole = elem['Er_wt']/mole_wei['Er']
    Zr_mole = elem['Zr_wt']/mole_wei['Zr']
    Y_mole  = elem['Y_wt']/mole_wei['Y']
    Yb_mole = elem['Yb_wt']/mole_wei['Yb']
    Al_mole = Al_wc/mole_wei['Al']
    # 
    tot_mole = Ni_mole+Er_mole+\
    Zr_mole+Y_mole+Yb_mole+Al_mole
    # 
    resu={}
    resu['NiMole']=Ni_mole/tot_mole
    resu['ErMole']=Er_mole/tot_mole
    resu['ZrMole']=Zr_mole/tot_mole
    resu['YMole']=Y_mole/tot_mole
    resu['YbMole']=Yb_mole/tot_mole
    resu['AlMole']=Al_mole/tot_mole
    return resu

# ================================================
# 
def extract_phase_info_from_elem_string(
    elem,
    target_list=[
        'FCC_A1#1', 'AL3M_L12#1', 'AL23NI6M4#1', 
        'AL3NI1#1', 'AL3M_D023#1'
    ],
    
):
    # if not (elem['All_Phase_Name']==None or elem['All_Phase_MF']==None):
    #     phase_name_list = elem['All_Phase_Name'].split('@')
    #     phase_mole_list = elem['All_Phase_MF'].split('@')
    # else:
    #     phase_name_list = 'FCC_A1#1@AL3M_L12#1@AL23NI6M4#1@AL3NI1#1@AL3M_D023#1'.split('@')
    #     phase_mole_list = '0.@0.@0.@0.@0.'.split('@')

    try:
        phase_name_list = elem['All_Phase_Name'].split('@')
        phase_mole_list = elem['All_Phase_MF'].split('@')
    except:
        phase_name_list = 'FCC_A1#1@AL3M_L12#1@AL23NI6M4#1@AL3NI1#1@AL3M_D023#1'.split('@')
        phase_mole_list = '0.@0.@0.@0.@0.'.split('@')
        
    # 
    # phase list: 
    # target_list = [
    #     'FCC_A1#1', 'AL3M_L12#1', 'AL23NI6M4#1', 
    #     'AL3NI1#1', 'AL3M_D023#1'
    # ]
    # FCC, L12, L3, Al3Ni, Al3Zr
    # 
    resu = {}
    for this_target in target_list:
        if this_target in phase_name_list: 
            this_idx = phase_name_list.index(this_target)
            this_phase_MF = phase_mole_list[this_idx]
        else:
            this_phase_MF = '0.0'
        this_phase_MF = float(this_phase_MF)
        # 
        resu[this_target]=this_phase_MF
    
    return resu
    
def postprocess_modeling_resu_from_gene_comp(
    base_path = None, #  gene_source_PtoSC,
):
    csv_SC = base_path+'/resu/1_working_place/3_tot_resu_for_task_1_Scheil.csv'
    csv_SE = base_path+'/resu/1_working_place/3_tot_resu_for_task_2_Single_Equ.csv'

    df_SingEqui = pd.read_csv(csv_SE)
    # keys
    print (f"Initial keys: ")
    print (f"On Single equ: \n", df_SingEqui.keys().tolist())

    # start to process
    # on Single Equilibrium
    df_SingEqui['AlMolePC']=df_SingEqui.apply(
        lambda x: convert_wei_PC_to_mole_PC(x)['AlMole']*100.,
        axis=1
    )
    df_SingEqui['NiMolePC']=df_SingEqui.apply(
        lambda x: convert_wei_PC_to_mole_PC(x)['NiMole']*100.,
        axis=1
    )
    df_SingEqui['ErMolePC']=df_SingEqui.apply(
        lambda x: convert_wei_PC_to_mole_PC(x)['ErMole']*100.,
        axis=1
    )
    df_SingEqui['ZrMolePC']=df_SingEqui.apply(
        lambda x: convert_wei_PC_to_mole_PC(x)['ZrMole']*100.,
        axis=1
    )
    df_SingEqui['YMolePC']=df_SingEqui.apply(
        lambda x: convert_wei_PC_to_mole_PC(x)['YMole']*100.,
        axis=1
    )
    df_SingEqui['YbMolePC']=df_SingEqui.apply(
        lambda x: convert_wei_PC_to_mole_PC(x)['YbMole']*100.,
        axis=1
    )
    # add the normalized ones
    df_SingEqui['NZ_Bulk_DiffResi_FCC_L12']=df_SingEqui.apply(
        lambda x: x['Bulk_DiffResi_FCC_L12']/ref_alloy_SE_resu['Bulk_DiffResi_FCC_L12'],
        axis=1,
    )
    # 
    df_SingEqui['NZ_Misfit_Amp_btw_L12_FCC']=df_SingEqui.apply(
        lambda x: x['Misfit_Amp_btw_L12_FCC']/ref_alloy_SE_resu['Misfit_Amp_btw_L12_FCC'],
        axis=1,
    )
    # 
    # Secondary properties:
    # prefer not to use the 'Coarsening_Metric' from the code
    # as the prefactor is a bit different
    #
    df_SingEqui['NZ_Coarsening_Metric']=df_SingEqui.apply(
        lambda x: x['NZ_Misfit_Amp_btw_L12_FCC']/x['NZ_Bulk_DiffResi_FCC_L12'] if x['NZ_Bulk_DiffResi_FCC_L12']!=0 else -1 ,
        axis=1,
    )
    # 
    df_SingEqui['NZ_Coarsening_Metric_CutAt4']=df_SingEqui.apply(
        lambda x: np.amin([x['NZ_Coarsening_Metric'], 4.]),
        axis=1,
    )
    # on other phases
    # print (df_SingEqui.keys())
    # # 
    # df_SingEqui['MF_L12']=df_SingEqui.apply(
    #     lambda x: extract_phase_info_from_elem_string(x)['AL3M_L12#1'],
    #     axis=1
    # )
    # 
    df_SingEqui['MF_L3']=df_SingEqui.apply(
        lambda x: extract_phase_info_from_elem_string(x)['AL23NI6M4#1'],
        axis=1
    )
    # 
    df_SingEqui['MF_Al3Ni']=df_SingEqui.apply(
        lambda x: extract_phase_info_from_elem_string(x)['AL3NI1#1'],
        axis=1
    )
    # 
    df_SingEqui['MF_Al3Zr']=df_SingEqui.apply(
        lambda x: extract_phase_info_from_elem_string(x)['AL3M_D023#1'],
        axis=1
    )
    # check keys
    print (f"UPdated Single equ keys: \n", df_SingEqui.keys().tolist())
    # 
    # ==================================================================
    # Scheil calculations
    df_ScheCalc = pd.read_csv(csv_SC)
    
    print (f"Initial keys: ")
    print (f"On Sche Calc: \n", df_ScheCalc.keys().tolist())
    
    df_ScheCalc['NZ_Freezing_Range_From_fccAl']=df_ScheCalc.apply(
        lambda x: x['Freezing_Range_From_fccAl']/ref_alloy_SC_resu['Freezing_Range_From_fccAl'],
        axis=1,
    )
    #
    df_ScheCalc['NZ_Freezing_Range_From_AllSolid_w_cutoff']=df_ScheCalc.apply(
        lambda x: x['Freezing_Range_From_AllSolid_w_cutoff']/ref_alloy_SC_resu['Freezing_Range_From_AllSolid_w_cutoff'],
        axis=1,
    )
    # Note, CSC_1 is already dimensionless, will not normalze it
    # secondary property
    # 
    df_ScheCalc['NZ_HCS']=df_ScheCalc.apply(
        lambda x: x['CSC_1']*x['NZ_Freezing_Range_From_fccAl'],
        axis=1,
    )
    print (f"UPdated Scheil Calc keys: \n", df_ScheCalc.keys().tolist())
    #
    # =========================================================================
    # safety keys 
    print (f"Include Safety keys. From SE, try IF_SafeCal_SE; from SC, try IF_SafeCal_SC")
    # 
    # =========================================================================
    # merge
    df_resu = None
    # 
    df_resu = pd.DataFrame(columns=[])
    # add some
    # =================================================
    # composition
    df_resu['AlMolePC']=df_SingEqui['AlMolePC']
    df_resu['NiMolePC']=df_SingEqui['NiMolePC']
    df_resu['ErMolePC']=df_SingEqui['ErMolePC']
    df_resu['ZrMolePC']=df_SingEqui['ZrMolePC']
    df_resu['YMolePC']=df_SingEqui['YMolePC']
    df_resu['YbMolePC']=df_SingEqui['YbMolePC']
    # 
    # =================================================
    # structures: phase info 
    # L12 Mole PC using single equilibrium
    df_resu['L12MolePC']=df_SingEqui['MF_L12']*100.
    df_resu['TerneryMolePC']=df_SingEqui['MF_L3']*100.
    df_resu['Al3NiMolePC']=df_SingEqui['MF_Al3Ni']*100.
    df_resu['Al3ZrMolePC']=df_SingEqui['MF_Al3Zr']*100.
    # 
    # 
    # # L12 Mole PC at freezing point using Scheil
    # df_resu['ScheilL12MeltMolePC']=df_ScheCalc['MF_L12_at_Freezing_From_fccAl']*100.
    # L12 Mole PC at full solid using Scheil
    df_resu['ScheilL12MolePC']=df_ScheCalc['MF_L12_at_FullSolid']*100.
    # 
    # # L3 Mole PC at freezing point using Scheil
    # df_resu['ScheilTerneryMeltMolePC']=df_ScheCalc['MF_L3_at_Freezing_From_fccAl']*100.
    # L3 Mole PC at full solid using Scheil
    df_resu['ScheilTernaryMolePC']=df_ScheCalc['MF_L3_at_FullSolid']*100.
    # 
    # # Al3Ni Mole PC at freezing point using Scheil
    # df_resu['ScheilAl3NiMeltMolePC']=df_ScheCalc['MF_Al3Ni_at_Freezing_From_fccAl']*100.
    # Al3Ni Mole PC at full solid using Scheil
    df_resu['ScheilAl3NiMolePC']=df_ScheCalc['MF_Al3Ni_at_FullSolid']*100.
    # 
    # # Al3Zr Mole PC at freezing point using Scheil
    # df_resu['ScheilAl3ZrMeltMolePC']=df_ScheCalc['MF_Al3Zr_at_Freezing_From_fccAl']*100.
    # Al3Zr Mole PC at full solid using Scheil
    df_resu['ScheilAl3ZrMolePC']=df_ScheCalc['MF_Al3Zr_at_FullSolid']*100.
    # 
    # =================================================
    # properties: may want to normalize the properties using a reference compositions
    # 
    # 1. from single equilibrium
    # bulk diffusion resistivity
    df_resu['NZ_BulkResistivity']=df_SingEqui['NZ_Bulk_DiffResi_FCC_L12']
    # Misfit amplitude Per Centage
    df_resu['NZ_Misfit']=df_SingEqui['NZ_Misfit_Amp_btw_L12_FCC']
    # Coarsening metric
    df_resu['NZ_CoarseningMetric']=df_SingEqui['NZ_Coarsening_Metric_CutAt4']
    # 
    # ===========================================================================
    # 2. from Scheil calculations
    # Freezing range using Matrix
    df_resu['NZ_Freezing_Range_From_fccAl']=df_ScheCalc['NZ_Freezing_Range_From_fccAl']
    # Crack sus. coefficient
    df_resu['CSC']=df_ScheCalc['CSC_1']
    # Secondary property: hot crack sus.
    df_resu['NZ_HCS']=df_ScheCalc['NZ_HCS']
    # 
    # ===========================================================================
    # 3. safety keys
    if 'IF_SafeCal_SE' in df_SingEqui.keys():
        print (f"add safety key from SingEqui: \nIF_SafeCal_SE\n\n")
        df_resu['IF_SafeCal_SE']=df_SingEqui['IF_SafeCal_SE']
    else:
        print (f"No safety key found for SingEqui...\n\n")
        
    if 'IF_SafeCal_SC' in df_ScheCalc.keys():
        print (f"add safety key from ScheCalc: \nIF_SafeCal_SC\n\n")
        df_resu['IF_SafeCal_SC']=df_ScheCalc['IF_SafeCal_SC']
    else:
        print (f"No safety key found for ScheCalc\n\n")
    

    # =============================================================================
    # MAKE the names close to those from model predictions
    # 
    # rename some columns: based on DataPack.py
    # assemble_one_sentence
    # 
    df_resu = df_resu.rename(
        columns={
            # composition
            'AlMolePC': '(Al)', 
            'NiMolePC': '(Ni)',
            'ErMolePC': '(Er)',
            'ZrMolePC': '(Zr)',
            'YMolePC': '(Y)',
            'YbMolePC': '(Yb)',
            # structure numbers
            # as-built
            'ScheilL12MolePC': 'AsBuilt_L12Mol%',
            'ScheilTernaryMolePC': 'AsBuilt_TernaryMol%',
            'ScheilAl3NiMolePC': 'AsBuilt_Al3NiMol%',
            'ScheilAl3ZrMolePC': 'AsBuilt_Al3ZrMol%',
            # aged
            'L12MolePC': 'L12Mol%',
            'TerneryMolePC': 'TernaryMol%',
            'Al3NiMolePC': 'Al3NiMol%',
            'Al3ZrMolePC': 'Al3ZrMol%',
            # properties
            # dimensionless
            'NZ_BulkResistivity': 'DiffusionResistivity',
            'NZ_Misfit': 'Misfit',
            'NZ_CoarseningMetric': 'CoarseningMetric',
            'NZ_Freezing_Range_From_fccAl': 'FreezingRange',
            'CSC': 'CrackSusceptibilityCoefficient',
            'NZ_HCS': 'HotCrackingSusceptibility',
        }
    )
    # 
    print (f"Result keys: \n{df_resu.keys().tolist()}")
    
    return df_SingEqui, df_ScheCalc, df_resu

def recover_initial_generated_records(
    base_path = None,
):
    csv_other_info = base_path + '/resu/0_compositions/1_other_info_tot.csv'
    df_ori_info = pd.read_csv(csv_other_info)

    return df_ori_info

def plot_loop_for_given_keys(
    this_df,
    wordroot_list,
    added_tail_list,
    prefix,
    x_y_labels = None,
):
    # r2_list = []
    # mse_list = []
    # mae_list = []
    # 
    r2_list = {}
    mse_list = {}
    mae_list = {}
    for this_key in wordroot_list:
    # this_key = Forward_Resu_Keys[ii]
        # 
        fig = plt.figure(figsize=(24,16),dpi=200)
        fig, ax0 = plt.subplots()

        ax0.scatter(
            this_df[this_key+added_tail_list[0]], 
            this_df[this_key+added_tail_list[1]], 
            color='red',
            alpha=0.01,
        )
        x1 = np.amax(
            [max(this_df[this_key+added_tail_list[0]]),
             max(this_df[this_key+added_tail_list[1]])]
        )
        x0 = np.amin(
            [min(this_df[this_key+added_tail_list[0]]),
             min(this_df[this_key+added_tail_list[1]])]
        )
        ax0.plot(
            [x0,x1],
            [x0,x1],
            linestyle='dotted',
            color='b'
        )

        this_r2 = r2_score(
            y_true=this_df[this_key+added_tail_list[0]], 
            y_pred=this_df[this_key+added_tail_list[1]]
        )
        this_mse = mean_squared_error(
            y_true=this_df[this_key+added_tail_list[0]], 
            y_pred=this_df[this_key+added_tail_list[1]]
        )
        this_mae = mean_absolute_error(
            y_true=this_df[this_key+added_tail_list[0]], 
            y_pred=this_df[this_key+added_tail_list[1]]
        )

        print(f"r2 of %s: %10.3f" % (
                '{0: >25}'.format(this_key), 
                  this_r2
            )
        )
        print(f"mse of %s: %10.3f" % (
                '{0: >25}'.format(this_key), 
                  this_mse
            )
        )
        print(f"mae of %s: %10.3f" % (
                '{0: >25}'.format(this_key), 
                  this_mae
            )
        )
        # r2_list.append(this_r2)
        # mse_list.append(this_mse)
        # mae_list.append(this_mae)
        r2_list[this_key] = this_r2
        mse_list[this_key] = this_mse
        mae_list[this_key] = this_mae
        # 
        # plt.text()
        anchored_text = AnchoredText(
            # f"r2: {this_r2}\nMSE: {this_mse}\nMAE: {this_mae}", 
            f"R2:   %10.3f\nMSE: %10.3f\nMAE: %10.3f" % (
                this_r2, this_mse, this_mae
            ), 
            loc=2
        )
        ax0.add_artist(anchored_text)

        plt.title(prefix +this_key)
        if x_y_labels is None:
            plt.xlabel(added_tail_list[0][1:])
            plt.ylabel(added_tail_list[1][1:])
        else:
            plt.xlabel(x_y_labels[0])
            plt.ylabel(x_y_labels[1])
        # plt.xlim([-60, 500])
        # plt.ylim([-60, 500])
        plt.show()
        plt.close()
        
    return r2_list, mse_list, mae_list
# 
def extend_metrics_from_part_keys_to_full_keys(
    full_key_list,
    part_key_dict,
):
    resu = {}
    resu_list = []
    for this_key in full_key_list:
        if this_key in part_key_dict:
            resu[this_key]=part_key_dict[this_key]
        else:
            resu[this_key]=0
        # 
        resu_list.append(resu[this_key])
    return resu, resu_list
# 
def analyze_difference_based_on_model_id_temp(
    df, # all data stored here
    wordroot, 
    word_ends, # use these two keys pick up values to be compared
    sample_id_key, # to identify the model
    pred_temp_key, # prediction temp
):
    # find how many samples we have
    unique_sample_id_list = np.array(
        list(set(df[sample_id_key].values))
    )
    unique_sample_id_list = np.sort(unique_sample_id_list)
    print (f"Find unique samples as {unique_sample_id_list}")

    # find how many pred_temp we have
    unique_T_list = np.array(
        list(set(df[pred_temp_key].values))
    )
    unique_T_list = np.sort(unique_T_list)
    print (f"Find unique T as {unique_T_list}")

    num_word = len(wordroot)
    
    # analyze sample one by one
    # result arr: (t_i, sample_i, word_i, key_result)
    # key_result: PR_n_try, PR_ave, PR_std, 
    #             PR_GT_ave_relaL1, PR_GT_R2, PR_GT_mse, PR_GT_mae
    resu_key_list = [
        "PR_n_try", "PR_ave", "PR_std",
        "PR_GT_ave_relaL1", "PR_coeff_of_variation"
        # "PR_GT_R2", "PR_GT_MSE", "PR_MAE",
    ]
    resu_arr = np.zeros(
        (
            len(unique_T_list), len(unique_sample_id_list),
            len(wordroot), len(resu_key_list)
        )
    )
    for i_sample, this_sample_id in enumerate(unique_sample_id_list):
        print (f"Working on sample : {this_sample_id}")
        
        for i_T, this_T in enumerate(unique_T_list):
            # get records of this sample
            this_sample_T_list = df.index[
                ((df[sample_id_key]==this_sample_id) & (df[pred_temp_key]==this_T))
            ].tolist()
            
            for i_word, this_WR in enumerate(wordroot):
                Y_GT = df[this_WR+word_ends[0]].values[this_sample_T_list]
                Y_PR = df[this_WR+word_ends[1]].values[this_sample_T_list]
                # collect the results
                # 
                this_PR_n_try = len(Y_PR)
                this_PR_ave = np.mean(Y_PR)
                this_PR_var = np.std(Y_PR)
                # 
                this_PR_GT_relaL1_list = [ np.fabs((Y_PR[i]-Y_GT[i])/Y_GT[i]) for i in range(this_PR_n_try)]
                this_PR_GT_ave_relaL1 = np.mean(this_PR_GT_relaL1_list)
                # 
                this_PR_cv = this_PR_var/this_PR_ave
                # # 
                # this_PR_GT_R2 = r2_score(
                #     y_true=Y_GT,
                #     y_pred=Y_PR,
                # )
                # this_PR_GT_MSE = mean_squared_error(
                #     y_true=Y_GT,
                #     y_pred=Y_PR,
                # )
                # this_PR_GT_MAE = mean_absolute_error(
                #     y_true=Y_GT,
                #     y_pred=Y_PR,
                # )
                # deliver
                resu_arr[i_T, i_sample, i_word, :] = [
                    this_PR_n_try, this_PR_ave, this_PR_var,
                    this_PR_GT_ave_relaL1, 
                    this_PR_cv,
                    # this_PR_GT_R2,
                    # this_PR_GT_MSE,
                    # this_PR_GT_MAE
                ]
            
            
            
    return resu_arr, resu_key_list, unique_T_list, unique_sample_id_list
#
# =============================================================
# 
def predict_at_various_T(
    csv_PtoSC_vari_T_set,
    txt_PtoSC_vari_T_set,
    df_key_list,
    added_keys=['Pred_Temp', 'ID_Sample', 'ID_Pred'],
    sample_pick_list=None,
    full_sentence_dict_on_task=None, # sentence_dataset_dict_sepe['test']['Gene001_sentence'],
    input_promp_len=None, # input_len_for_tasks['Gene_PtoSC'],
    pred_temp_list=None,
    n_pred_at_one_time=30,
    # === model prediction ===
    ctx=None,
    model=None,
    tokenizer=None,
    device=None,
    top_k=None,
):
    
    if not os.path.exists(csv_PtoSC_vari_T_set):
        # 1. handle the top line
        # top_line = 'Pred_Temp,ID_Sample,ID_Pred,'
        top_line = ''
        for this_word in added_keys:
            top_line += this_word+','
        for this_word in df_key_list:
            top_line += this_word+'_PR,'+this_word+'_GT,'
        top_line = top_line[:-1]+'\n'
        # 
        with open(csv_PtoSC_vari_T_set, "w") as f:
            f.write(top_line)
        #
        # body part
        for this_ii_pick in sample_pick_list:
            print (f"Working on sample #: {this_ii_pick}")
            # pick the full
            # full_promp = sentence_dataset_dict_sepe['train']['Gene001_sentence'][this_ii_pick]
            full_promp = full_sentence_dict_on_task[this_ii_pick]
            print (f"Full line (i.e., GT): {full_promp}")
            # test_promp = full_promp[:input_len_for_tasks['Gene_PtoSC']]
            test_promp = full_promp[:input_promp_len]
            print (f"input promp: {test_promp}")
            # get into GT
            # this_resu_dict_GT = TestPack.covert_one_line_head_and_body(
            this_resu_dict_GT = covert_one_line_head_and_body(
                this_reco=full_promp,
                df_key_list=df_key_list,
            )
            print (f"GT: {this_resu_dict_GT}")
            # 
            # make predictions at different T
            for this_pred_T in pred_temp_list:
                # test_line_preds = TrainPack.generate_single_example_AR_LM_w_token_tokenizer(
                test_line_preds = generate_single_example_AR_LM_w_token_tokenizer(
                    ctx,
                    model,
                    tokenizer,
                    test_promp,
                    device=device,
                    pred_num=n_pred_at_one_time,
                    max_new_tokens=1024-input_promp_len, # 1024-input_len_for_tasks['Gene_PtoSC'],
                    pred_temp=this_pred_T, # 0.1, # 1.,
                    top_k=top_k, # None, # 100,
                )
                # print and record the results
                for ii, this_line in enumerate(test_line_preds):
                    
                    add_line = "For T={}, Input={}, Sample={}: \n{}\n".format(this_pred_T,this_ii_pick, ii, this_line)
                    # print (add_line)
                    with open(txt_PtoSC_vari_T_set, "a") as ft:
                        ft.write(add_line)
                    # 
                    # explain the results
                    # 
                    # generated recrod
                    # test_resu_dict = TestPack.covert_one_line_head_and_body(
                    test_resu_dict = covert_one_line_head_and_body(
                        this_reco=this_line,
                        df_key_list=df_key_list,
                    )
                    # print (test_resu_dict)
                    # add a line
                    # top_line = 'Pred_Temp,ID_Sample,ID_Pred,'
                    this_line = str(this_pred_T)+','+str(this_ii_pick)+','+str(ii)+','
                    # 
                    for this_word in df_key_list:
                        this_line += test_resu_dict[this_word]+','+this_resu_dict_GT[this_word]+','
                    this_line = this_line[:-1] + '\n'

                    # deliver 
                    with open(csv_PtoSC_vari_T_set, "a") as f:
                        f.write(this_line)
    else:
        # 
        print (f"CSV file exits")
        
    return 0
#
#
# =============================================================
# 
def predict_at_various_T_1(
    csv_PtoSC_vari_T_set,
    txt_PtoSC_vari_T_set,
    df_key_list,
    added_keys=['Pred_Temp', 'ID_Sample', 'ID_Pred'],
    sample_pick_list=None,
    full_sentence_dict_on_task=None, # sentence_dataset_dict_sepe['test']['Gene001_sentence'],
    input_promp_len=None, # input_len_for_tasks['Gene_PtoSC'],
    pred_temp_list=None,
    n_pred_at_one_time=30,
    # === model prediction ===
    ctx=None,
    model=None,
    tokenizer=None,
    device=None,
    top_k=None,
):
    n_pred_T = len(pred_temp_list)
    
    if not os.path.exists(csv_PtoSC_vari_T_set):
        # 1. handle the top line
        # top_line = 'Pred_Temp,ID_Sample,ID_Pred,'
        top_line = ''
        for this_word in added_keys:
            top_line += this_word+','
        for this_word in df_key_list:
            top_line += this_word+'_PR,'+this_word+'_GT,'
        top_line = top_line[:-1]+'\n'
        # 
        with open(csv_PtoSC_vari_T_set, "w") as f:
            f.write(top_line)
        # 
        n_record_df = 0
    else:
        this_df = pd.read_csv(csv_PtoSC_vari_T_set)
        # need to know which part the of job is done
        n_record_df = len(this_df)
    # # of job to-do: n_sample x n_temp x n_pred
    # now, need to know where we are at this point
    n_finished_sample = n_record_df//(n_pred_at_one_time*n_pred_T)
    n_record_remain = n_record_df%(n_pred_at_one_time*n_pred_T)
    n_finished_T = n_record_remain//n_pred_at_one_time
    n_record_remain = n_record_remain%n_pred_at_one_time
        

        
        #
    # body part
    for i_sample, this_ii_pick in enumerate(sample_pick_list):
        
        if (i_sample+1)>n_finished_sample:
            print (f"Working on sample #: {this_ii_pick}")
            # pick the full
            # full_promp = sentence_dataset_dict_sepe['train']['Gene001_sentence'][this_ii_pick]
            full_promp = full_sentence_dict_on_task[this_ii_pick]
            print (f"Full line (i.e., GT): {full_promp}")
            # test_promp = full_promp[:input_len_for_tasks['Gene_PtoSC']]
            test_promp = full_promp[:input_promp_len]
            print (f"input promp: {test_promp}")
            # get into GT
            # this_resu_dict_GT = TestPack.covert_one_line_head_and_body(
            this_resu_dict_GT = covert_one_line_head_and_body(
                this_reco=full_promp,
                df_key_list=df_key_list,
            )
            print (f"GT: {this_resu_dict_GT}")
        # 
        # make predictions at different T
            for i_pred_temp, this_pred_T in enumerate(pred_temp_list):
                if i_pred_temp+1>n_finished_T: # only need for the 1st time
                    # 
                    n_finished_T=-100 # for next time, every T should be worked on
                    # 
                    # test_line_preds = TrainPack.generate_single_example_AR_LM_w_token_tokenizer(
                    test_line_preds = generate_single_example_AR_LM_w_token_tokenizer(
                        ctx,
                        model,
                        tokenizer,
                        test_promp,
                        device=device,
                        pred_num=n_pred_at_one_time,
                        max_new_tokens=1024-input_promp_len, # 1024-input_len_for_tasks['Gene_PtoSC'],
                        pred_temp=this_pred_T, # 0.1, # 1.,
                        top_k=top_k, # None, # 100,
                    )
                    
                    # 
                    # print and record the results
                    for ii, this_line in enumerate(test_line_preds):
                        # 
                        # if ii+1>n_record_remain:
                        if ii+1>n_record_remain:
                            # 
                            n_record_remain=-100 # onnly needed fro the
                            #                         
                            add_line = "For T={}, Input={}, Sample={}: \n{}\n".format(this_pred_T,this_ii_pick, ii, this_line)
                            # print (add_line)
                            with open(txt_PtoSC_vari_T_set, "a") as ft:
                                ft.write(add_line)
                            # 
                            # explain the results
                            # 
                            # generated recrod
                            # test_resu_dict = TestPack.covert_one_line_head_and_body(
                            test_resu_dict = covert_one_line_head_and_body(
                                this_reco=this_line,
                                df_key_list=df_key_list,
                            )
                            # print (test_resu_dict)
                            # add a line
                            # top_line = 'Pred_Temp,ID_Sample,ID_Pred,'
                            this_line = str(this_pred_T)+','+str(this_ii_pick)+','+str(ii)+','
                            # 
                            for this_word in df_key_list:
                                this_line += test_resu_dict[this_word]+','+this_resu_dict_GT[this_word]+','
                            this_line = this_line[:-1] + '\n'
        
                            # deliver 
                            with open(csv_PtoSC_vari_T_set, "a") as f:
                                f.write(this_line)

        
    return 0
#

# =============================================================
# 
def remove_zeros_PR_lines(
    df_0
):
    # criterion
    df_1 = df_0.loc[(df_0['(Al)_PR']!=0)] 
    # double check
    df_1 = df_1.loc[(df_1['(Al)_GT']!=0)] 
    # clean up
    df_1.reset_index(drop=True, inplace=True)
    
    return df_1
# 
def remove_zeros_PR_lines_based_on_SafeRec(
    df_0
):
    if "SafeRec_PR" in df_0.keys().tolist():
        # PR should be safely recovered
        df_1 = df_0.loc[df_0['SafeRec_PR']==1]
        # add a check on GT: should be OK
        df_1 = df_1.loc[df_1['SafeRec_GT']==1]
        # clean up
        df_1.reset_index(drop=True, inplace=True)
    else:
        print (f"Error in finding SafeRec_PR or SafeRec_GT")
        df_1 = None
    
    return df_1
# 
def clean_and_remove_trivial_lines(
    df,
):
    # convert into numbers
    key_list = df.keys().tolist()
    df_1 = df.copy()
    for this_key in key_list:
        print (f"Working on {this_key}...")
        df_1[this_key] = df_1.apply(
            lambda x: pd.to_numeric( x[this_key], errors='coerce' ),
            axis=1,
        )
        print (f"{this_key}: {infer_dtype(df_1[this_key])}")
    # drop those with NAN
    print ("drop those NANs")
    print ("==============================================")
    print ("Before dropping, len(df): ", len(df_1))
    df_1 = df_1.dropna()
    print ("After dropping, len(df): ", len(df_1))

    # remove the trivial ones
    # if prediction is trivial, no need to keep them
    print (f"Remove zero predictions of (Al)_PR...")
    # this ASSUME safeRec is stored in the csv file.
    print ("==============================================")
    n_1_0 = len(df_1)
    print ("Before dropping, len(df): ", n_1_0)
    
    # df_1 = remove_zeros_PR_lines(df_1)
    df_1 = remove_zeros_PR_lines_based_on_SafeRec(df_1)
    n_1_1 = len(df_1)
    print ("After dropping, len(df): ", n_1_1)
    print (f"Reduce {n_1_0-n_1_1} records.")
    
    return df_1
# 
# ====================================================================
# conunt the safe_rec w.r.t. pred_temp
def check_SafeRec_PR_wrt_Pred_Temp(
    df,
    pred_temp_key='Pred_Temp',
    safe_rec_key='SafeRec_PR',
):
    # find how many pred_temp we have
    unique_T_list = np.array(
        list(set(df[pred_temp_key].values))
    )
    unique_T_list = np.sort(unique_T_list)
    print (f"Find unique T as {unique_T_list}")

    try_num_list = []
    safe_rec_ratio_list = []
    for i_T, this_T in enumerate (unique_T_list):
        # get records of this sample
        this_T_list = df.index[
            (df[pred_temp_key]==this_T)
        ].tolist()
        # check how many 0 and 1
        this_safe_rec_list = df[safe_rec_key].values[this_T_list]
        this_try_num = len(this_safe_rec_list)
        this_safe_ratio = sum(this_safe_rec_list)/this_try_num
        # deliever
        try_num_list.append(this_try_num)
        safe_rec_ratio_list.append(this_safe_ratio)

    resu={}
    resu['pred_temp_list']=unique_T_list
    resu['pred_attempt_num_list']=try_num_list
    resu['recover_ratio_list']=safe_rec_ratio_list
    
    return resu
# 
# ============================================================
# create comp df based on individual elements
def write_one_line(this_file,this_line,open_mode):
    with open(this_file, open_mode) as f:
        f.write(this_line)
        
def expand_elem_comp_to_full_comp(
    indi_elem_comp_list_resu,
    output_csv,
    # default
    elem_orde_list = [
        '(Ni)_PR','(Er)_PR','(Zr)_PR',
        '(Y)_PR','(Yb)_PR'
    ] # this key name is related to parallel modeling code
):
    """
    1. create a csv file with all composition record needed
    2. order of elements as: Ni,Er,Zr,Y,Yb
    3. as initial file, here use Mole % (Modelling code will translate them into wei or whatever needed)
    """
    csv_exist = os.path.exists(output_csv)
    # check if csv file already exists
    if csv_exist:
        print (f"Output csv file already exist. Code stops and use caution...")
    else:
        # to the task
        # 1. top line
        top_line = ''
        for this_elem in elem_orde_list:
            top_line += this_elem+','
        top_line = top_line[:-1]+'\n'
        write_one_line(
            this_file=output_csv,
            this_line=top_line,
            open_mode='w', # erase mode
        )
        # 2. body part
        Ni_List = indi_elem_comp_list_resu['Ni_Mole%_List']
        Er_List = indi_elem_comp_list_resu['Er_Mole%_List']
        Zr_List = indi_elem_comp_list_resu['Zr_Mole%_List']
        Y_List  = indi_elem_comp_list_resu['Y_Mole%_List']
        Yb_List = indi_elem_comp_list_resu['Yb_Mole%_List']
        # get one composition
        for this_Ni in Ni_List:
            for this_Er in Er_List:
                for this_Zr in Zr_List:
                    for this_Y in Y_List:
                        for this_Yb in Yb_List:
                            # 
                            this_line = str(this_Ni)+','+str(this_Er)+','+\
                            str(this_Zr)+','+str(this_Y)+','+str(this_Yb)+'\n'
                            # deliver
                            write_one_line(
                                this_file=output_csv,
                                this_line=this_line,
                                open_mode='a', # erase mode
                            )
            
    
    return output_csv
# 
# =============================================================
# use input comp to finish the sentence
def predict_from_init_comp_to_finish_sentence(
    df_DeNovo_CtoSP_only_comp,
    csv_CtoSP_new_comp_full_ML_prediction,
    txt_CtoSP_new_comp_full_ML_prediction,
    # on model part
    df_key_list,
    key_tail,
    model,
    tokenizer,
    ctx,
    device,
    input_promp_len,
    this_pred_T=1.0,
    top_k=100,
):
    if not os.path.exists(csv_CtoSP_new_comp_full_ML_prediction):
        # 1. handle top line
        top_line=''
        for this_key in df_key_list:
            top_line += (this_key+key_tail+',')
        top_line=top_line[:-1]+'\n'
        # 
        write_one_line(
            this_file=csv_CtoSP_new_comp_full_ML_prediction,
            this_line=top_line,
            open_mode='w', # erase mode
        )
        finished_record = 0
        # 
    else:
        df_out = pd.read_csv(csv_CtoSP_new_comp_full_ML_prediction)
        finished_record = len(df_out)
    
    print (f"Previously finished {finished_record} records...")
    # 2.handle the body part
    for ii in range(len(df_DeNovo_CtoSP_only_comp)):
        if ii>=finished_record:
            print (f"Working on the {ii} record...")
            # build input promp
            test_promp = \
"{{Task:Pred001}}=\
{{Composition:[(Al):{:+.3e},(Ni):{:+.3e},(Er):{:+.3e},\
(Zr):{:+.3e},(Y):{:+.3e},(Yb):{:+.3e}]}}=>\
{{Structure:[".format (
                df_DeNovo_CtoSP_only_comp['(Al)_PR'][ii],
                df_DeNovo_CtoSP_only_comp['(Ni)_PR'][ii],
                df_DeNovo_CtoSP_only_comp['(Er)_PR'][ii],
                df_DeNovo_CtoSP_only_comp['(Zr)_PR'][ii],
                df_DeNovo_CtoSP_only_comp['(Y)_PR'][ii],
                df_DeNovo_CtoSP_only_comp['(Yb)_PR'][ii],
            )
            # do prediction
            test_line_preds = generate_single_example_AR_LM_w_token_tokenizer(
                ctx,
                model,
                tokenizer,
                test_promp,
                device=device,
                pred_num=1,
                max_new_tokens=1024-input_promp_len, # 1024-input_len_for_tasks['Gene_PtoSC'],
                pred_temp=this_pred_T, # 0.1, # 1.,
                top_k=top_k, # None, # 100,
            )
            # only one line
            test_line_preds=test_line_preds[0]

            # ================================================================
            # deliver for txt
            this_line = \
"Record: {}\n{}\n".format(
                ii,
                test_line_preds   
            )
            write_one_line(
                this_file=txt_CtoSP_new_comp_full_ML_prediction,
                this_line=this_line,
                open_mode='a', # erase mode
            )

            # deliver the keys
            test_resu_dict = covert_one_line_head_and_body(
                this_reco=test_line_preds,
                df_key_list=df_key_list,
            )
            this_line = ''
            for this_key in df_key_list:
                this_line += (str(test_resu_dict[this_key])+',')
            this_line = this_line[:-1]+'\n'
            write_one_line(
                this_file=csv_CtoSP_new_comp_full_ML_prediction,
                this_line=this_line,
                open_mode='a', # erase mode
            )
            
        
        
    return 0

# ++
# ======================================================================
#  add for de novo compositon and CtoSP predictions
# ======================================================================
# def merge_Forward_Prediction_with_Modelling(
#     df_Modelling,
#     df_ML,
# ):
#     """
#     The two are assumed to be the same composition list.
#     We add modelling as GT to ML predictions
#     """
    
#     return 0

# group the records
def group_errors_based_on_mutation_distance(
    df,
    x_key,
    dx,
    y_key_list,
):
    x_min = np.amin(df[x_key])
    x_max = np.amax(df[x_key])
    # 
    x_list = np.arange(x_min, x_max, dx)
    x_list = np.append(x_list, [x_max])
    # if np.fabs(x_list[-1]-x_max)<dx:
    #     x_list[-1]=x_max
    # else:
    #     x_list = np.append(x_list, [x_max])
    # # print (x_list)
    # # 
    
        # 
    x_sample_list=[]
    for i_x in range(len(x_list)-1):
        x_0 = x_list[i_x]
        x_1 = x_list[i_x+1]
        x_mid = (x_0+x_1)/2.
        x_sample_list.append(x_mid)
        
    n_y_key = len(y_key_list)
    n_x_sample = len(x_sample_list)
    # 
    # store the results
    y_resu_arr = np.zeros((n_x_sample, n_y_key, 2))
    
    for i_x in range(len(x_list)-1):
        x_0 = x_list[i_x]
        x_1 = x_list[i_x+1]
        x_mid = (x_0+x_1)/2.
        # 
        # pick_idx_list = df.index[ (df[y_key]>=x_0) and (df[y_key]<x_1) ].tolist()
        pick_idx_list = (df[(df[x_key]  >= x_0) & (df[x_key] < x_1)].index.tolist())

        for i_y, y_key in enumerate(y_key_list):
            # 
            y_value_pick = df[y_key].values[pick_idx_list]
            # 
            y_value_mean = np.mean(y_value_pick)
            y_value_std  = np.std(y_value_pick)
            # 
            y_resu_arr[i_x, i_y, 0] = y_value_mean
            y_resu_arr[i_x, i_y, 1] = y_value_std
    
    return x_sample_list, y_resu_arr

# ==============================================================
# try boxplot
def group_datapoints_by_adding_discrete_x_labels(
    df,
    x_key,
    dx,
    adding_grouped_x_key,
):
    x_min = np.amin(df[x_key])
    x_max = np.amax(df[x_key])
    # 
    x_list = np.arange(x_min, x_max, dx)
    x_list = np.append(x_list, [x_max])

    x_sample_list=[]
    for i_x in range(len(x_list)-1):
        x_0 = x_list[i_x]
        x_1 = x_list[i_x+1]
        x_mid = (x_0+x_1)/2.
        x_sample_list.append(x_mid)
        
    # add a new key
    # discrete_x_arr = np.zeros((len(df)))
    discrete_x_arr = np.array(
        ['0.0' for ii in range(len(df))],
        dtype=object
    )

    for i_x in range(len(x_list)-1):
        x_0 = x_list[i_x]
        x_1 = x_list[i_x+1]
        # 
        pick_idx_list = (df[(df[x_key]  >= x_0) & (df[x_key] <= x_1)].index.tolist())
        # discrete_x_arr[pick_idx_list]=x_sample_list[i_x]
        discrete_x_arr[pick_idx_list]="%.1f" % (x_sample_list[i_x])

    if adding_grouped_x_key in df.keys().tolist():
        print (f"{adding_grouped_x_key} already exists. Overwrite now...")
        pass
    else:
        print (f"add a discrete {x_key} key: {adding_grouped_x_key}")
        
    df[adding_grouped_x_key] = discrete_x_arr

    return df
        


