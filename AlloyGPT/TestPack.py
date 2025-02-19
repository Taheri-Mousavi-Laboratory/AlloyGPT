import os
import math
import re
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
# ++
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib.offsetbox import AnchoredText

def convert_one_line (this_reco, target_key_list):
    # # for debug
    # print (this_reco)
    # print ()
    
    try: 
        #
        data_blocks = re.findall(r"\[.*?\]", this_reco)
        # data_blocks = re.findall(r'\[([^]]*)\]', this_reco)
        # print (data_blocks)
        resu_dict = {}
        for ii, this_block in enumerate(data_blocks):
            # handle one block
            # 1. get rid of the braket
            this_block = this_block[1:-1]
            # 2. collect keys and values
            this_KV_pairs = this_block.split(',')
            for jj, this_pair in enumerate(this_KV_pairs):
                this_resu = this_pair.split(':')
                resu_dict[this_resu[0]] = this_resu[1]
        #
        resu_dict['SafeRec'] = '1'
        # 
        # check the completeness
        # 
        for this_key in target_key_list:
            assert this_key in resu_dict.keys()
            # ++
            
    except:
        # 
        # ++ for debug
        print ("Cannot translate the line:")
        print (this_reco)
        print ()
        
        resu_dict = {}
        for this_key in target_key_list:
            resu_dict[this_key] = '0.0'
        # 
        resu_dict['SafeRec'] = '0'
    
            
    return resu_dict


def covert_one_line_head_and_body(
    this_reco,
    df_key_list,
):
    # 
    # 1. break into head and body
    temp_reco = this_reco.split('}={')
    head_reco = temp_reco[0][1:]
    body_reco = temp_reco[1]
    # 2. handle the head
    head_dict = {}
    temp_head = head_reco.split(":")
    head_dict[temp_head[0]]=temp_head[1]
    # 3. handle the body
    body_dict = convert_one_line(body_reco,df_key_list)
    # 4. merge the two
    resu_dict = {**head_dict, **body_dict}
    
    return resu_dict

# =============================================================
def Pred_on_Sentence_Test_Set_Per_Task(
    csv_out_file, #  = Task_Pred_on_TestSet_csv_short_inference
    df_key_list,
    sentence_task_dataloader, # =sentence_PRED_test_dataloader 
    batch_size, 
    target_batch, # = 10
    input_str_len, # =input_len_for_tasks['Pred_CtoSP']
    complete_sent_len, #  = 1024
    # on the model fun
    ctx,
    model,
    tokenizer,
    device,
    pred_num = 1,
    top_k=100,
    top_p=0.9, # not used
    pred_temp=1.,
):
    # ---------------------------------------------------
    # for new file
    if not os.path.exists(csv_out_file):
        # create file and top line
        top_line = 'Task,'
        for this_key in df_key_list:
            top_line = top_line+this_key+'_PR,'+this_key+'_GT,'
        top_line = top_line[:-1]+'\n'
        with open(csv_out_file, "w") as f:
            f.write(top_line)
        # start from the begining
        # iter_prev = 0
        finished_batch = 0
    else:
        # for exisiting file, need to know how many iterations have been done already
        # 
        df_prev = pd.read_csv(csv_out_file)
        # # --
        # iter_prev = math.ceil(len(df_prev)/batch_size)
        # ++
        finished_batch = math.ceil(len(df_prev)/batch_size)
        
    print ("The past runs have finished %d records \n" % (finished_batch))
    print ("In total there are about %d mini-batches\n" % (len(sentence_task_dataloader)))

    for this_batch_id, this_item in enumerate(sentence_task_dataloader):
        if this_batch_id+1 > finished_batch and this_batch_id<=target_batch : # for debug
            print ('Working on %d batch' % (this_batch_id))
            n_record = len(this_item)
            # 0. truncate the input sentences as input prompts
            batch_list = []
            for this_line in this_item:
                add_line = this_line[:input_str_len]
                batch_list.append(add_line)
            # 1. tokneization
            batch_token_pack  = tokenizer.encode_batch(
                input=batch_list,
                add_special_tokens=False,
            )
            this_batch_ids = []
            for this_token_pack in batch_token_pack:
                this_ids = [
                    this_ids for (this_ids, this_a_mask) in \
                    zip(this_token_pack.ids, this_token_pack.attention_mask) \
                    if this_a_mask == 1 
                ]
                this_batch_ids.append(this_ids)
            # convert it into torch.tensor
            this_batch_ids = torch.tensor(this_batch_ids).to(device)
            # 2. make predictions
            model.eval()
            with torch.no_grad():
                with ctx:
                    # for k in range(TestKeys['num_samples']):
                    this_batch_gene = model.generate(
                        idx=this_batch_ids, 
                        max_new_tokens=complete_sent_len - input_str_len, # TestKeys['max_new_tokens'], 
                        temperature=pred_temp, # TestKeys['temperature'], # temperature, # TestKeys['temperature'], 
                        top_k=top_k, # TestKeys['top_k'], # top_k, # TestKeys['top_k']
                    )
            
            
            # this_batch_gene = model.generate(
            #     inputs=this_batch_ids,
            #     eos_token_id=tokenizer.token_to_id('</s>'),
            #     max_length=complete_sent_len,
            #     # The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.
            #     do_sample=True,
            #     # If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.
            #     top_k=top_k, # defaults to 50
            #     # The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Defau
            #     top_p=top_p, 
            #     # If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation
            #     # The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.
            #     num_return_sequences=pred_num, # 3, # 1,
            #     temperature=pred_temp, 
            #     # (higher temperature => more likely to sample low probability tokens)
            #     # if temperature != 1.0:
            #     #             scores = scores / temperature
            #     use_cache=True,
            # )
            # 3. decode back to sentence list
            # translate back into sentences
            this_batch_sentence_pred = tokenizer.decode_batch(
                sequences=this_batch_gene.tolist(),
                skip_special_tokens=True,
            )
            # 4. convert the records into csv file lines
            for i_reco in range(n_record):
                # convert into dict: assume this one is error free
                this_resu_dict_GT = covert_one_line_head_and_body(
                    this_item[i_reco],
                    df_key_list
                )
                this_resu_dict_PR = covert_one_line_head_and_body(
                    this_batch_sentence_pred[i_reco],
                    df_key_list
                )

                # write this one into the files
                # add a new line
                this_line = this_resu_dict_GT['Task']+','
                for this_key in df_key_list:
                    this_line = this_line + this_resu_dict_PR[this_key] + ',' \
                    + this_resu_dict_GT[this_key] + ','
                this_line = this_line[:-1]+'\n'
                # print (this_line)
                
                with open(csv_out_file, "a") as f:
                    f.write(this_line)
                    
        elif this_batch_id>target_batch :
            print ("Reach to the limit.")
            break
            
# ================================================================================ 
def check_one_line_if_df(
    df,
    key_list,
    ii,
):
    diff_list = []
    for this_key in key_list:
        this_diff = df[this_key+'_GT'][ii]-df[this_key+'_PR'][ii]
        diff_list.append(this_diff)
    
    return diff_list

# ==========================================================================
# 
from pandas.api.types import infer_dtype
# 
def read_and_convert_format(
    csv_file,
    key_list,
):
    """
    read in and convert SOME keys into numbers
    """
    df = pd.read_csv(csv_file)
    print (df.head(2))
    # 1. check the format
    keys_need_to_fix = []
    # for this_key in key_list:
    #     this_key_pr = this_key+'_PR'
    #     this_key_gt = this_key+'_GT'
    #     dtype_pr = infer_dtype(df[this_key_pr])
    #     dtype_gt = infer_dtype(df[this_key_gt])
    #     # 
    #     if dtype_pr!='floating':
    #         print (f"mistaken key: {this_key_pr}, type: {dtype_pr}")
    #         keys_need_to_fix.append(this_key_pr)
    #     if dtype_gt!='floating':
    #         print (f"mistaken key: {this_key_gt}, type: {dtype_gt}")
    #         keys_need_to_fix.append(this_key_gt)
    for this_key_base in key_list:
        for this_key in [this_key_base+'_PR', this_key_base+'_GT']:
            this_type = infer_dtype(df[this_key])
            if this_type != 'floating':
                print (f"mistaken key: {this_key}, type: {this_type}")
                keys_need_to_fix.append(this_key)
        # 
    # 2. do fix
    this_df = df.copy()
    # do a quick fix
    for this_key in keys_need_to_fix:
        print (f"working on {this_key}")
        this_df[this_key] = this_df.apply(
            lambda x: pd.to_numeric( x[this_key], errors='coerce' ),
            axis=1,
        )
    # 3. recheck
    print ("Re-check the keys, should be float.")
    print ("=======================")
    # check
    for this_key_base in key_list:
        for this_key in [this_key_base+'_PR', this_key_base+'_GT']:
            print (f"{this_key}: {infer_dtype(this_df[this_key])}")
    # drop those with nan
    print ("drop those NANs")
    print ("=======================")
    print ("Before dropping, len(df): ", len(this_df))
    this_df = this_df.dropna()
    print ("After dropping, len(df): ", len(this_df))
    
        

    return this_df
# ++
# clean all keys
def clean_df_row_format(
    df,
    key_root_list,
):
    # we only want the C-S-P keys to be float values
    keys_need_to_fix = []
    for this_key_root in key_root_list:
        for this_key in [this_key_root+'_PR', this_key_root+'_GT']:
            this_type = infer_dtype(df[this_key])
            if this_key != 'floating':
                print (f"mistaken key: {this_key}, type: {this_type}")
                keys_need_to_fix.append(this_key)
    # 2. do fix
    this_df = df.copy()
    # do a quick fix
    for this_key in keys_need_to_fix:
        print (f"working on {this_key}")
        this_df[this_key] = this_df.apply(
            lambda x: pd.to_numeric( x[this_key], errors='coerce' ),
            axis=1,
        )
    # 3. recheck
    print ("Re-check the keys, should be float.")
    print ("=======================")
    # check
    for this_key_root in key_root_list:
        for this_key in [this_key_root+'_PR', this_key_root+'_GT']:
            print (f"{this_key}: {infer_dtype(this_df[this_key])}")
    # drop those with nan
    print ("drop those NANs")
    print ("=======================")
    print ("Before dropping, len(df): ", len(this_df))
    this_df = this_df.dropna()
    print ("After dropping, len(df): ", len(this_df))
        
    return this_df
# 
# make into a function
def plot_one_record(
    ii,
    sslabels,
    df_task,
    title_text,
):
    stru_prop_array_gt=[]
    stru_prop_array_pr=[]
    for this_key in sslabels:
        stru_prop_array_gt.append(
            df_task[this_key+'_GT'][ii]
        )
        stru_prop_array_pr.append(
            df_task[this_key+'_PR'][ii]
        )

    # ['L12MolePC', 'ScheilL12MeltMolePC', 'ScheilL12MolePC', 'ScheilTerneryMeltMolePC', 'ScheilTernaryMolePC', 'ScheilAl3NiMeltMolePC', 'ScheilAl3NiMolePC', 'ScheilAl3ZrMeltMolePC', 'ScheilAl3ZrMolePC', 'BulkResistivity', 'Misfit', 'CoarseningRate', 'ScheilFRCutoff', 'ScheilFRMatrix', 'ScheilCSC', 'ScheilHCS']

    x=np.linspace (0, len(sslabels)-1, len(sslabels))

    fig, ax = plt.subplots(1, 1, figsize=(6,3))

    ax.bar(x-0.15, stru_prop_array_gt, width=0.3, color='b', align='center')
    ax.bar(x+0.15, stru_prop_array_pr, width=0.3, color='r', align='center')

    # ax.set_ylim([0, 1])

    plt.xticks(range(len(sslabels)), sslabels, size='medium', rotation=90)
    plt.legend (['GT','Prediction'])

    plt.ylabel ('stru-prop results')
    plt.title(title_text)
    plt.show()

    # return 0

# ============================================================
# 
# a loop

def plot_loop_all_keys(
    this_df,
    key_list,
    task_type,
    #
    PC_not_show = 2,
):
    r2_list = []
    mse_list = []
    mae_list = []
    # normalized mean abs error
    nmae_list_1 = [] # normalized by the mean of true values
    nmae_list_2 = [] # normalized by the range of true values
    
    for this_key in key_list:
    # this_key = Forward_Resu_Keys[ii]

        # 
        fig = plt.figure(figsize=(24,16),dpi=200)
        fig, ax0 = plt.subplots()

        ax0.scatter(
            this_df[this_key+'_GT'], 
            this_df[this_key+'_PR'], 
            color='red',
            alpha=0.01,
        )
        x_max = np.amax(
            [max(this_df[this_key+'_GT']),
             max(this_df[this_key+'_PR'])]
        )
        x_min = np.amin(
            [min(this_df[this_key+'_GT']),
             min(this_df[this_key+'_PR'])]
        )
        x_mid = (x_max+x_min)/2.
        # ++
        # PC_not_show = 2
        x0_GT = np.percentile(this_df[this_key+'_GT'], PC_not_show, axis=0)
        x1_GT = np.percentile(this_df[this_key+'_GT'], 100-PC_not_show, axis=0)
        x0_PR = np.percentile(this_df[this_key+'_PR'], PC_not_show, axis=0)
        x1_PR = np.percentile(this_df[this_key+'_PR'], 100-PC_not_show, axis=0)
        x0 = np.amin([x0_GT, x0_PR])
        x1 = np.amax([x1_PR, x1_PR])
        ax0.plot(
            [x0,x1],
            [x0,x1],
            linestyle='dotted',
            color='b'
        )

        this_r2 = r2_score(
            y_true=this_df[this_key+'_GT'], 
            y_pred=this_df[this_key+'_PR']
        )
        this_mse = mean_squared_error(
            y_true=this_df[this_key+'_GT'], 
            y_pred=this_df[this_key+'_PR']
        )
        this_mae = mean_absolute_error(
            y_true=this_df[this_key+'_GT'], 
            y_pred=this_df[this_key+'_PR']
        )
        # ++
        # this_mape = mean_absolute_percentage_error(
        #     y_true=this_df[this_key+'_GT'], 
        #     y_pred=this_df[this_key+'_PR'],
        # )
        # https://d2uars7xkdmztq.cloudfront.net/app_resources/38997/documentation/135539_en.pdf
        this_nmae_1 = this_mae/np.mean(this_df[this_key+'_GT'])
        this_nmae_2 = this_mae/(
            np.amax(this_df[this_key+'_GT'])-np.amin(this_df[this_key+'_GT'])
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
        # ++
        # print(f"mape of %s: %10.3f" % (
        #         '{0: >25}'.format(this_key), 
        #           this_mape
        #     )
        # )
        print(f"nmae_1 of %s: %10.3f" % (
                '{0: >25}'.format(this_key), 
                  this_nmae_1
            )
        )
        print(f"nmae_2 of %s: %10.3f" % (
                '{0: >25}'.format(this_key), 
                  this_nmae_2
            )
        )
        
        r2_list.append(this_r2)
        mse_list.append(this_mse)
        mae_list.append(this_mae)
        # mape_list.append(this_mape)
        nmae_list_1.append(this_nmae_1)
        nmae_list_2.append(this_nmae_2)
        # 
        # plt.text()
        anchored_text = AnchoredText(
            # f"r2: {this_r2}\nMSE: {this_mse}\nMAE: {this_mae}", 
            f"R2:   %10.3f\nMSE: %10.3f\nMAE: %10.3f\nNMAE_1: %10.3f%%\nNMAE_2: %10.3f%%" % (
                this_r2, this_mse, this_mae, this_nmae_1*100, this_nmae_2*100
            ), 
            loc=2
        )
        anchored_text.patch.set_alpha(0.1)
        ax0.add_artist(anchored_text)

        plt.title(task_type +this_key)
        plt.xlabel('GT')
        plt.ylabel('Prediction')
        plt.xlim([x0, x1])
        plt.ylim([x0, x1])
        plt.show()
        plt.close()
    return r2_list, mse_list, mae_list, nmae_list_1, nmae_list_2
# 
def plot_one_key(
    this_df,
    key_list,
    task_type,
):
    # r2_list = []
    # mse_list = []
    # mae_list = []
    for this_key in key_list:
    # this_key = Forward_Resu_Keys[ii]

        # 
        fig = plt.figure(figsize=(24,16),dpi=200)
        fig, ax0 = plt.subplots()

        ax0.scatter(
            this_df[this_key+'_GT'], 
            this_df[this_key+'_PR'], 
            color='red',
            alpha=0.01,
        )
        x1 = np.amax(
            [max(this_df[this_key+'_GT']),
             max(this_df[this_key+'_PR'])]
        )
        x0 = np.amin(
            [min(this_df[this_key+'_GT']),
             min(this_df[this_key+'_PR'])]
        )
        ax0.plot(
            [x0,x1],
            [x0,x1],
            linestyle='dotted',
            color='b'
        )

        this_r2 = r2_score(
            y_true=this_df[this_key+'_GT'], 
            y_pred=this_df[this_key+'_PR']
        )
        this_mse = mean_squared_error(
            y_true=this_df[this_key+'_GT'], 
            y_pred=this_df[this_key+'_PR']
        )
        this_mae = mean_absolute_error(
            y_true=this_df[this_key+'_GT'], 
            y_pred=this_df[this_key+'_PR']
        )

        # print(f"r2 of %s: %10.3f" % (
        #         '{0: >25}'.format(this_key), 
        #           this_r2
        #     )
        # )
        # print(f"mse of %s: %10.3f" % (
        #         '{0: >25}'.format(this_key), 
        #           this_mse
        #     )
        # )
        # print(f"mae of %s: %10.3f" % (
        #         '{0: >25}'.format(this_key), 
        #           this_mae
        #     )
        # )
        # r2_list.append(this_r2)
        # mse_list.append(this_mse)
        # mae_list.append(this_mae)
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

        plt.title(task_type +this_key)
        plt.xlabel('GT')
        plt.ylabel('Prediction')
        # plt.xlim([-60, 500])
        # plt.ylim([-60, 500])
        plt.show()
        plt.close()
    # return 0
# =======================================================
def plot_one_key_1(
    this_df,
    this_key,
    task_type,
    # for out-lier
    n_extreme=2,
    alpha_point=0.1,
    key_ends=['_GT', '_PR'],
    x_y_label=['GT', 'Prediction'],
):
    # r2_list = []
    # mse_list = []
    # mae_list = []
    # for this_key in key_list:
    # this_key = Forward_Resu_Keys[ii]

    # 
    fig = plt.figure(figsize=(24,16),dpi=200)
    # fig, ax0 = plt.subplots()

  
    # x_max = np.amax(
    #     [max(this_df[this_key+'_GT']),
    #      max(this_df[this_key+'_PR'])]
    # )
    # x_min = np.amin(
    #     [min(this_df[this_key+'_GT']),
    #      min(this_df[this_key+'_PR'])]
    # )
    # x_mid = (x_max+x_min)/2.
    # x_r = ()

    x_1 = np.sort( this_df[this_key+key_ends[0]])[-n_extreme]
    x_2 = np.sort( this_df[this_key+key_ends[1]])[-n_extreme]
    x_3 = -np.sort(-this_df[this_key+key_ends[0]])[-n_extreme]
    x_4 = -np.sort(-this_df[this_key+key_ends[1]])[-n_extreme]
    x1 = np.amax(
        [x_1, x_2]
    )
    x0 = np.amin(
        [x_3, x_4]
    )
    x_r = (x1-x0)/2.

        # ++
    a = sns.jointplot(
        data=this_df,
        x=this_key+key_ends[0],
        y=this_key+key_ends[1],
        xlim=[x0-x_r*0.05, x1+x_r*0.05],
        ylim=[x0-x_r*0.05, x1+x_r*0.05],
        color='red',
        space=0,
        joint_kws = dict(alpha=alpha_point)
    )
    ax0 = a.ax_joint
    # ax0.scatter(
    #     this_df[this_key+'_GT'], 
    #     this_df[this_key+'_PR'], 
    #     marker='+',
    #     color='red',
    #     alpha=0.05,
    # )
    ax0.plot(
        [x0,x1],
        [x0,x1],
        linestyle='dotted',
        color='b'
    )

    this_r2 = r2_score(
        y_true=this_df[this_key+key_ends[0]], 
        y_pred=this_df[this_key+key_ends[1]]
    )
    this_mse = mean_squared_error(
        y_true=this_df[this_key+key_ends[0]], 
        y_pred=this_df[this_key+key_ends[1]]
    )
    this_mae = mean_absolute_error(
        y_true=this_df[this_key+key_ends[0]], 
        y_pred=this_df[this_key+key_ends[1]]
    )

    # print(f"r2 of %s: %10.3f" % (
    #         '{0: >25}'.format(this_key), 
    #           this_r2
    #     )
    # )
    # print(f"mse of %s: %10.3f" % (
    #         '{0: >25}'.format(this_key), 
    #           this_mse
    #     )
    # )
    # print(f"mae of %s: %10.3f" % (
    #         '{0: >25}'.format(this_key), 
    #           this_mae
    #     )
    # )
    # r2_list.append(this_r2)
    # mse_list.append(this_mse)
    # mae_list.append(this_mae)
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

    # plt.title(task_type +this_key)
    plt.suptitle(task_type +this_key, y=1)
    plt.xlabel(x_y_label[0])
    plt.ylabel(x_y_label[1])
    # plt.xlim([-60, 500])
    # plt.ylim([-60, 500])
    plt.show()
    plt.close()

    # return x0, x1
        
# =======================================================
# only calculate
# 
def analyze_loop_for_subkeys_in_all_keys(
    this_df,
    this_key_list,
    full_key_list,
    base_value,
):
    # initialize all keys with base value
    r2_list = np.ones(len(full_key_list))*base_value
    mse_list = np.ones(len(full_key_list))*base_value
    mae_list = np.ones(len(full_key_list))*base_value
    # ++
    nmae_list_1 = np.ones(len(full_key_list))*base_value
    nmae_list_2 = np.ones(len(full_key_list))*base_value
    
    # modify those in this_key_list
    for i_sub, this_key in enumerate(this_key_list):
        i_full_list = [i for i, x in enumerate(full_key_list) if x == this_key]
        if len(i_full_list)==1:
            # pass
            this_r2 = r2_score(
                y_true=this_df[this_key+'_GT'], 
                y_pred=this_df[this_key+'_PR']
            )
            this_mse = mean_squared_error(
                y_true=this_df[this_key+'_GT'], 
                y_pred=this_df[this_key+'_PR']
            )
            this_mae = mean_absolute_error(
                y_true=this_df[this_key+'_GT'], 
                y_pred=this_df[this_key+'_PR']
            )
            # assign
            r2_list[i_full_list[0]] = this_r2
            mse_list[i_full_list[0]] = this_mse
            mae_list[i_full_list[0]] = this_mae
            # 
            nmae_list_1[i_full_list[0]] = this_mae/np.mean(this_df[this_key+'_GT'])
            nmae_list_2[i_full_list[0]] = this_mae/(
                np.amax(this_df[this_key+'_GT'])-np.amin(this_df[this_key+'_GT'])
            )
            
        else: 
            print (f"{this_key} is missing")
    

    return r2_list, mse_list, mae_list, nmae_list_1, nmae_list_2
# ========================================================
# 
def pick_save_recover(df_0):
    print (f"ori len: {len(df_0)}")
    # pick only those that ['SafeRec_PR'] and ['SafeRec_GT'] == 1
    df = df_0.loc[(df_0['SafeRec_PR']==1) & (df_0['SafeRec_GT']==1)]
    df = df.reset_index(drop=True)
    print (f"new len: {len(df)}")
    return df
# =========================================================
# 
# 
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html
def plot_compare_bar_plots(
    sslabels,
    value_list_1,
    value_list_2,
    label_1,
    label_2,
    y_label,
    # 
    legend_loc='best'
):
    
    # sslabels = Forward_Short_Resu_Keys

    x=np.linspace (0, len(sslabels)-1, len(sslabels))

    fig, ax = plt.subplots(1, 1, figsize=(6,3))

    ax.bar(x-0.15, value_list_1, width=0.3, color='b', align='center')
    ax.bar(x+0.15, value_list_2, width=0.3, color='r', align='center')

    # ax.set_ylim([0, 1])

    plt.xticks(range(len(sslabels)), sslabels, size='medium', rotation=90)
    plt.legend ([label_1, label_2], loc=legend_loc)

    plt.ylabel (y_label)
    plt.show()