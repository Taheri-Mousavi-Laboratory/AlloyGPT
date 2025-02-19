from transformers import ByT5Tokenizer
# 
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import decoders
#
from datasets import (
    load_dataset,
    load_from_disk,
    load_dataset_builder,
    get_dataset_split_names,
    DatasetDict,
)
# =========================================================
# 
def data_dict_seperate_train_vali_test(
    dataset_dict,
    train_ratio,
    vali_ratio,
    if_seed=True,
    seed=1234,
):
    '''
    trai + vali for training process
    test for final standalone test
    '''
    if if_seed:
        trai_vali_test = dataset_dict['train'].train_test_split(
            train_size=train_ratio,
            seed=seed,
        )

        size_vali_test = 1.-train_ratio
        vali_test = trai_vali_test['test'].train_test_split(
            train_size=vali_ratio/size_vali_test,
            seed=seed,
        )
    else:
        trai_vali_test = dataset_dict['train'].train_test_split(
            train_size=train_ratio,
        )

        size_vali_test = 1.-train_ratio
        vali_test = trai_vali_test['test'].train_test_split(
            train_size=vali_ratio/size_vali_test,
        )
    
    # deliver the result
    dataset_sepe = DatasetDict({
        'train': trai_vali_test['train'],
        'valid': vali_test['train'],
        'test': vali_test['test']
    })
    return dataset_sepe 


# =========================================================
# 
def build_tokenizer(
    tokenizer_type,
    tokenizer_file=None,
    seq_len=None,
):
    tokenizer=None
    
    if tokenizer_type=='Byte_Tokenizer':
        '''
        vanilia unift-8 code encoder: from transformers
        '''
        tokenizer = ByT5Tokenizer(
            eos_token='</s>', # use default
            unk_token='<unk>',
            pad_token='</s>',
            extra_ids=25, # use a small one to save the model size, # default 125
        )
        # add some checks
        print (tokenizer)
        print (len(tokenizer))
        print (tokenizer.vocab_size)
        print (len(tokenizer.additional_special_tokens))
        # utf-8 + 25 = tokenizer.voc
        print (tokenizer.vocab_size+len(tokenizer.additional_special_tokens))
        # 
    elif tokenizer_type=='BPE_customerized_0':
        '''
        customized based on the alloy language
        Load directly from a file, built on tokenizers
        ref: 6_...
        '''
        tokenizer=Tokenizer.from_file(tokenizer_file)
        # initalize it
        tokenizer.enable_padding(
            pad_id=0,
            pad_token='</s>',
            length=seq_len,
        )
        # add decoder
        tokenizer.decoder = decoders.BPEDecoder()
        
    else:
        print ("tokenizer type not found!")
    return tokenizer
# =========================================================
# recover tokenizers
def save_tokenizer(
    DataKeys,
    tokenizer,
):
    if DataKeys['tokenizer_type']=='Byte_Tokenizer':
        """
        from transformers
        """
        tokenizer.save_pretrained(
            DataKeys['tokenizer_dir']
        )
    elif DataKeys['tokenizer_type']=='BPE_customerized_0':
        """
        from tokenizers
        """
        tokenizer.save(
            DataKeys['tokenizer_dir'] + '/tokenizer.json'
        )
    else:
        print ("None tokenizer found...")
    return 0
# 
def reload_tokenizer(
    DataKeys,
):
    tokenizer = None
    if DataKeys['tokenizer_type']=='Byte_Tokenizer':
        """
        from transformers
        """
        tokenizer = AutoTokenizer.from_pretrained(
            DataKeys['tokenizer_dir']
        )
    elif DataKeys['tokenizer_type']=='BPE_customerized_0':
        """
        from tokenizers
        """
        tokenizer_file = DataKeys['tokenizer_dir'] + '/tokenizer.json'
        tokenizer=Tokenizer.from_file(tokenizer_file)
    else:
        print ("None tokenizer found...")
        
    return tokenizer
# =========================================================
# updated one, copied from 
# =========================================================
# 2_sbatch_.../7_2_.../
def assemble_one_sentence(
    element, # raw recrod: all keys in
    sentence_id, # to know which sentence to make
):
    '''
    convert csv row into text
    updated with the new tokenizer
    '''
    this_line = None
    if   sentence_id == "Pred001":
        # Type: forward predicting 1
        # id: Pred001
#         this_line = \
#         "{{Task:Pred001}}=\
# {{Composition:[(Al):{:+.3e},(Ni):{:+.3e},(Er):{:+.3e},\
# (Zr):{:+.3e},(Y):{:+.3e},(Yb):{:+.3e}]}}=>\
# {{Structure:[L12MolePC:{:+.3e},ScheilL12MeltMolePC:{:+.3e},ScheilL12MolePC:{:+.3e},\
# ScheilTerneryMeltMolePC:{:+.3e},ScheilTernaryMolePC:{:+.3e},ScheilAl3NiMeltMolePC:{:+.3e},\
# ScheilAl3NiMolePC:{:+.3e},ScheilAl3ZrMeltMolePC:{:+.3e},ScheilAl3ZrMolePC:{:+.3e}]}}=>\
# {{Property:[BulkResistivity:{:+.3e},Misfit:{:+.3e},CoarseningRate:{:+.3e},\
# ScheilFRCutoff:{:+.3e},ScheilFRMatrix:{:+.3e},ScheilCSC:{:+.3e},\
# ScheilHCS:{:+.3e}]}}".format (
#             # composition
#             element['AlMolePC'],
#             element['NiMolePC'],
#             element['ErMolePC'],
#             element['ZrMolePC'],
#             element['YMolePC'],
#             element['YbMolePC'],
#             # structure numbers
#             element['L12MolePC'],
#             element['ScheilL12MeltMolePC'],
#             element['ScheilL12MolePC'],
#             element['ScheilTerneryMeltMolePC'],
#             element['ScheilTernaryMolePC'],
#             element['ScheilAl3NiMeltMolePC'],
#             element['ScheilAl3NiMolePC'],
#             element['ScheilAl3ZrMeltMolePC'],
#             element['ScheilAl3ZrMolePC'],
#             # properties
#             element['BulkResistivity'],
#             element['Misfit'],
#             element['CoarseningRate'],
#             element['ScheilFRCutoff'],
#             element['ScheilFRMatrix'],
#             element['ScheilCSC'],
#             element['ScheilHCS'],
#         )
        #
        # for the remaked datset
        # 
        this_line = \
        "{{Task:Pred001}}=\
{{Composition:[(Al):{:+.3e},(Ni):{:+.3e},(Er):{:+.3e},\
(Zr):{:+.3e},(Y):{:+.3e},(Yb):{:+.3e}]}}=>\
{{Structure:[AsBuilt_L12Mol%:{:+.3e},AsBuilt_TernaryMol%:{:+.3e},\
AsBuilt_Al3NiMol%:{:+.3e},AsBuilt_Al3ZrMol%:{:+.3e},\
L12Mol%:{:+.3e},TernaryMol%:{:+.3e},Al3NiMol%:{:+.3e},\
Al3ZrMol%:{:+.3e}]}}=>\
{{Property:[DiffusionResistivity:{:+.3e},Misfit:{:+.3e},CoarseningMetric:{:+.3e},\
FreezingRange:{:+.3e},CrackSusceptibilityCoefficient:{:+.3e},\
HotCrackingSusceptibility:{:+.3e}]}}".format (
            # composition
            element['AlMolePC'],
            element['NiMolePC'],
            element['ErMolePC'],
            element['ZrMolePC'],
            element['YMolePC'],
            element['YbMolePC'],
            # structure numbers
            # as-built
            element['ScheilL12MolePC'],
            element['ScheilTernaryMolePC'],
            element['ScheilAl3NiMolePC'],
            element['ScheilAl3ZrMolePC'],
            # aged
            element['L12MolePC'],
            element['TerneryMolePC'],
            element['Al3NiMolePC'],
            element['Al3ZrMolePC'],
            # properties
            # dimension less
            element['NZ_BulkResistivity'],
            element['NZ_Misfit'],
            element['NZ_CoarseningMetric'],
            element['NZ_Freezing_Range_From_fccAl'],
            element['CSC'],
            element['NZ_HCS'],
        )
    elif sentence_id == "Gene001":
        # for 
#         this_line = \
#         "{{Task:Gene001}}=\
# {{Property:[BulkResistivity:{:+.3e},Misfit:{:+.3e},CoarseningRate:{:+.3e},\
# ScheilFRCutoff:{:+.3e},ScheilFRMatrix:{:+.3e},ScheilCSC:{:+.3e},\
# ScheilHCS:{:+.3e}]}}=>\
# {{Structure:[L12MolePC:{:+.3e},ScheilL12MeltMolePC:{:+.3e},ScheilL12MolePC:{:+.3e},\
# ScheilTerneryMeltMolePC:{:+.3e},ScheilTernaryMolePC:{:+.3e},ScheilAl3NiMeltMolePC:{:+.3e},\
# ScheilAl3NiMolePC:{:+.3e},ScheilAl3ZrMeltMolePC:{:+.3e},ScheilAl3ZrMolePC:{:+.3e}]}}=>\
# {{Composition:[(Al):{:+.3e},(Ni):{:+.3e},(Er):{:+.3e},(Zr):{:+.3e},(Y):{:+.3e},\
# (Yb):{:+.3e}]}}".format (
#             # properties
#             element['BulkResistivity'],
#             element['Misfit'],
#             element['CoarseningRate'],
#             element['ScheilFRCutoff'],
#             element['ScheilFRMatrix'],
#             element['ScheilCSC'],
#             element['ScheilHCS'],
#             # structure numbers
#             element['L12MolePC'],
#             element['ScheilL12MeltMolePC'],
#             element['ScheilL12MolePC'],
#             element['ScheilTerneryMeltMolePC'],
#             element['ScheilTernaryMolePC'],
        #     element['ScheilAl3NiMeltMolePC'],
        #     element['ScheilAl3NiMolePC'],
        #     element['ScheilAl3ZrMeltMolePC'],
        #     element['ScheilAl3ZrMolePC'],

        #     # composition
        #     element['AlMolePC'],
        #     element['NiMolePC'],
        #     element['ErMolePC'],
        #     element['ZrMolePC'],
        #     element['YMolePC'],
        #     element['YbMolePC'],

        # )
        this_line = \
        "{{Task:Gene001}}=\
{{Property:[DiffusionResistivity:{:+.3e},Misfit:{:+.3e},CoarseningMetric:{:+.3e},\
FreezingRange:{:+.3e},CrackSusceptibilityCoefficient:{:+.3e},\
HotCrackingSusceptibility:{:+.3e}]}}=>\
{{Structure:[AsBuilt_L12Mol%:{:+.3e},AsBuilt_TernaryMol%:{:+.3e},\
AsBuilt_Al3NiMol%:{:+.3e},AsBuilt_Al3ZrMol%:{:+.3e},\
L12Mol%:{:+.3e},TernaryMol%:{:+.3e},Al3NiMol%:{:+.3e},\
Al3ZrMol%:{:+.3e}]}}=>\
{{Composition:[(Al):{:+.3e},(Ni):{:+.3e},(Er):{:+.3e},(Zr):{:+.3e},(Y):{:+.3e},\
(Yb):{:+.3e}]}}".format (
            # properties
            # dimension less
            element['NZ_BulkResistivity'],
            element['NZ_Misfit'],
            element['NZ_CoarseningMetric'],
            element['NZ_Freezing_Range_From_fccAl'],
            element['CSC'],
            element['NZ_HCS'],
            # structure numbers
            # as-built
            element['ScheilL12MolePC'],
            element['ScheilTernaryMolePC'],
            element['ScheilAl3NiMolePC'],
            element['ScheilAl3ZrMolePC'],
            # aged
            element['L12MolePC'],
            element['TerneryMolePC'],
            element['Al3NiMolePC'],
            element['Al3ZrMolePC'],
            # composition
            element['AlMolePC'],
            element['NiMolePC'],
            element['ErMolePC'],
            element['ZrMolePC'],
            element['YMolePC'],
            element['YbMolePC'],
)
    else:
        print ("Sentence type is not known")
    return this_line
    
# old one
def assemble_one_sentence_old(
    element, # raw recrod: all keys in
    sentence_id, # to know which sentence to make
):
    '''
    convert csv row into text
    updated with the new tokenizer
    '''
    this_line = None
    if   sentence_id == "Pred001":
        # Type: forward predicting 1
        # id: Pred001
        this_line = \
        "{{Task:Pred001}}=\
{{Composition:[(Al):{:+.3e},(Ni):{:+.3e},(Er):{:+.3e},\
(Zr):{:+.3e},(Y):{:+.3e},(Yb):{:+.3e}]}}=>\
{{Structure:[L12MolePC:{:+.3e},ScheilL12MeltMolePC:{:+.3e},ScheilL12MolePC:{:+.3e},\
ScheilTerneryMeltMolePC:{:+.3e},ScheilTernaryMolePC:{:+.3e},ScheilAl3NiMeltMolePC:{:+.3e},\
ScheilAl3NiMolePC:{:+.3e},ScheilAl3ZrMeltMolePC:{:+.3e},ScheilAl3ZrMolePC:{:+.3e}]}}=>\
{{Property:[BulkResistivity:{:+.3e},Misfit:{:+.3e},CoarseningRate:{:+.3e},\
ScheilFRCutoff:{:+.3e},ScheilFRMatrix:{:+.3e},ScheilCSC:{:+.3e},\
ScheilHCS:{:+.3e}]}}".format (
            # composition
            element['AlMolePC'],
            element['NiMolePC'],
            element['ErMolePC'],
            element['ZrMolePC'],
            element['YMolePC'],
            element['YbMolePC'],
            # structure numbers
            element['L12MolePC'],
            element['ScheilL12MeltMolePC'],
            element['ScheilL12MolePC'],
            element['ScheilTerneryMeltMolePC'],
            element['ScheilTernaryMolePC'],
            element['ScheilAl3NiMeltMolePC'],
            element['ScheilAl3NiMolePC'],
            element['ScheilAl3ZrMeltMolePC'],
            element['ScheilAl3ZrMolePC'],
            # properties
            element['BulkResistivity'],
            element['Misfit'],
            element['CoarseningRate'],
            element['ScheilFRCutoff'],
            element['ScheilFRMatrix'],
            element['ScheilCSC'],
            element['ScheilHCS'],
        )
    elif sentence_id == "Gene001":
        this_line = \
        "{{Task:Gene001}}=\
{{Property:[BulkResistivity:{:+.3e},Misfit:{:+.3e},CoarseningRate:{:+.3e},\
ScheilFRCutoff:{:+.3e},ScheilFRMatrix:{:+.3e},ScheilCSC:{:+.3e},\
ScheilHCS:{:+.3e}]}}=>\
{{Structure:[L12MolePC:{:+.3e},ScheilL12MeltMolePC:{:+.3e},ScheilL12MolePC:{:+.3e},\
ScheilTerneryMeltMolePC:{:+.3e},ScheilTernaryMolePC:{:+.3e},ScheilAl3NiMeltMolePC:{:+.3e},\
ScheilAl3NiMolePC:{:+.3e},ScheilAl3ZrMeltMolePC:{:+.3e},ScheilAl3ZrMolePC:{:+.3e}]}}=>\
{{Composition:[(Al):{:+.3e},(Ni):{:+.3e},(Er):{:+.3e},(Zr):{:+.3e},(Y):{:+.3e},\
(Yb):{:+.3e}]}}".format (
            # properties
            element['BulkResistivity'],
            element['Misfit'],
            element['CoarseningRate'],
            element['ScheilFRCutoff'],
            element['ScheilFRMatrix'],
            element['ScheilCSC'],
            element['ScheilHCS'],
            # structure numbers
            element['L12MolePC'],
            element['ScheilL12MeltMolePC'],
            element['ScheilL12MolePC'],
            element['ScheilTerneryMeltMolePC'],
            element['ScheilTernaryMolePC'],
            element['ScheilAl3NiMeltMolePC'],
            element['ScheilAl3NiMolePC'],
            element['ScheilAl3ZrMeltMolePC'],
            element['ScheilAl3ZrMolePC'],

            # composition
            element['AlMolePC'],
            element['NiMolePC'],
            element['ErMolePC'],
            element['ZrMolePC'],
            element['YMolePC'],
            element['YbMolePC'],

        )
    else:
        print ("Sentence type is not known")
    return this_line
# 
def assemble_multi_sentence(
    element,
    key_list,
):
    resu_dict = {}
    for this_key in key_list:
        # 1. assemble the sentence
        this_sentence = assemble_one_sentence(
            element,
            this_key
        )
        # deliver the result
        this_key_name = this_key+"_sentence"
        resu_dict[this_key_name] = this_sentence
        
    return resu_dict
# ===================================================
# ver_0: for ByT5 tokenizer
# 
def tokenize_multi_sentence(
    element,
    key_list,
    tokenizer=None,
    context_length=1024,
):
    input_batch = []
    for this_key in key_list:
        # 2. tokenized it
        outputs = tokenizer(
            element[this_key+'_sentence'],
            # truncation=True,
            # return_overflowing_tokens=True,
            max_length=context_length,
            return_length=True,
            # ++
            padding='max_length',
        )
        # input_batch = []
        # for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        #     if length == context_length:
        #         input_batch.append(input_ids)
        # for input_ids in outputs["input_ids"]:
        #     if len(input_ids)==context_length:
        #         input_batch.append(input_ids)
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length <= context_length+1:
                input_batch.append(input_ids)
        
    return {'input_ids': input_batch}
# ===================================================
# 
def tokenize_multi_sentence_with_BPE(
    element,
    key_list,
    tokenizer=None,
    context_length=1024,
):
    input_batch = []
    for this_key in key_list:
        # 2. tokenized it
        outputs = tokenizer.encode_batch(
            element[this_key+'_sentence'],
        )
        # # input_batch = []
        # # for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        # #     if length == context_length:
        # #         input_batch.append(input_ids)
        # # for input_ids in outputs["input_ids"]:
        # #     if len(input_ids)==context_length:
        # #         input_batch.append(input_ids)
        # for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        #     if length <= context_length+1:
        #         input_batch.append(input_ids)
        # -----------------------------------------
        # for input_ids in zip(outputs.ids):
        #     input_batch.append(input_ids)
        # ++
        for this_input in outputs:
            # print (this_input)
            input_batch.append(this_input.ids)
        
        
    return {'input_ids': input_batch}

