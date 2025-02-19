import os
import time
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch
from tqdm import tqdm
import math
# 
# other functions 
# from hmGPT.UtilityPack import add_one_line_to_file
from AlloyGPT.UtilityPack import add_one_line_to_file
# ===================================================================
# New: initialization for training loop
# 
# ========================================================================
# for transformer.tokenizers
def translate_words_w_transformers(
        tokenizer,
        target_keywords,
    ):
    # 
    # we pay attension to numbers here
    keytoken_ids = []
    
    for keyword in target_keywords:
        
        ids = tokenizer([keyword]).input_ids[0]
        if len(ids) == 2: # 1:
            keytoken_ids.append(ids[0])
        else:
            print(f"Keyword has not single token: {keyword}")
            
    return keytoken_ids
# 
# ========================================================================
# use tokenizers
def translate_words_w_tokenizers(
        tokenizer,
        target_keywords,
    ):
    # 
    # we pay attension to numbers here
    keytoken_ids = []
    
    for keyword in target_keywords:
        
        ids = tokenizer.token_to_id(keyword)
        
        if ids == None:
            print (f"Keyword has not single token: {keyword}")
        else:
            keytoken_ids.append(ids)
            
    return keytoken_ids
# ===========================================================================
# 
def build_weight_list_for_vocab(
    vocab_size,
    keytoken_ids, # ids for those important
    wei_0, # wei for those are normal
    wei_1, # wei for those are important
):
    wei_list_for_all_classes = torch.ones(
        (vocab_size,)
    )*wei_0
    # modify those important
    wei_list_for_all_classes[keytoken_ids] = wei_1
    
    return wei_list_for_all_classes
# 
def initialize_train_fun(TrainKeys):
    # 0. unload input
    backend = TrainKeys['backend']
    device = TrainKeys['device']
    gradient_accumulation_steps = TrainKeys['gradient_accumulation_steps']
    batch_size = TrainKeys['batch_size'] 
    block_size = TrainKeys['block_size']
    dtype = TrainKeys['dtype']
    out_dir = TrainKeys['out_dir']
        
    # various inits, derived attributes, I/O setup
    # 
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        # add some fake one to make it pass one
        ddp_local_rank = None
        ddp_rank = None
        
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration/GAS will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # 3. passon data
    TrainKeys['master_process']=master_process
    TrainKeys['seed_offset']=seed_offset
    TrainKeys['ddp_world_size']=ddp_world_size
    # ++
    TrainKeys['ddp_local_rank'] = None
    TrainKeys['ddp_rank'] = None
    
    TrainKeys['tokens_per_iter']=tokens_per_iter
    TrainKeys['ptdtype']=ptdtype
    TrainKeys['device_type']=device_type
    TrainKeys['ddp']=ddp

    return TrainKeys, ctx
# ==============================================================
# 
# learning rate decay scheduler (cosine with warmup)
def get_lr(
    it,
    warmup_iters,
    lr_decay_iters,
    min_lr,
    learning_rate,
):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
# 
# ================================================================
def prep_shifted_ids_from_batch(
    batch,
    device_type,
    device,
):    
    this_batch_size, this_block_size = batch['input_ids'].shape
    x = batch['input_ids'][:,0:0+this_block_size-1]
    y = batch['input_ids'][:,1:1+this_block_size-1]
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        
    return x, y               
# ================================================================
# 
@torch.no_grad()
def evaluate_on_dataloader_for_AR_LM_GPT2(
    ctx,
    model,
    eval_dataloader,
    device_type,
    device,
    pick_num_batch = 1.E9,
):
    '''
    for auto regressive language model
    '''
    model.eval()
    losses = []
    for ii, batch in enumerate(tqdm(eval_dataloader)):
        if ii<pick_num_batch:
            x, y = prep_shifted_ids_from_batch(
                batch,
                device_type,device,
            )
            with ctx:
                # logits, loss = model(x, y)
                _, loss = model(x, y)
            # losses[ii] = loss.item()
            losses.append(loss.item())
        else:
            break
    loss_mean = sum(losses)/len(losses)
    # back into training mode
    model.train()
    # 
    return loss_mean
# =================================================================
# 
# =================================================
# make one prediction
def generate_single_example_AR_LM_w_trans_tokenizer(
    ctx,
    model,
    tokenizer,
    test_prompt,
    device,
    pred_num,
    max_new_tokens,
    pred_temp,
    top_k
):
    test_ids = torch.tensor(
        tokenizer.encode(test_prompt, add_special_tokens=False)
    ).unsqueeze(0).to(device) # .to(device)
    
    resu = []
    model.eval()
    with torch.no_grad():
        with ctx:
            for k in range(pred_num):
                test_sample_output = model.generate(
                    idx=test_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=pred_temp, 
                    top_k=top_k
                )
                # 
                # print (test_sample_output.shape)
                # decode
                for i, sample_output in enumerate(test_sample_output):
                    resu.append(
                        tokenizer.decode(sample_output, skip_special_tokens=True)
                    )
                    # print(
                    #     "{}: {}\n\n".format(
                    #         i, tokenizer.decode(sample_output, skip_special_tokens=True)
                    #     )
                    # )
                
    return resu
# 
def generate_single_example_AR_LM_w_token_tokenizer(
    ctx,
    model,
    tokenizer,
    test_prompt,
    device,
    pred_num,
    max_new_tokens,
    pred_temp,
    top_k
):
    # test_ids = torch.tensor(
    #     tokenizer.encode(test_prompt, add_special_tokens=False)
    # ).unsqueeze(0).to(device) # .to(device)
    
    # assume test_prompt only has one line
    test_token_pack = tokenizer.encode(
        test_prompt,
        add_special_tokens=False,
    )
    # print (test_token_pack)
    # pick only those are useful
    test_ids = [
        this_ids for (this_ids, this_a_mask) in \
        zip(test_token_pack.ids, test_token_pack.attention_mask) \
        if this_a_mask == 1 
    ]
    # convert it into torch.tensor
    test_ids = torch.tensor(test_ids).unsqueeze(0).to(device)
    
    resu = []
    model.eval()
    with torch.no_grad():
        with ctx:
            for k in range(pred_num):
                test_sample_output = model.generate(
                    idx=test_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=pred_temp, 
                    top_k=top_k
                )
                # 
                # print (test_sample_output.shape)
                # decode with Tokenizers
                test_sample_output = tokenizer.decode_batch(
                    sequences=test_sample_output.tolist()
                )
                # decode
                for i, sample_output in enumerate(test_sample_output):
                    resu.append(
                        sample_output
                    )
                    # print(
                    #     "{}: {}\n\n".format(
                    #         i, tokenizer.decode(sample_output, skip_special_tokens=True)
                    #     )
                    # )
                
    return resu
# =================================================================
# 
def generate_multi_single_example_AR_LM_w_token_tokenizer(
    ctx,
    model,
    tokenizer,
    test_prompts,
    device,
    pred_num,
    max_new_tokens,
    pred_temp,
    top_k,
):
    resu = []
    for test_prompt in test_prompts:
        single_predict = generate_single_example_AR_LM_w_token_tokenizer(
            ctx,
            model,
            tokenizer,
            test_prompt,
            device,
            pred_num,
            max_new_tokens,
            pred_temp,
            top_k
        )
        for this_pred in single_predict:
            resu.append(this_pred)
            
    return resu
# =================================================================
# 
from tqdm import tqdm
# 
def training_loop_AR_LM(
    # 
    model,
    tokenizer,
    ctx,
    optimizer,
    TrainKeys,
    train_dataloader,
    eval_dataloader,
    eval_size,
    wei_list_for_all_vocab,
    scaler,
    test_prompts_during_train,
    # 
    best_val_loss_0,
    GAS_at_best_val_loss_0,
    finished_steps_0,
    completed_updating_steps_0,
    model_args,
):
    # 
    # pick up records
    best_val_loss = best_val_loss_0
    GAS_at_best_val_loss = GAS_at_best_val_loss_0
    finished_steps = finished_steps_0
    completed_updating_steps = completed_updating_steps_0
    # 
    ddp = TrainKeys['ddp']
    gradient_accumulation_steps = TrainKeys['gradient_accumulation_steps']
    device_type = TrainKeys['device_type']
    # device = TrainKeys['device']
    grad_clip = TrainKeys['grad_clip']
    master_process = TrainKeys['master_process']
    # pick up records


    model.train() # change into training mode
    raw_model = model.module if ddp else model # unwrap DDP container if needed

    this_step = 0
    for epoch in range(TrainKeys['num_train_epochs']):
        for batch in train_dataloader: # assume this dataloader has fixed order
        # for batch in tqdm(train_dataloader): # assume this dataloader has fixed order
            this_step += 1
            if this_step > finished_steps: # get into the new training steps
            # for debug 
            # if this_step < 32+1: #  finished_steps: # get into the new training steps
                # # for debug
                # print ('step: ', this_step)

                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                # --
                # for micro_step in range(gradient_accumulation_steps):
                # ++
                micro_step = this_step % gradient_accumulation_steps
                # ==========================================================
                # I.1. calculate the loss per step/batch
                # ==========================================================
                # 1. get one batch as input and output
                x, y = prep_shifted_ids_from_batch(
                    batch,
                    device_type,
                    TrainKeys['device'],
                )
                # this_batch_size, this_block_size = batch['input_ids'].shape
                # x = batch['input_ids'][:,0:0+this_block_size-1]
                # y = batch['input_ids'][:,1:1+this_block_size-1]
                # if device_type == 'cuda':
                #     # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                #     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
                # else:
                #     x, y = x.to(device), y.to(device)

                # 
                # 2. calculate the loss
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    # --
                    # # model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                    # ++
                    model.require_backward_grad_sync = (micro_step == 0)
                # https://pytorch.org/docs/stable/amp.html#torch.cpu.amp.autocast
                with ctx: 
                    # note: output of model() is (logits, loss)
                    # from the model:
                    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                    logits, loss_from_model = model (x, y) # many consider skip loss_from_model
                    # calculate a weighted loss if needed
                    # calculate per-token loss
                    loss_weighted = F.cross_entropy(
                        input=logits.view(-1, logits.size(-1)), 
                        target=y.view(-1),
                        weight=wei_list_for_all_vocab,
                        ignore_index=-1
                    )
                    # loss_from_model = loss_from_model/gradient_accumulation_steps
                    loss_weighted = loss_weighted/gradient_accumulation_steps
                # 
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss_weighted).backward()

                # ==========================================================
                # I.2. update the model at GAS: Gradient Accumulated Step
                # ==========================================================
                if micro_step==0:
                    print ("#", end='')
                    completed_updating_steps += 1
                    # get lr for this GAS.
                    # NOTE: define lr using GAS
                    # determine and set the learning rate for GAS
                    lr = get_lr(
                        it=completed_updating_steps,
                        warmup_iters=TrainKeys['warmup_iters'],
                        lr_decay_iters=TrainKeys['lr_decay_iters'],
                        min_lr=TrainKeys['min_lr'],
                        learning_rate=TrainKeys['learning_rate'],
                    ) if TrainKeys['decay_lr'] else TrainKeys['learning_rate']
                    # 
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # clip the gradient
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    # step the optimizer and scaler if training in fp16
                    scaler.step(optimizer)
                    scaler.update()
                    # make a mark

                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)

                # ====================================================================
                # II. reporting and recroding
                # ====================================================================
                # 
                # 1. cheap reporting: training loss at some GAS
                # here, we haven't deal with multi-GPU cases
                if this_step % (
                    TrainKeys['report_1_trai_loss_this_GAS'] * \
                    TrainKeys['gradient_accumulation_steps']
                )==0:
                    print ("\n")
                    # print to file
                    # train_loss, mean_loss
                    this_write_line = f"%d,%d,%d,%f,%f,%f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss_weighted.item()*TrainKeys['gradient_accumulation_steps'],
                        loss_from_model.item(),lr,
                    )
                    this_print_line = f"epoch: %d, step: %d, GAS: %d, wei_loss/trai: %f, plain_loss/trai: %f, lr: %f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss_weighted.item()*TrainKeys['gradient_accumulation_steps'],
                        loss_from_model.item(),lr,
                    )
                    if master_process:
                        add_one_line_to_file(
                            file_name=TrainKeys['1_train_loss.log'],
                            this_line=this_write_line,
                            mode='a',
                        )
                        print (this_print_line)

                # 2. expensive reporting: vali loss + pred examples
                # 
                if this_step % (
                    TrainKeys['report_2_vali_pred_this_GAS'] * \
                    TrainKeys['gradient_accumulation_steps']
                )==0:
                    # a. vali loss: on the whole eval_dataloader
                    eval_loss = evaluate_on_dataloader_for_AR_LM_GPT2(
                        ctx,
                        model,
                        eval_dataloader,
                        device_type,
                        TrainKeys['device'],
                        pick_num_batch = eval_size
                    )
                    # 
                    this_write_line = f"%d,%d,%d,%f,%f,%f,%f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss_weighted.item()*TrainKeys['gradient_accumulation_steps'],
                        loss_from_model.item(), # unweighted trainloss
                        eval_loss, # ave loss on the eval_dataloader
                        lr, # learning rate
                    )
                    # 
                    this_print_line = f"epoch: %d, step: %d, GAS: %d, wei_loss/trai: %f, plain_loss/trai: %f, loss/eval: %f, lr: %f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss_weighted.item()*TrainKeys['gradient_accumulation_steps'],
                        loss_from_model.item(), # unweighted trainloss
                        eval_loss, # ave loss on the eval_dataloader
                        lr, # learning rate
                    )
                    if master_process:
                        add_one_line_to_file(
                            file_name=TrainKeys['2_vali_loss.log'],
                            this_line=this_write_line,
                            mode='a',
                        )
                        print (this_print_line)
                    # 
                    # b. predict some examples
                    # generate_single_example_AR_LM(
                    #     ctx,
                    #     model,
                    #     tokenizer,
                    #     test_prompt,
                    #     device,
                    #     pred_num,
                    #     max_new_tokens,
                    #     pred_temp,
                    #     top_k
                    # )
                    # 
                    line_preds = generate_multi_single_example_AR_LM_w_token_tokenizer(
                        ctx,
                        model,
                        tokenizer,
                        test_prompts=test_prompts_during_train,
                        device=TrainKeys['device'],
                        pred_num=1,
                        max_new_tokens=TrainKeys['block_size'],
                        pred_temp=1.,
                        top_k=TrainKeys['vocab_size'],
                    )
                    if master_process:
                        for ii, this_line in enumerate(line_preds):
                            add_line = '{}: \n{}\n'.format (ii, this_line)
                            add_one_line_to_file(
                                file_name=TrainKeys['2_vali_gene.log'],
                                this_line=add_line,
                                mode='a',
                            )
                            print (add_line)
                    # 

                # 3. save checkpoints: every, the best and the last
                # 
                if this_step % (
                    TrainKeys['report_3_save_mode_this_GAS'] * \
                    TrainKeys['gradient_accumulation_steps']
                )==0:
                    # check if be the best
                    IF_Record_Best = 0
                    if eval_loss < best_val_loss:
                        # update
                        best_val_loss = eval_loss
                        GAS_at_best_val_loss = completed_updating_steps
                        IF_Record_Best = 1
                    # ++++++++++++++++++++++++++++++++
                    # prepare the model package
                    # Note, "best_val_loss" is updated if needed at this step
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        # 'iter_num': completed_updating_steps,
                        'completed_updating_steps': completed_updating_steps,
                        'step_num': this_step,
                        'iter_num_at_best_loss': GAS_at_best_val_loss,
                        'best_val_loss': best_val_loss
                    }
                    # ++++++++++++++++++++++++++++++
                    # update the best: 
                    # NOTE, this has a limit freq
                    # if eval_loss < best_val_loss:
                    if IF_Record_Best==1:
                        # # update
                        # best_val_loss = eval_loss
                        # GAS_at_best_val_loss = completed_updating_steps
                        # 
                        # best_checkpoint = {
                        #     'model': raw_model.state_dict(),
                        #     'optimizer': optimizer.state_dict(),
                        #     'model_args': model_args,
                        #     # 'iter_num': completed_updating_steps,
                        #     'completed_updating_steps': completed_updating_steps,
                        #     'step_num': this_step,
                        #     'iter_num_at_best_loss': GAS_at_best_val_loss,
                        #     'best_val_loss': best_val_loss
                        #     # 'config': config,
                        # }
                        
                        torch.save (
                            checkpoint, 
                            os.path.join(
                                TrainKeys['out_dir_best'],
                                'Best_ckpt.pt'
                            )
                        )
                        # add a note
                        if master_process:
                            this_write_line = f"%d,%d,%d\n" % (
                                epoch,this_step,completed_updating_steps,
                            )
                            add_one_line_to_file(
                                file_name=TrainKeys['3_save_model_best.log'],
                                this_line=this_write_line,
                                mode='w',
                            )
                            print (f"\nsave so-far BEST checkpoint.\n")
                    # ==================================================
                    # save regularly
                    
                    torch.save (
                        checkpoint, 
                        os.path.join(
                            TrainKeys['out_dir_last'],
                            'Last_ckpt.pt'
                        )
                    )
                    if master_process:
                        this_write_line = f"%d,%d,%d\n" % (
                            epoch,this_step,completed_updating_steps,
                        )
                        add_one_line_to_file(
                            file_name=TrainKeys['3_save_model_last.log'],
                            this_line=this_write_line,
                            mode='w',
                        )
                        print (f"\nsave checkpoint REGULARLY at GAS: {completed_updating_steps}.\n")

                    # ++++++++++++++++++++++++++++++


                # clean the eval tail 
                if not model.training:
                    model.train()
            else:
                pass




                    # update the model if this reaches an accumulated updating step


    return 0

# ===============================================================
# for Test part
# 
import re
# ===================================================================
# New: initialization for testing loop
def initialize_test_fun(TestKeys):
    
    device = TestKeys['device']
    dtype = TestKeys['dtype']
    
    torch.manual_seed(TestKeys['seed'])
    torch.cuda.manual_seed(TestKeys['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # 
    TestKeys['device_type'] = device_type
    TestKeys['ptdtype'] = ptdtype
    
    return TestKeys, ctx
# ====================================================================
# 
# def convert_one_line (this_reco, target_key_list):
#     # # for debug
#     # print (this_reco)
#     # print ()
    
#     try: 
#         #
#         data_blocks = re.findall(r"\[.*?\]", this_reco)
#         # data_blocks = re.findall(r'\[([^]]*)\]', this_reco)
#         # print (data_blocks)
#         resu_dict = {}
#         for ii, this_block in enumerate(data_blocks):
#             # handle one block
#             # 1. get rid of the braket
#             this_block = this_block[1:-1]
#             # 2. collect keys and values
#             this_KV_pairs = this_block.split(',')
#             for jj, this_pair in enumerate(this_KV_pairs):
#                 this_resu = this_pair.split(':')
#                 resu_dict[this_resu[0]] = this_resu[1]
#         #
#         resu_dict['SafeRec'] = '1'
#         # 
#         # check the completeness
#         # 
#         for this_key in target_key_list:
#             assert this_key in resu_dict.keys()
#             # ++
            
#     except:
#         # 
#         # ++ for debug
#         print ("Cannot translate the line:")
#         print (this_reco)
#         print ()
        
#         resu_dict = {}
#         for this_key in target_key_list:
#             resu_dict[this_key] = '0.0'
#         # 
#         resu_dict['SafeRec'] = '0'
    
            
#     return resu_dict
