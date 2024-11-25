import os
import torch
import numpy as np
import random
import json
import time
import math
import pickle as pkl
import sys
from packaging import version
#from hfm_gen_eval import hfm_generation
from cap_hal_eval import cap_gen_val

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from sklearn.metrics import roc_auc_score
import transformers
from transformers.utils import is_sagemaker_mp_enabled, is_torch_tpu_available
from logp_gen_for_pairs import get_batch_logps
"""
trainer_utils no longer has ShardedDDPOption
"""
from transformers.trainer_utils import has_length
from transformers.trainer_callback import TrainerState
from accelerate import Accelerator, skip_first_batches
from accelerate import __version__ as accelerate_version
import time
import utils

def compute_auc_score(logits,label):
    bz=logits.shape[0]
    logits=logits.numpy()
    label=label.numpy()
    auc=roc_auc_score(label,logits,average='weighted')*bz
    return auc

def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))

def evaluate_posneg_dpo(trainer, data_collator, pad_token_id, args, dpo_weight=1.0, sft_weight=0.0):
    """
    re-write the evaluate functions in huggingface trainers
    """
    eval_dataset = trainer.eval_dataset
    metrics={}
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.micro_batch_size, 
                                                  collate_fn=data_collator,
                                                  num_workers=5, shuffle=False)
    num_bz=len(eval_dataloader)
    print ('Length of val data:',num_bz)
    trainer.model.eval()
    total_loss=0.0
    start_time = time.time()
    for step, inputs in enumerate(eval_dataloader):
        with torch.no_grad():
            val_loss_step=compute_dpo_loss(trainer.model, inputs, dpo_weight, sft_weight, pad_token_id, args)
        total_loss+=val_loss_step.item()
        
    trainer.model.train()
    
    metrics['eval_runtime']=time.time()-start_time
    metrics['eval_loss']=total_loss/num_bz
    return metrics
    
def mine_maybe_log_save_evaluate(trainer,all_iters, all_acc, all_auc, logger, opt, tokenizer,image_processor,
                                 tr_loss, model, trial, epoch, ignore_keys_for_eval):
    if trainer.control.should_log:
        if is_torch_tpu_available():
            xm.mark_step()
        # all_gather + mean() to get average loss over all processes
        tr_loss_scalar = trainer._nested_gather(tr_loss).mean().item()

        # reset tr_loss to zero
        tr_loss -= tr_loss
        logger.write('\tIteration %d' % (trainer.state.global_step))
        logger.write('\tLoss: %.2f, learning rate: %f' %
                     (round(tr_loss_scalar / (trainer.state.global_step - trainer._globalstep_last_logged), 4),
                      trainer._get_learning_rate()))

        trainer._total_loss_scalar += tr_loss_scalar
        trainer._globalstep_last_logged = trainer.state.global_step
        trainer.store_flos()
    metrics = None
    if trainer.control.should_evaluate:
        logger.write('Iteration %d, evaluation...' % (trainer.state.global_step))
        
        if isinstance(trainer.eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
                dataset_metrics = trainer.evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys_for_eval,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
        else:
            if opt.DATA_NAME in ['vsg','coco']:
                metrics=evaluate_posneg_dpo(trainer, trainer.data_collator, tokenizer.pad_token_id, opt)
            else:
                metrics = trainer.evaluate(ignore_keys=ignore_keys_for_eval)
                trainer._report_to_hp_search(trial, trainer.state.global_step, metrics)
        logger.write('\tEval loss: %.2f, runtime %.2f' % 
                     (metrics['eval_loss'],metrics['eval_runtime']))
        
        if trainer.state.global_step>=opt.gen_start and trainer.state.global_step%opt.gen_step==0:
            """
            This is deprecated!!!
            Be careful and ignore it
            """
            if opt.DATA_NAME in ['mem','harm','mimc']:
                invalid,auc,acc=hfm_generation(model,tokenizer,image_processor,opt)
                if len(invalid)>0:
                    logger.write('\tThere are %d invalid examples' % (len(invalid)))
                logger.write('\tAUC %.2f, Acc %.2f' % (auc,acc))
                all_acc.append(acc)
                all_auc.append(auc)
                all_iters.append(trainer.state.global_step)
            elif opt.DATA_NAME in ['hate-speech']:
                invalid,auc,acc=hsp_generation(model,tokenizer,image_processor,opt)
                if len(invalid)>0:
                    logger.write('\tThere are %d invalid examples' % (len(invalid)))
                logger.write('\tAUC %.2f, Acc %.2f' % (auc,acc))
                all_acc.append(acc)
                all_auc.append(auc)
                all_iters.append(trainer.state.global_step)
            elif opt.DATA_NAME in ['vsg','coco']:
                """
                for simplicity ==> acc for chair_i, auc for chair_s
                """
                hal_datasets=opt.HAL_DATASETS.split(',')
                acc,auc=cap_gen_val(model,tokenizer,image_processor,opt,trainer.state.global_step)
                for num,ac in enumerate(acc):
                    logger.write('Hallucination chair eval: %s' % (hal_datasets[num]))
                    logger.write('\tChair_s %.2f, Chair_i %.2f' % (auc[num],ac))
                all_acc.append(acc[0])
                all_auc.append(auc[0])
                #all_acc for storing hallucination rate
                all_iters.append(trainer.state.global_step)

def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out

#adopt the code from https://github.com/RLHF-V/RLAIF-V/blob/main/muffin/train/trainers.py#L91
def dpo_loss(policy_chosen_logps,
             policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps,
             beta,
             reference_free= False):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != -100)
    # print(f'compute wlogp {labels.shape} {loss_mask.shape}, {token_weight.shape}, {per_token_logp.shape}')
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp

def get_beta_and_logps(inputs, model, pad_token_id, args):
    win_dict=inputs['win_dict']
    rej_dict=inputs['rej_dict']

    #prepare all input data for win/reject
    win_input_ids=win_dict['input_ids']
    win_labels=win_dict['labels']
    win_logp=win_dict["logp"]
    win_avg_logp=win_dict["avg_logp"]
    win_per_token_logp=win_dict["per_token_logp"]
    win_token_weight=win_dict["token_weight"]

    rej_input_ids=rej_dict['input_ids']
    rej_labels=rej_dict['labels']
    rej_logp=rej_dict["logp"]
    rej_avg_logp=rej_dict["avg_logp"]
    rej_per_token_logp=rej_dict["per_token_logp"]
    rej_token_weight=rej_dict["token_weight"]

    concatenated_input_ids=concate_pad(win_input_ids, rej_input_ids, pad_token_id)
    concatenated_labels=concate_pad(win_labels, rej_labels, -100)
    concatenated_attention_mask=concatenated_input_ids.ne(pad_token_id)
    concatenated_token_weight=concate_pad(win_token_weight, rej_token_weight, 0)
    concatenated_labels=concatenated_labels.to(model.device)
    #print ('ID:',concatenated_input_ids[0])
    #print (pad_token_id)
    #print ('Mask:',concatenated_attention_mask[0])
    
    """
    (_,_,_,_,
    concatenated_inputs_embeds,
    concatenated_labels) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids.cuda(),
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels.cuda(),
            images=concatenated_images.cuda(),
            )
    """
    #concatenation
    if args.MODEL_NAME in ["llava-hf/llava-1.5-7b-hf","llava-hf/llava-1.5-13b-hf"]:
        images=inputs['pixel_values'].cuda()
        concatenated_images = torch.cat([images, images], dim=0)
        output = model.forward(
            input_ids=concatenated_input_ids.cuda(),
            attention_mask=concatenated_attention_mask.cuda(),
            #labels=concatenated_labels.cuda(),
            pixel_values=concatenated_images.cuda(),
            labels=None,
        )
    elif args.MODEL_NAME=='BAAI/Bunny-v1_0-3B':
        images=inputs['images'].cuda()
        concatenated_images = torch.cat([images, images], dim=0)
        output = model.forward(
            input_ids=concatenated_input_ids.cuda(),
            attention_mask=concatenated_attention_mask.cuda(),
            #labels=concatenated_labels.cuda(),
            images=concatenated_images.cuda(),
            labels=None,
        )[0]
    elif args.MODEL_NAME=='Qwen/Qwen-VL-Chat':
        output = model.forward(
            input_ids=concatenated_input_ids.to(model.device),
            attention_mask=concatenated_attention_mask.to(model.device),
            labels=None,
        )
    """
    transformers 4.39.2 will have bugs
        dtype mismatch (float and half)
    """
    output.logits=output.logits[:,-concatenated_labels.shape[1]:]
    log_prob, average_log_prob = get_batch_logps(
        output.logits, concatenated_labels, return_per_token_logp=False)
    #print (log_prob)
    #print (average_log_prob)
    if args.dpo_use_average:
        concatenated_logp = average_log_prob
    else:
        concatenated_logp =log_prob
    win_size = win_input_ids.shape[0]
    rej_size = rej_input_ids.shape[0]

    if args.dpo_use_average:
        ref_win_logp=win_dict['logp']
        ref_rej_logp=rej_dict['logp']
    else:
        ref_win_logp=win_dict['avg_logp']
        ref_rej_logp=rej_dict['avg_logp']
    if args.dpo_token_weighted:
        ref_win_logp = compute_weighted_logp(
            win_per_token_logp, win_labels, win_token_weight, args.dpo_use_average)
        ref_rej_logp = compute_weighted_logp(
            rej_per_token_logp, rej_labels, rej_token_weight, args.dpo_use_average)
        concatenated_logp = compute_weighted_logp(
            concatenated_logp, concatenated_labels, concatenated_token_weight, args.dpo_use_average)
    policy_win_logp, policy_rej_logp = concatenated_logp.split([win_size, rej_size])
    return policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, args.dpo_beta

def compute_dpo_loss(model, inputs, dpo_weight, sft_weight, pad_token_id, args):
    policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(inputs, model, pad_token_id, args)
    losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                        policy_rej_logp,
                                                        ref_win_logp.to(policy_win_logp.device),
                                                        ref_rej_logp.to(policy_win_logp.device),
                                                        beta=beta)
    loss= losses.mean()*dpo_weight - policy_win_logp.mean()*sft_weight
    #print (loss.item())
    return loss
"""
Re-write the trainig_step function
"""
def wrap_training_step_dpo(trainer, model, inputs, pad_token_id, args, dpo_weight=1.0, sft_weight=0.0):
    model.train()
    loss=compute_dpo_loss(model, inputs, dpo_weight, sft_weight, pad_token_id, args)
    del inputs
    kwargs = {}
    """
    omit the lr rate setting explicitly as we do not employ LOMO optimizer
    """
    if trainer.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    if trainer.use_apex:
        with amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        trainer.accelerator.backward(loss, **kwargs)
    return loss.detach() / trainer.args.gradient_accumulation_steps


def rewrite_train(trainer,tokenizer,image_processor,opt,logger,
                  train_cls,data_collator,
                  trail=None,ignore_keys_for_eval=None):
    #using opt for my own config to avoid confusion with args in trainer
    trainer._memory_tracker.start()
    args=trainer.args
    trainer.is_in_train = True
    trial=None
    if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
        trainer._move_model_to_device(trainer.model, args.device)
    trainer._hp_search_setup(trial)
    trainer._train_batch_size = trainer.args.train_batch_size
    trainer.accelerator.free_memory()
    #initializing dataloaders
    #train_dataloader = trainer.get_train_dataloader()
    train_dataloader = torch.utils.data.DataLoader(train_cls, batch_size=opt.micro_batch_size, collate_fn=data_collator,
                                                   num_workers=4, shuffle=False)
    total_train_batch_size = trainer._train_batch_size * args.gradient_accumulation_steps * args.world_size
    logger.write('Training batch size: %d' % (trainer._train_batch_size))#bz per device
    logger.write('Total training batch size: %d' % (total_train_batch_size))
    logger.write('Total number of epochs: %d' % (opt.num_epochs))
    #getting the length of data loaders
    len_dataloader = None
    if has_length(train_dataloader):
        len_dataloader = len(train_dataloader)
        #the dataloader is devided with batch size per device
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = trainer.num_examples(train_dataloader)
        logger.write('\tLength of train dataloader: %d' % (len_dataloader))
        logger.write('\tNumber of iterations per epoch: %d' % (num_update_steps_per_epoch))
        logger.write('\tNumber of instances: %d' % (num_examples))
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = trainer.num_examples(train_dataloader) * args.num_train_epochs
        logger.write('\tTotal number of iterations: %d' % (max_steps))
    elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
        max_steps = args.max_steps
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_examples = total_train_batch_size * args.max_steps
        num_train_samples = args.max_steps * total_train_batch_size
    else:
        raise ValueError(
            "args.max_steps must be set to a positive value if dataloader does not have a length, was"
            f" {args.max_steps}"
        )
    """
    delay_optimizer_creation = (
        trainer.sharded_ddp is not None
        and trainer.sharded_ddp != ShardedDDPOption.SIMPLE
        or is_sagemaker_mp_enabled()
        or trainer.fsdp is not None
        or trainer.is_fsdp_enabled
        )
    """
    delay_optimizer_creation = (
        is_sagemaker_mp_enabled()
        #or trainer.fsdp is not None
        #or trainer.is_fsdp_enabled
        )
    if trainer._created_lr_scheduler:
        trainer.lr_scheduler = None
        trainer._created_lr_scheduler = False
    if not delay_optimizer_creation:
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)
        trainer.state = TrainerState()
        trainer.state.is_hyper_param_search = trial is not None
    #about model saving and evaluation
    if args.logging_steps is not None:
        if args.logging_steps < 1:
            trainer.state.logging_steps = math.ceil(max_steps * args.logging_steps)
        else:
            trainer.state.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        if args.eval_steps < 1:
            trainer.state.eval_steps = math.ceil(max_steps * args.eval_steps)
        else:
            trainer.state.eval_steps = args.eval_steps
    if args.save_steps is not None:
        if args.save_steps < 1:
            trainer.state.save_steps = math.ceil(max_steps * args.save_steps)
        else:
            trainer.state.save_steps = args.save_steps

    model = trainer._wrap_model(trainer.model_wrapped)
    use_accelerator_prepare = True if model is trainer.model else False
    logger.write('\tAccelerate: %d' % (int(use_accelerator_prepare)))
    if delay_optimizer_creation:
        if use_accelerator_prepare:
            trainer.model = trainer.accelerator.prepare(trainer.model)
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)
    if use_accelerator_prepare:
        trainer.model.train()
        if hasattr(trainer.lr_scheduler, "step"):
            if trainer.use_apex:
                model = trainer.accelerator.prepare(trainer.model)
            else:
                model, trainer.optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                trainer.model, trainer.optimizer, trainer.lr_scheduler
            )
    if model is not trainer.model:
        trainer.model_wrapped = model
    #resume_from_checkpoint is set to be None
    trainer._load_optimizer_and_scheduler(None)

    trainer.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None
    # Update the references
    trainer.callback_handler.model = trainer.model
    trainer.callback_handler.optimizer = trainer.optimizer
    trainer.callback_handler.lr_scheduler = trainer.lr_scheduler
    trainer.callback_handler.train_dataloader = train_dataloader
    #hp and trail both set to be None
    trainer.state.trial_params = None
    trainer.state.max_steps = max_steps
    trainer.state.num_train_epochs = num_train_epochs
    trainer.state.is_local_process_zero = trainer.is_local_process_zero()
    trainer.state.is_world_process_zero = trainer.is_world_process_zero()
    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0)
    #.to(args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    trainer._total_loss_scalar = 0.0
    trainer._globalstep_last_logged = trainer.state.global_step
    model.zero_grad()
    trainer.control = trainer.callback_handler.on_train_begin(args, trainer.state, trainer.control)

     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    if not args.ignore_data_skip:
        for epoch in range(epochs_trained):
            for _ in train_dataloader:
                break

    total_batched_samples = 0
    all_acc=[]
    all_auc=[]
    all_iters=[]

    #model.eval()
    #chair_is, chair_ss = cap_gen_val(model,tokenizer,image_processor,opt,0)
    #exit()
    #invalid,auc,acc=hfm_generation(model,processor,opt)
    pad_token_id=tokenizer.pad_token_id
    logger.write('Padding token id: %d' % (pad_token_id))
    for epoch in range(epochs_trained, num_train_epochs):
        epoch_iterator = train_dataloader        
        if args.past_index >= 0:
            trainer._past = None
        steps_in_epoch = (
            len(epoch_iterator)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        logger.write('Epoch: %d out of %d' % (epoch+1, num_train_epochs))
        logger.write('\tIterations in total %d' % (steps_in_epoch))
        trainer.control = trainer.callback_handler.on_epoch_begin(args, trainer.state, trainer.control)
        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True
        step = -1
        #step_trained_in_current_epoch: resuming from checkpoints
        for step, inputs in enumerate(epoch_iterator):
            #print (inputs)
            """
            Line 283 in transformers, modeling_llava.py changed
                def _merge_input_ids_with_image_features -- force image_features to be the same dtype as input_embeds (float32)
            """
            #['pixel_values']=inputs['pixel_values'].float()
            total_batched_samples += 1
            if rng_to_sync:
                trainer._load_rng_state(None)
                rng_to_sync = False

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.update(1)
                if steps_trained_in_current_epoch == 0:
                    trainer._load_rng_state(None)
                continue
            elif steps_trained_progress_bar is not None:
                steps_trained_progress_bar.close()
                steps_trained_progress_bar = None

            if step % args.gradient_accumulation_steps == 0:
                trainer.control = trainer.callback_handler.on_step_begin(args, trainer.state, trainer.control)

            with trainer.accelerator.accumulate(model):
                """
                re-write training_step function
                    source code: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py Line 3282
                """
                #tr_loss_step = trainer.training_step(model, inputs)
                tr_loss_step= wrap_training_step_dpo(trainer, model, inputs, pad_token_id, opt)

            if (
                args.logging_nan_inf_filter
                and not is_torch_tpu_available()
                and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
            ):
                # if loss is nan or inf simply add the average of previous logged losses
                tr_loss += tr_loss / (1 + trainer.state.global_step - trainer._globalstep_last_logged)
            else:
                #tr_loss += tr_loss_step.to(tr_loss.device)
                tr_loss += tr_loss_step.item()
                
            trainer.current_flos += float(trainer.floating_point_ops(inputs))
            #steps_in_epoch number of steps for each epoch
            is_last_step_and_steps_less_than_grad_acc = (
                steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
            )
            if (
                total_batched_samples % args.gradient_accumulation_steps == 0
                or
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                is_last_step_and_steps_less_than_grad_acc
            ):
                # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                # in accelerate. So, explicitly enable sync gradients to True in that case.
                if is_last_step_and_steps_less_than_grad_acc or (
                    version.parse(accelerate_version) <= version.parse("0.20.3")
                ):
                    trainer.accelerator.gradient_state._set_sync_gradients(True)
                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    # deepspeed does its own clipping
                    """
                    if trainer.do_grad_scaling:
                        # Reduce gradients first for XLA
                        if is_torch_tpu_available():
                            gradients = xm._fetch_gradients(trainer.optimizer)
                            xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                        # AMP: gradients need unscaling
                        trainer.scaler.unscale_(trainer.optimizer)
                    """
                    if is_sagemaker_mp_enabled() and args.fp16:
                        trainer.optimizer.clip_master_grads(args.max_grad_norm)
                    elif hasattr(trainer.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        trainer.optimizer.clip_grad_norm(args.max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(args.max_grad_norm)
                    elif trainer.use_apex:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        nn.utils.clip_grad_norm_(
                            amp.master_params(trainer.optimizer),
                            args.max_grad_norm,
                        )
                    else:
                        trainer.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )
                        
                # Optimizer step
                optimizer_was_run = True
                """
                trainer.do_grad_scaling no longer available!!!!
                if is_torch_tpu_available():
                    if trainer.do_grad_scaling:
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                    else:
                        # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                        trainer.optimizer.step()
                elif trainer.do_grad_scaling:
                    scale_before = trainer.scaler.get_scale()
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                    scale_after = trainer.scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    trainer.optimizer.step()
                    optimizer_was_run = not trainer.accelerator.optimizer_step_was_skipped
                """
                trainer.optimizer.step()
                optimizer_was_run = not trainer.accelerator.optimizer_step_was_skipped

                if optimizer_was_run:
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        trainer.lr_scheduler.step()

                model.zero_grad()
                trainer.state.global_step += 1
                trainer.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                trainer.control = trainer.callback_handler.on_step_end(args, trainer.state, trainer.control)
                """
                re-writing to record loss and evaluate
                """
                #original: trainer._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                #if trainer.state.global_step%100==0 and trainer.state.global_step>0:
                #    logger.write('Clearing cache!! %d' % trainer.state.global_step)
                #    torch.cuda.empty_cache()
                mine_maybe_log_save_evaluate(trainer,all_iters,all_acc, all_auc,logger,opt,tokenizer,image_processor,
                                             tr_loss, model, trial, epoch, ignore_keys_for_eval)
            else:
                trainer.control = trainer.callback_handler.on_substep_end(args, trainer.state, trainer.control)
            if trainer.control.should_epoch_stop or trainer.control.should_training_stop:
                return
            ###To Remove###
            if trainer.state.global_step>=5600:
                break
        if step < 0:
            trainer.control.should_training_stop = True
        trainer.control = trainer.callback_handler.on_epoch_end(args, trainer.state, trainer.control)
        #original: trainer._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        
        mine_maybe_log_save_evaluate(trainer,all_iters,all_acc, all_auc, logger,opt,tokenizer,image_processor,
                                     tr_loss, model, trial, epoch, ignore_keys_for_eval)
        if trainer.control.should_training_stop:
            break
    if args.past_index and hasattr(trainer, "_past"):
        # Clean the state at the end of training
        delattr(trainer, "_past")
    logger.write("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    if len(all_acc)>0:
        max_idx=sorted(range(len(all_iters)),
                        key=lambda k: all_auc[k]+all_acc[k],
                   reverse=True)[0]
        logger.write('Maximum epoch: %d' %(all_iters[max_idx]))
        logger.write('\tevaluation auc: %.2f, accuracy: %.2f' % (all_auc[max_idx], 
                                                                 all_acc[max_idx]))

def train_for_epochs(model,tokenizer,image_processor,args,
                     data_collator,
                     train_set,test_set):
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    #logger initialization
    log_path=os.path.join('../results/tune_results',args.DATA_NAME)
    if os.path.exists(log_path)==False:
        os.mkdir(log_path)
    logger=utils.Logger(os.path.join(log_path,str(args.SAVE_NUM)+'.txt'))  
    log_hyperpara(logger,args)
    logger.write('Length of training set: %d, length of testing set: %d' %
                 (len(train_set),len(test_set)))
    logger.write('Batch size: %d, Gradient accumulation: %d' %
                 (args.batch_size,gradient_accumulation_steps))
    logger.write('Number of device: %d' % (torch.cuda.device_count()))
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    #initialization of trainer from huggingface
    """
    if args.PUNISH_SHORT:
        eval_steps=1000000 # a big number ==> infinite
    else:
        eval_steps=args.eval_step
    """
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=test_set,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_step if args.val_set_size > 0 else None,
            save_steps=args.save_step,
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end= False,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to=None,
            run_name=None
        ),
        data_collator=data_collator,
    )
    model.config.use_cache = False
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    start_time=time.time()
    rewrite_train(trainer,tokenizer,image_processor,args,logger,
                  train_set, data_collator)
    end_time=time.time()
    logger.write('Time Consumption: %.2f' % (end_time-start_time))
    """
    if args.PUNISH_SHORT:
        logger.write("Training to punish short sentences for 2 epochs")
        logger.write("Re-initializing training data......")
        from posneg_pair_dataset import PosNeg_Pair_Data
        train_set=PosNeg_Pair_Data(args,processor,mode='train',dataset="vsg",punish=True)
        logger.write("\tLength of new training dataset: %d" % len(train_set))
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_set,
            eval_dataset=test_set,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=args.warmup_steps,
                num_train_epochs=2,
                learning_rate=args.learning_rate,
                fp16=args.fp16,
                logging_steps=args.logging_steps,
                optim="adamw_torch",
                evaluation_strategy="steps" if args.val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=args.eval_step if args.val_set_size > 0 else None,
                save_steps=args.save_step,
                output_dir=args.output_dir,
                save_total_limit=args.save_total_limit,
                load_best_model_at_end= False,
                ddp_find_unused_parameters=None,
                group_by_length=args.group_by_length,
                report_to=None,
                run_name=None
            ),
            data_collator=data_collator,
        )
        rewrite_train(trainer,processor,args,logger,
                      train_set, data_collator)
    """