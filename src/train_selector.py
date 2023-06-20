from utils import Recorder
from rank_data_util_cross import Data
import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
#from dataclass import Data
import numpy as np
import argparse
import logging
import nltk
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizerFast, BertTokenizerFast
from nltk.translate.bleu_score import sentence_bleu
from rank_model_cross import trainer, Model
import re
import progressbar
import time
from utils import Recorder
import torch.multiprocessing as mp
from sklearn.metrics import f1_score
import glob

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)

global record_scores
record_scores = []


def init_seed(seed=528):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def evaluate_test_pred_text(test_batch_score, test_batch_candidate_text_list, test_batch_genscore, diff_limit=None, weight=None, tau_rank=None, tau_gen=None, threshold=None, single=False):
    global record_scores
    pred_text_list = [] 
    # test_batch_score_list size is (bsz, args.test_candi_span)
    # TODO: check the logits generated for the same batch -> first 10
    test_batch_score_list = test_batch_score.detach().cpu().numpy()
    test_batch_genscore = np.array(test_batch_genscore)
    bsz = len(test_batch_score_list)
    if weight != None:
        test_batch_score_list = [softmax(test_batch_score_i) for test_batch_score_i in test_batch_score_list] # use softmax
        test_batch_genscore = [softmax(test_batch_genscore_i) for test_batch_genscore_i in test_batch_genscore]
        new_probs = np.array(test_batch_score_list)*weight + np.array(test_batch_genscore)*(1-weight)
    elif tau_gen != None and tau_rank != None:
        test_batch_score_tau_list = [softmax(test_batch_score_i*tau_rank) for test_batch_score_i in test_batch_score_list]
        test_batch_genscore_tau = [softmax(test_batch_genscore_i*tau_gen) for test_batch_genscore_i in test_batch_genscore]
        new_probs = np.array(test_batch_score_tau_list) + np.array(test_batch_genscore_tau)
    else:
        new_probs = [softmax(test_batch_score_i) for test_batch_score_i in test_batch_score_list] # use softmax    
    record_scores += list(new_probs)
    for idx in range(bsz):
        one_score_list = new_probs[idx]
        if single:
            one_select_idx_list = np.argsort(one_score_list)[::-1]
            one_text_list = test_batch_candidate_text_list[idx]
            # one_pred_text_list = [one_text_list[s_idx] for s_idx in one_select_idx_list]
            if diff_limit != None:
                # diff_limit >= 1 is equivalent to greedy
                if (one_score_list[one_select_idx_list[0]] - one_score_list[one_select_idx_list[1]]) > diff_limit:
                    one_pred_text  = one_text_list[one_select_idx_list[0]]
                else:
                    one_pred_text = one_text_list[0]
            else:
                one_pred_text  = one_text_list[one_select_idx_list[0]]
        else:
            # use probability threshold to select positives
            # TODO: update the evaluation
            one_select_idxs = [s_idx for s_idx, s_i in enumerate(one_score_list) if s_i > threshold]
            # one_select_idxs = [np.argmax(one_score_list)] #either ranker or weighted ranker (greedy if weight=0, ranker if weight=1)
            one_pred_text = [test_batch_candidate_text_list[idx][one_select_idx] for one_select_idx in one_select_idxs]
        pred_text_list.append(one_pred_text)
    
    return pred_text_list


def evaluation(args):
    # load data
    global record_scores
    init_seed()
    if os.path.exists(args.save_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(args.save_prefix, exist_ok=True)

    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print ('Cuda is available.')
    else:
        print ('Cuda is not available.')
    print(f"args.single is {args.single}")
    device = args.gpu_id
    SEP, EOS = '<sep>', '<eos>'
    model_name = args.model_name
    print(args.model_name)
    if model_name.startswith('bert'):
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif model_name.startswith('roberta'):
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    else:
        raise Exception('Wrong Tokenization Mode!')
    special_token_list = [SEP, EOS]
    tokenizer.add_tokens(special_token_list)

    model = Model(args.model_name, tokenizer)
    if cuda_available:
        model = model.cuda(device)

    model_path = glob.glob(args.ckpt_path+"*.bin")[0]
    print(model_path)
    if cuda_available:
        model_ckpt = torch.load(model_path)
    else:
        model_ckpt = torch.load(model_path, map_location='cpu')

    model_parameters = model_ckpt['model']
    model.load_state_dict(model_parameters)
    model.eval()

    train_path = args.candidates_path + '/train/'
    best_params={"weight":1, "threshold":0.5, "tau_gen":0, "tau_rank": 1}

    for header in ['dev','test']:
        print ('------------------------------------------')
        print ('Start inference {} data...'.format(header))
        if header == 'train':
            val_path = args.candidates_path + '/train/'
        elif header == 'dev':
            val_path = args.candidates_path + '/dev/'
        else:
            val_path = args.candidates_path + '/test/'

        data = Data(args.model_name, train_path, val_path, args.train_candi_pool_size, args.train_negative_num, args.test_candi_span, 
                args.max_content_len, args.max_tgt_len)

        test_num = data.test_num
        batch_size = args.batch_size
        test_step_num = int(test_num / batch_size) + 1

        print("Done data loading")
        best_f1_t=0 
        best_f1_detail = None
        print_f1 = False
        ## redeem conf_diff -> useless -> replace by weight
        with torch.no_grad():
            # initialize hyperparameters
            # weight = 1 -> use the ranker result; weight = 0 -> use the generation score only
            tau_gens, tau_ranks, weights, thresholds = [0], [1], [1], [0.5]
            if header == "dev" and not args.plot:
                print ('Evaluation on Development set, searching for hyperparameter...')
                # search over the hyperparameters on development set 
                if args.use_rank:               
                    if not args.single:
                        # use --weight by default
                        # only search through weights and thresholds
                        weights = np.linspace(0, 0.6,13)
                        # weights = [0] # use only the generator
                        # weights = [1] # use only the reranked score
                        thresholds = np.linspace(0.05, 0.5, 10)
                        
                        # weights = [0.2]
                        # thresholds = [0.1]
                    else:
                        if args.tau:
                            tau_gens, tau_ranks = np.linspace(0,1,21), np.linspace(0,1,21)
                        elif args.weight:
                            weights = np.linspace(0,1,41)
                        elif args.diff_limit:
                            weights = np.linspace(0,1,21)
                else:
                    # greedy result
                    # weight = 0 -> use the generation result
                    tau_gens, tau_ranks, weights, thresholds = [0], [1], [0], [0.5]
            else:
                print_f1 = True
                print("Evaluation on test, using the optimal hyperparameter")
                thresholds = [best_params['threshold']]
                weights = [best_params['weight']]
                tau_gens = [best_params['tau_gen']] 
                tau_ranks = [best_params['tau_rank']] 
                print("best params ", best_params)
                print("current model ", args.ckpt_path.split("/")[-2])
            
            if not args.plot:
                for tau_gen in tau_gens:
                    for tau_rank in tau_ranks:
                        for thres in thresholds:
                            for weight in weights:
                                test_ref_text_list, test_pred_text_list = [], []
                                # p = progressbar.ProgressBar(test_step_num)
                                # p.start()
                                for test_step in range(test_step_num):
                                    # p.update(test_step)
                                    test_all_batch_token_list, test_all_batch_mask_list, test_all_batch_seg_list, \
                                    test_batch_summary_text_list, test_batch_candidate_summary_list, test_batch_candidate_genscore_list = data.get_next_test_batch(batch_size)
                                    test_batch_score = trainer(model, test_all_batch_token_list, test_all_batch_mask_list, 
                                        test_all_batch_seg_list, cuda_available, device)
                                    if args.diff_limit:
                                        test_batch_select_text = evaluate_test_pred_text(test_batch_score, test_batch_candidate_summary_list, test_batch_candidate_genscore_list, diff_limit=weight, single=args.single)
                                    elif args.tau:
                                        test_batch_select_text = evaluate_test_pred_text(test_batch_score, test_batch_candidate_summary_list, test_batch_candidate_genscore_list, tau_gen=tau_gen, tau_rank=tau_rank, single=args.single)
                                    else:
                                        test_batch_select_text = evaluate_test_pred_text(test_batch_score, test_batch_candidate_summary_list, test_batch_candidate_genscore_list, weight=weight, threshold=thres, single=args.single)
                                    test_pred_text_list += test_batch_select_text
                                    test_ref_text_list += test_batch_summary_text_list
                                # p.finish()
                                test_ref_text_list = test_ref_text_list[:test_num]
                                test_pred_text_list = test_pred_text_list[:test_num]
                                
                                record_scores

                                if args.single:
                                    _, _, f1_detail = eval_f1_score(test_pred_text_list, test_ref_text_list, toprint=print_f1)
                                else:
                                    _, _, f1_detail = eval_f1_score_multi(test_pred_text_list, test_ref_text_list, toprint=print_f1)

                                if f1_detail[0] > best_f1_t:
                                    best_f1_t = f1_detail[0]
                                    best_params["weight"] = weight
                                    best_params['threshold'] = thres
                                    best_params["tau_gen"] = tau_gen
                                    best_params["tau_rank"] = tau_rank
                                    best_f1_detail = f1_detail

                ## Naming rules
                # Ture      -> use ranker       + weight 0        -> ranker result
                # True      -> use ranker       + other_weight    -> use linear_combination 
                # False     -> not using ranker + weight 1             -> greedy
                
                params_postfix = "_weight_{:.3f}_threshold_{:.3f}".format(best_params['weight'], best_params['threshold'])
                
                if args.use_rank:
                    save_rerank_path = f"_top_{args.test_candi_span}_rerank" + params_postfix + "_result.txt"
                else:
                    save_rerank_path = f"_top_{args.test_candi_span}_greedy" + params_postfix + "_result.txt"

                if args.save_pred:
                    print("------------------save prediction results")
                    idx = 0
                    if "trig" in args.candidates_path:
                        with open(args.save_prefix + '/' + header + save_rerank_path, 'w', encoding = 'utf8') as o:
                            if not args.single:
                                for text1, text2, conte in zip(test_ref_text_list, test_pred_text_list, data.test_content_text_list):
                                    o.writelines(str(idx) + '\n' + conte + '\n' + text1 + "  ----  " + "||".join(text2) + '\n\n')
                            else:
                                idx += 1
                    
                    # save arginput for post_ranker generation
                    with open(args.save_prefix + f"/{header}_ranked_{args.use_rank}" + params_postfix + "_arginput.txt", 'w', encoding = 'utf8') as o:
                        idx_cont = 0
                        for text, conte in zip(test_pred_text_list, data.test_content_text_list):
                            idx_cont+=1
                            if not args.single:
                                if text != []:
                                    for tx_i in text:
                                        o.writelines("EventExtract: " + conte + " " + tx_i.split( " [")[0] + '\n')
                                        # o.writelines("EventExtract: " + conte + " " + tx_i + '\n')
                                else:
                                    o.writelines("EventExtract: " + conte + " " + '\n')
                            else:
                                o.writelines("EventExtract: " + conte + " " + text.split( " [")[0] + '\n')
                    
                    
                    with open(args.save_prefix + f"/event_{header}_ranked_{args.use_rank}" + params_postfix + "_arginput.txt", 'w', encoding = 'utf8') as o:
                            idx_cont = 0
                            for text, conte in zip(test_pred_text_list, data.test_content_text_list):
                                idx_cont+=1
                                if not args.single:
                                    events = []
                                    if text != []:
                                        for tx_i in text:
                                            events.append("["+tx_i.split( " [")[1])

                                        o.writelines(" ||sep|| ".join(events) + "\n")
                                    else:
                                        o.writelines(" ||sep|| ".join(events) + "\n")
                                else:
                                    o.writelines("EventExtract: " + conte + " " + text.split( " [")[0] + '\n')

                    
                    # save argout for post_ranker generation
                    if "trig" not in args.candidates_path:
                        with open(args.save_prefix + f"/test_ranked_{args.use_rank}_weight_{best_params['weight']}_threshold_{best_params['threshold']}_argoutput.txt", 'w', encoding = 'utf8') as o:   
                            for ref_arg in data.test_reference_text_list:
                                o.writelines("] ".join(ref_arg.split("] ")[1:]) + "\n")


            elif header=="test":
                # thres=0.2
                # weights = list(np.linspace(0,1,11))
                # weights.insert(2,0.15)

                thresholds=list(np.linspace(0,1,11))
                thresholds.insert(2,0.15)
                weight=0.4

                trig_is = []
                trig_cs = []
                # for weight in weights:
                for thres in thresholds:
                    test_ref_text_list, test_pred_text_list = [], []
                    # p = progressbar.ProgressBar(test_stepf_num)
                    # p.start()
                    for test_step in range(test_step_num):
                        # p.update(test_step)
                        test_all_batch_token_list, test_all_batch_mask_list, test_all_batch_seg_list, \
                        test_batch_summary_text_list, test_batch_candidate_summary_list, test_batch_candidate_genscore_list = data.get_next_test_batch(batch_size)
                        test_batch_score = trainer(model, test_all_batch_token_list, test_all_batch_mask_list, 
                            test_all_batch_seg_list, cuda_available, device)
                        
                        test_batch_select_text = evaluate_test_pred_text(test_batch_score, test_batch_candidate_summary_list, test_batch_candidate_genscore_list, weight=weight, threshold=thres, single=args.single)
                        test_pred_text_list += test_batch_select_text
                        test_ref_text_list += test_batch_summary_text_list
                    test_ref_text_list = test_ref_text_list[:test_num]
                    test_pred_text_list = test_pred_text_list[:test_num]
                    
                    record_scores

                    if args.single:
                        _, _, f1_detail = eval_f1_score(test_pred_text_list, test_ref_text_list, toprint=print_f1)
                    else:
                        _, _, f1_detail = eval_f1_score_multi(test_pred_text_list, test_ref_text_list, toprint=print_f1)

                # print("For model {}, the {} result is {} \n Trig-I: {:.3f}, Trig-C: {:.3f}, Arg-I: {:.3f}, Arg-C: {:.3f}".format(
                #             args.ckpt_path.split("/")[-2], header, best_params, *best_f1_detail))
                    trig_is.append(round(f1_detail[0], 2))
                    trig_cs.append(round(f1_detail[1], 2))
            
                    params_postfix = "_weight_{:.3f}_threshold_{:.3f}".format(weight, thres)

                    if args.use_rank:
                        save_rerank_path = f"_top_{args.test_candi_span}_rerank" + params_postfix + "_result.txt"
                    else:
                        save_rerank_path = f"_top_{args.test_candi_span}_greedy" + params_postfix + "_result.txt"

                    if args.save_pred:
                        idx = 0
                        if "trig" in args.candidates_path:
                            with open(args.save_prefix + '/' + header + save_rerank_path, 'w', encoding = 'utf8') as o:
                                if not args.single:
                                    for text1, text2, conte in zip(test_ref_text_list, test_pred_text_list, data.test_content_text_list):
                                        o.writelines(str(idx) + '\n' + conte + '\n' + text1 + "  ----  " + "||".join(text2) + '\n\n')
                                else:
                                    idx += 1
                        
                        # save arginput for post_ranker generation
                        with open(args.save_prefix + f"/{header}_ranked_{args.use_rank}" + params_postfix + "_arginput.txt", 'w', encoding = 'utf8') as o:
                            idx_cont = 0
                            for text, conte in zip(test_pred_text_list, data.test_content_text_list):
                                idx_cont+=1
                                if not args.single:
                                    if text != []:
                                        for tx_i in text:
                                            o.writelines("EventExtract: " + conte + " " + tx_i.split( " [")[0] + '\n')
                                            # o.writelines("EventExtract: " + conte + " " + tx_i + '\n')
                                    else:
                                        o.writelines("EventExtract: " + conte + " " + '\n')
                                else:
                                    o.writelines("EventExtract: " + conte + " " + text.split( " [")[0] + '\n')
                        
                        with open(args.save_prefix + f"/event_{header}_ranked_{args.use_rank}" + params_postfix + "_arginput.txt", 'w', encoding = 'utf8') as o:
                            idx_cont = 0
                            for text, conte in zip(test_pred_text_list, data.test_content_text_list):
                                idx_cont+=1
                                if not args.single:
                                    events = []
                                    if text != []:
                                        for tx_i in text:
                                            # if len(tx_i.split(" [")) < 2:
                                            #     print(tx_i)
                                            events.append("["+tx_i.split("[")[1])
                                        o.writelines(" ||sep|| ".join(events) + "\n")
                                    else:
                                        o.writelines(" ||sep|| ".join(events) + "\n")
                                else:
                                    o.writelines("EventExtract: " + conte + " " + text.split( " [")[0] + '\n')
                
                print(f"trig_results for weight {weight}, threshold {thres}")
                print(trig_is, '\n', trig_cs)


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def eval_f1_score(p_texts, g_texts, recorder=None, toprint=False):
    tp_trig, num_p_trig, num_g_trig = 0, 0, 0
    tp_et, num_p_et, num_g_et = 0, 0, 0
    tp_arg, num_p_arg, num_g_arg = 0, 0, 0
    tp_rol, num_p_rol, num_g_rol = 0, 0, 0
    p_trig_list, g_trig_list = [], []
    which_wrong = 0
    for p_text, g_text in zip(p_texts, g_texts):
        # ground truth triggerword
        try:
            tag = re.search("\<\|triggerword\|\> (.*?) \[", g_text)
            g_trigger = tag.group()[16:-2]
            g_trig_list.append(g_trigger)
        except:
            # in 1177, there is an error
            print("error ---------------- ", which_wrong)
            print(g_text)
        num_g_trig += 1

        # get predicted triggerword
        p_trigger = ''
        tag = re.search("\<\|triggerword\|\> (.*?) \[", p_text)
        if tag:
            p_trigger = tag.group()[16:-2]
            num_p_trig += 1
            p_trig_list.append(p_trigger)
        else:
            p_trig_list.append("[None]")
        if p_trigger == g_trigger:
            tp_trig += 1

        which_wrong +=1
        
        # ground truth trigger_event
        tag = re.search("\<\|triggerword\|\> (.*?) \[[^ />][^>]*\]", g_text)
        g_event = tag.group()[16:]
        num_g_et += 1

        # get predicted trigger_event
        p_event = ''
        tag = re.search("\<\|triggerword\|\> (.*?) \[[^ />][^>]*\]", p_text)
        if tag:
            p_event = tag.group()[16:]
            num_p_et += 1
        if p_event == g_event:
            tp_et += 1

        g_roles = re.findall('<\|/[^/>][^>]*\|>', g_text)
        g_roles = [x[3:-2] for x in g_roles]
        g_args = []
        g_role_args = []
        for role_i in g_roles:
            tag = re.search("<\|{0}\|> (.*?) <\|/{0}\|>".format(role_i), g_text)
            if tag:
                if tag.group(1) == "[None]":
                    continue
                temp_args = tag.group(1).split(" [and] ")
                for temp_arg in temp_args:
                    g_args.append(temp_arg)
                    g_role_args.append({role_i: temp_arg})
        num_g_arg += len(g_args)
        num_g_rol += len(g_role_args)

        p_roles = re.findall('<\|/[^/>][^>]*\|>', p_text)
        p_roles = [x[3:-2] for x in p_roles]
        p_args = []
        p_role_args = []
        for role_i in p_roles:
            tag = re.search("<\|{0}\|> (.*?) <\|/{0}\|>".format(role_i), p_text)
            if tag:
                if tag.group(1) == "[None]":
                    continue
                temp_args = tag.group(1).split(" [and] ")
                for temp_arg in temp_args:
                    p_args.append(temp_arg)
                    p_role_args.append({role_i: temp_arg})
        num_p_arg += len(p_args)
        num_p_rol += len(p_role_args)

        for p_arg in p_args:
            if p_arg in g_args:
                tp_arg += 1
                g_args.remove(p_arg)

        for p_role_arg in p_role_args:
            if p_role_arg in g_role_args:
                tp_rol += 1
                g_role_args.remove(p_role_arg)

    avg_t, avg_a, f1_ai, f1_ac = 0,0,0,0

    pre, rec = safe_div(tp_trig, num_p_trig), safe_div(tp_trig, num_g_trig)
    f1_ti = 2*safe_div(pre*rec, (pre+rec))*100
    pre, rec = safe_div(tp_et,num_p_et), safe_div(tp_et,num_g_et)
    f1_tc = 2*safe_div(pre*rec, (pre+rec))*100
    avg_t = (f1_ti+f1_tc)/2

    if toprint:
        if recorder:
            recorder.print("Trigger I ---- tp/total {:4d}/{:4d}, f1 {:6.2f}".format(tp_trig, num_g_trig, f1_ti))

            recorder.print("Trigger C ---- tp/total {:4d}/{:4d}, {:6.2f}".format(tp_et, num_g_et, f1_tc))
        else:
            print("Trigger C ---- tp/total {:4d}/{:4d}, f1 {:6.2f}".format(tp_et, num_g_et, f1_tc))
            print("Trigger I ---- tp/total {:4d}/{:4d}, f1 {:6.2f}".format(tp_trig, num_g_trig, f1_ti))

    return avg_t, avg_a, [f1_ti, f1_tc, f1_ai, f1_ac]


def eval_f1_score_multi(p_texts_list, g_texts, recorder=None, toprint=False):
    g_texts_list = [g_i.split("\n ") for g_i in g_texts] # concatenated list -> len(g_texts) == num_instances == 305 for dev
    assert len(p_texts_list) == len(g_texts_list)

    tp_trig, num_p_trig, num_g_trig = 0, 0, 0
    tp_et, num_p_et, num_g_et = 0, 0, 0
    tp_arg, num_p_arg, num_g_arg = 0, 0, 0
    tp_rol, num_p_rol, num_g_rol = 0, 0, 0
    p_trig_list, g_trig_list = [], []
    which_wrong = 0
    for p_text_list, g_text_list in zip(p_texts_list, g_texts_list):
        one_g_trig_list, one_p_trig_list = [], []
        one_g_event_list, one_p_event_list = [], []
        # if len(g_text_list) != 1:
        #     continue
        for g_text in g_text_list:
            # ground truth triggerword
            try:
                tag = re.search("\<\|triggerword\|\> (.*?) \[", g_text)
                g_trigger = tag.group()[16:-2]
                one_g_trig_list.append(g_trigger)
                num_g_trig += 1
            except:
                # in 1177, there is an error
                print("error ---------------- ", which_wrong)
                print(g_text)        
            
            # ground truth trigger_event
            tag = re.search("\<\|triggerword\|\> (.*?) \[[^ />][^>]*\]", g_text)
            one_g_event_list.append(tag.group()[16:])
            num_g_et += 1
        g_trig_list.append(one_g_trig_list)

        for p_text in p_text_list:
            # get predicted triggerword
            p_trigger = ''
            tag = re.search("\<\|triggerword\|\> (.*?) \[", p_text)
            if tag:
                p_trigger = tag.group()[16:-2]
                num_p_trig += 1
                one_p_trig_list.append(p_trigger)
            else:
                one_p_trig_list.append("[None]")
            if p_trigger in one_g_trig_list:
                tp_trig += 1
                one_g_trig_list.remove(p_trigger)
            
            # get predicted trigger_event
            p_event = ''
            tag = re.search("\<\|triggerword\|\> (.*?) \[[^ />][^>]*\]", p_text)
            if tag:
                p_event = tag.group()[16:]
                one_p_event_list.append(p_event)
                num_p_et += 1
            if p_event in one_g_event_list:
                tp_et += 1
                one_g_event_list.remove(p_event)
        p_trig_list.append(one_p_trig_list)

        which_wrong +=1

    avg_t, avg_a, f1_ai, f1_ac = 0,0,0,0

    pre, rec = safe_div(tp_trig, num_p_trig), safe_div(tp_trig, num_g_trig)
    f1_ti = 2*safe_div(pre*rec, (pre+rec))*100
    pre, rec = safe_div(tp_et,num_p_et), safe_div(tp_et,num_g_et)
    f1_tc = 2*safe_div(pre*rec, (pre+rec))*100
    avg_t = (f1_ti+f1_tc)/2
    if toprint:
        if recorder:
            recorder.print("Trigger I ---- tp/total_g/total_p {:4d}/{:4d}/{:4d}, f1 {:6.2f}".format(tp_trig, num_g_trig, num_p_trig, f1_ti))
            recorder.print("Trigger C ---- tp/total_g/total_p {:4d}/{:4d}/{:4d}, d1 {:6.2f}".format(tp_et, num_g_et, num_p_et, f1_tc))
        else:
            print("Trigger I ---- tp/total_g/total_p {:4d}/{:4d}/{:4d}, f1 {:6.2f}".format(tp_trig, num_g_trig, num_p_trig, f1_ti))
            print("Trigger C ---- tp/total_g/total_p {:4d}/{:4d}/{:4d}, f1 {:6.2f}".format(tp_et, num_g_et, num_p_et, f1_tc))

    return avg_t, avg_a, [f1_ti, f1_tc, f1_ai, f1_ac]


# hinge loss used to train the rank model
def hinge_loss(scores, margin):
    # y_pred: bsz x candi_num
    loss = torch.nn.functional.relu(margin - (torch.unsqueeze(scores[:, 0], -1) - scores[:, 1:]))
    return torch.mean(loss)


## extract the final predicted result for test set
def extract_test_pred_text(test_batch_score, test_batch_candidate_text_list, test_batch_genscore):
    pred_text_list = []
    test_batch_score_list = test_batch_score.detach().cpu().numpy()
    
    # print(test_batch_score_list)
    if args.weight:
        # during training, use default weight 0.5 for evaluation if weighted
        test_batch_score_list = [softmax(test_batch_score_i) for test_batch_score_i in test_batch_score_list] # use softmax
        test_batch_genscore = [softmax(test_batch_genscore_i) for test_batch_genscore_i in test_batch_genscore]
        weight_probs = np.array(test_batch_score_list)*args.tune_weight + np.array(test_batch_genscore)*(1-args.tune_weight)
    
    bsz = len(test_batch_score_list)
    for idx in range(bsz):
        # one_score_list = test_batch_score_list[idx]
        one_score_list = weight_probs[idx]
        if args.single:
            one_select_idx = np.argsort(one_score_list)[::-1][0]
            one_select_text = test_batch_candidate_text_list[idx][one_select_idx]
        else:
            # use probability threshold to select positives
            one_select_idxs = [s_idx for s_idx, s_i in enumerate(one_score_list) if s_i > args.threshold]
            one_select_text = [test_batch_candidate_text_list[idx][one_select_idx] for one_select_idx in one_select_idxs]
        pred_text_list.append(one_select_text)
    
    return pred_text_list


def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--model_name', type=str, help="e.g. bert-base...")
    parser.add_argument('--candidates_path', type=str, help="the path that stores the generated candidates")
    parser.add_argument('--train_candi_pool_size', type=int, default=10, 
        help="Randomly selecting negative examples from the top-k retrieved candidates provided by the IR system.")
    parser.add_argument('--train_negative_num', type=int, default=10, 
        help="number of randomly selected negatives from the retrieved candidates from the IR system., must be smaller than train_candi_pool_size")
    parser.add_argument('--test_candi_span', type=int, default=10, 
        help="reranking the best response from top-n candidates from the IR system.")
    parser.add_argument('--max_content_len', type=int, default=512)
    parser.add_argument('--max_tgt_len', type=int, default=200)
    parser.add_argument("--seed", type=int, default=528)
    # training configuration
    parser.add_argument('--multigpu', action="store_true", default=False)
    parser.add_argument('--loss_margin', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--total_steps', type=int)
    parser.add_argument('--update_steps', type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=40)
    parser.add_argument('--tune_weight', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--single', action="store_true", default=False, help="if on single events only")
    # learning configuration
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_output_dir', type=str, default='cache_eval/')
    # evaluation configuration
    parser.add_argument('--set_weight', type=float, default=0.25, help="check on effect of weight")
    parser.add_argument('--evaluate', action="store_true", default=False, help="if evaluate")
    parser.add_argument('--save_pred', action="store_true", default=False, help="save prediction")
    parser.add_argument('--use_rank', action="store_true", default=False, help="if use rank")
    parser.add_argument('--plot', action="store_true", default=False, help="if evaluate")
    parser.add_argument('--ckpt_path', type=str, help="path of pre-trained prototype selector.")
    parser.add_argument('--save_prefix', type=str, help="path prefix to save the reranked context.")
    parser.add_argument('--diff_limit', action="store_true", default=False, help="if use confidence limit")
    parser.add_argument('--weight',  action="store_true", default=False, help="if use linear combination")
    parser.add_argument('--tau', action="store_true", default=False, help="if use temperature (on ligits")
    return parser.parse_args()


def run(args):
    id = args.candidates_path.split("/")[2] + "/" + time.strftime('%Y%m%d_%H%M%S', time.localtime())
    recorder = Recorder(id)

    init_seed(seed=args.seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        recorder.print('Cuda is available.')
    else:
        recorder.print('Cuda is not available.')

    device = args.gpu_id
    
    train_path = args.candidates_path + '/train/'
    dev_path = args.candidates_path + '/dev/'

    model_name = args.model_name
    data = Data(args.model_name, train_path, dev_path, args.train_candi_pool_size, args.train_negative_num, args.test_candi_span, 
                args.max_content_len, args.max_tgt_len)
    
    recorder.print('Initializing Model...')
    model = Model(args.model_name, data.tokenizer)
    if cuda_available:
        if args.multigpu:
            model = torch.nn.DataParallel(model)
            model.cuda()
        else:
            model = model.cuda(device)
        recorder.print('Model Loaded.')

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
            num_training_steps=args.total_steps)
    optimizer.zero_grad()

    train_num, test_num = data.train_num, data.test_num
    batch_size = args.batch_size
    train_step_num = int(train_num / batch_size) + 1
    test_step_num = int(test_num / batch_size) + 1

    batches_processed = 0
    total_steps = args.total_steps
    print_every, eval_every = args.print_every, args.eval_every
    train_loss_accum, max_test_f1 = 0, 0.
    model.train()
    test_output_dir = args.test_output_dir + "/" + id
    recorder.write_config(args, [model], __file__)
    cnt = 0
    for one_step in range(total_steps):
        if cnt == 200:
            print("stop early")
            break
        epoch = one_step // train_step_num
        batches_processed += 1

        train_all_batch_token_list, train_all_batch_mask_list, train_all_batch_seg_list = \
        data.get_next_train_batch(batch_size)
        train_batch_score = trainer(model, train_all_batch_token_list, train_all_batch_mask_list, 
            train_all_batch_seg_list, cuda_available, device, args.multigpu)
        train_loss = hinge_loss(train_batch_score, args.loss_margin)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if batches_processed % args.update_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        train_loss_accum += train_loss.item()
        if batches_processed % print_every == 0:
            curr_train_loss = train_loss_accum / print_every
            recorder.print('At epoch %d, batch %d, train loss %.5f, max score is %.5f' % 
                (epoch, batches_processed, curr_train_loss, max_test_f1))
            train_loss_accum = 0.
            recorder.plot("loss", {"loss": curr_train_loss}, one_step)


        if batches_processed % eval_every == 0:
            model.eval()
            test_ref_text_list, test_pred_text_list = [], []
            with torch.no_grad():
                recorder.print('Test Evaluation...')
                p = progressbar.ProgressBar(test_step_num)
                p.start()
                for test_step in range(test_step_num):
                    p.update(test_step)
                    test_all_batch_token_list, test_all_batch_mask_list, test_all_batch_seg_list, \
                    test_batch_summary_text_list, test_batch_candidate_summary_list, test_batch_candidate_genscore_list = data.get_next_test_batch(batch_size)
                    test_batch_score = trainer(model, test_all_batch_token_list, test_all_batch_mask_list, 
                        test_all_batch_seg_list, cuda_available, device, args.multigpu)
                    test_batch_select_text = extract_test_pred_text(test_batch_score, test_batch_candidate_summary_list, test_batch_candidate_genscore_list)
                    test_pred_text_list += test_batch_select_text
                    test_ref_text_list += test_batch_summary_text_list
                p.finish()
                test_ref_text_list = test_ref_text_list[:test_num]
                test_pred_text_list = test_pred_text_list[:test_num]
                
                if args.single:
                    _, _, f1_detail = eval_f1_score(test_pred_text_list, test_ref_text_list, recorder, toprint=True)
                else:
                    _, _, f1_detail = eval_f1_score_multi(test_pred_text_list, test_ref_text_list, recorder, toprint=True)
                
                recorder.plot("Trig-I", {"Trig-I": f1_detail[0]}, one_step)
                recorder.plot("Trig-C", {"Trig-C": f1_detail[1]}, one_step)   
                recorder.print('----------------------------------------------------------------')
                recorder.print('At epoch %d, batch %d, dev f1ti %5f, dev f1tc %5f' \
                    % (epoch, batches_processed, f1_detail[0], f1_detail[1]))
                save_name = '/epoch_%d_batch_%d_dev_f1ti_%.3f_f1tc_%.3f.bin' \
                        % (epoch, batches_processed, f1_detail[0], f1_detail[1])
                recorder.print('----------------------------------------------------------------')

                if f1_detail[0] > max_test_f1:
                    # keep track of model's best result
                    recorder.print('Saving Model...')
                    model_save_path = test_output_dir + save_name
                    import os
                    if os.path.exists(test_output_dir):
                        pass
                    else: # recursively construct directory
                        os.makedirs(test_output_dir, exist_ok=True)
                    if args.multigpu:
                        torch.save({'model':model.module.state_dict()}, model_save_path)
                    else:
                        print(model_save_path)
                        torch.save({'model':model.state_dict()}, model_save_path)

                    max_test_f1 = f1_detail[0]
                    fileData = {}
                    for fname in os.listdir(test_output_dir):
                        if fname.startswith('epoch'):
                            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                        else:
                            pass
                    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                    
                    if len(sortedFiles) < 1:
                        pass
                    else:
                        delete = len(sortedFiles) - 1
                        for x in range(0, delete):
                            os.remove(test_output_dir + '/' + sortedFiles[x][0])
                    cnt=0
                else:
                    cnt+=1
            model.train()


if __name__ ==  "__main__":
    args = parse_config()
    if args.evaluate:
        evaluation(args)
    else:
        run(args)
