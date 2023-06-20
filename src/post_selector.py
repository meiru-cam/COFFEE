import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, T5Tokenizer, BartTokenizer
from model import GenerativeModel, Prefix_fn_cls
from data import EEDataset
from utils import cal_scores, get_span_idxs, get_span_idxs_zh
from argparse import ArgumentParser, Namespace
import re
from sklearn.metrics import f1_score
import argparse
from copy import deepcopy

torch.backends.cudnn.enabled = False

## TODO: not in batch, will it matters? -> evaluation batch_size=1

def initialize_model(args):
    ## initialize tokenizer and model


    # TODO: remember to change the vocab when evaluating on different scheme
    with open(args.vocab_path) as f:
        vocab = json.load(f)
    config_path = args.config_path # use the config that was used to train the model
    with open(config_path) as fp:
        config = json.load(fp)
    config = Namespace(**config)
    # set beam_size
    config.beam_size = args.beam_size

    # tokenizer
    if config.model_name.startswith("google/mt5-"):
        tokenizer = MT5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    elif config.model_name.startswith("copy+google/mt5-"):
        model_name = config.model_name.split('copy+', 1)[1]
        tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
    elif config.model_name.startswith("t5-"):
        tokenizer = T5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    elif config.model_name.startswith("copy+t5-"):
        model_name = config.model_name.split('copy+', 1)[1]
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
    elif config.model_name.startswith("facebook/bart-") or config.model_name.startswith("bart-"):
        tokenizer = BartTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    else:
        raise NotImplementedError

    # set GPU device
    torch.cuda.set_device(0)
    # load Generator
    # TODO: modify model_dir if changed the scheme
    model_dir = args.ckpt_path
    model = GenerativeModel(config, tokenizer)
    model.load_state_dict(torch.load(model_dir, map_location=f'cuda:0'))
    model.cuda(device=0)
    model.eval()
    return model, tokenizer, config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arginput_path", type=str, help="the path of arginput.txt generated by ranker")
    parser.add_argument("--argoutput_path", type=str, help="the path of the argoutput.txt, contain arguments output")
    parser.add_argument("--refevent_path", type=str, help="the path of the ref_event.txt, contain ground truth event type predictions")
    parser.add_argument("--ckpt_path", type=str, help="the path to trained mdl file")
    parser.add_argument("--config_path", type=str, help="the config file used to trian the model")
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--beam_size", type=int, default=10, help="beam size for generator")
    return parser.parse_args()


def load_data(args):
    ## load input data
    # load the result from best Ranker
    # TODO: modify ranker result dir if changed the scheme
    with open(args.arginput_path) as f:
        lc_arginputs = []
        for line in f.readlines():
            if line != "\n":
                line = line.strip()    
                lc_arginputs.append(line)

    # laod the output -> should be the same for different schemes (contains only the argument-output)
    with open(args.argoutput_path) as f:
        ref_args = [line.strip() for line in f.readlines()]
    
    new_path = args.arginput_path.split("/")
    new_path[-1] = "event_"+new_path[-1]
    predeventpath = "/".join(new_path)

    with open(predeventpath) as f:
        lc_events = [line.strip() for line in f.readlines()]
        lc_events_list = [g_i.split(" ||sep|| ") for g_i in lc_events] 

    with open(args.refevent_path) as f:
        ref_events = [line.strip() for line in f.readlines() if line != "\n"]
        ref_events_list = [g_i.split(" ||sep|| ") for g_i in ref_events]
    
    return lc_arginputs, ref_args, lc_events_list, ref_events_list


def argument_gen(args, config, model, tokenizer, lc_arginputs):
    lc_pred_list, greedy_pred_list = [], []
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #GPU-WARM-UP
    for mode, arginputs in zip(["lc"], [lc_arginputs][:20]):
        for argin in arginputs:
            passage = argin.split(" <|triggerword")[0]

            eae_inputs = tokenizer(argin, return_tensors='pt', padding=True, max_length=config.max_length+2)
            enc_idxs = eae_inputs['input_ids']
            enc_idxs = enc_idxs.cuda()
            enc_attn = eae_inputs['attention_mask'].cuda()

            if config.beam_size == 1:
                model.model._cache_input_ids = enc_idxs
            else:
                expanded_return_idx = (
                    torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                )
                input_ids = enc_idxs.index_select(0, expanded_return_idx)
                model.model._cache_input_ids = input_ids
            
            # inference
            with torch.no_grad():
                # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                    num_beams=config.beam_size, max_length=config.max_output_length,
                    forced_bos_token_id=None)
    ## start argument generation
    # MEASURE PERFORMANCE
    timings = []
    map_passage_predout = {}
    for mode, arginputs in zip(["lc"], [lc_arginputs]):
        # selected_index = [21, 58, 104, 163]
        # arginputs = np.array(arginputs)[selected_index]
        # ref_args = np.array(ref_args)[selected_index]
        for argin in arginputs:
            passage = argin.split(" <|triggerword")[0] # used for map multi-event instances
            if "<|triggerword|>" not in argin:
                if passage not in map_passage_predout.keys():
                    map_passage_predout[passage] = [""]
                else:
                    map_passage_predout[passage].append("")
                continue
            eae_inputs = tokenizer(argin, return_tensors='pt', padding=True, max_length=config.max_length+2)
            enc_idxs = eae_inputs['input_ids']
            enc_idxs = enc_idxs.cuda()
            enc_attn = eae_inputs['attention_mask'].cuda()

            if config.beam_size == 1:
                model.model._cache_input_ids = enc_idxs
            else:
                expanded_return_idx = (
                    torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                )
                input_ids = enc_idxs.index_select(0, expanded_return_idx)
                model.model._cache_input_ids = input_ids
            
            # inference
            with torch.no_grad():
                starter.record()
                # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                    num_beams=config.beam_size, max_length=config.max_output_length,
                    forced_bos_token_id=None)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

            
            # eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
            eae_pred_outputs = [tokenizer.decode(outputs[0]).replace("<unk> ", "<").replace("<pad> ", "").replace(" </s>", "")]
            if mode == "lc":
                for i in eae_pred_outputs:
                    if passage not in map_passage_predout.keys():
                        map_passage_predout[passage] = [i]
                    else:
                        map_passage_predout[passage].append(i)
            
                    # lc_pred_list.append(i)
                    # lc_gold_list.append(ref_arg)
            else:
                for i in eae_pred_outputs:
                    greedy_pred_list.append(i)
            
    lc_pred_list = list(map_passage_predout.values())
    mean_syn = np.sum(timings) / len(timings)
    std_syn = np.std(timings)
    print(f"total number of instances: {len(timings)}, total time: {np.sum(timings)}, mean_syn: {mean_syn}, std_syn: {std_syn}")
    return lc_pred_list

# idx=0
# with open("./log/lc_conf_diff3.txt", 'w', encoding = 'utf8') as o:
#     for lc_out, conf_out, greedy_out, gold in zip(lc_pred_list, conf_pred_list, greedy_pred_list, lc_gold_list):
#         # if lc_out!=gold or conf_out!=gold:
#         o.writelines(str(idx)+'\n' + "gold  : "+gold+'\n' + "greedy: "+greedy_out+"\n" + "lc    : "+lc_out+"\n" + "conf  : "+conf_out +"\n")
#         idx += 1

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def eval_f1_score(p_texts, g_texts, recorder=None):
    tp_arg, num_p_arg, num_g_arg = 0, 0, 0
    tp_rol, num_p_rol, num_g_rol = 0, 0, 0
    for p_text, g_text in zip(p_texts, g_texts):
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
        
        # print("p_roles: {}, g_roles: {}".format(p_role_args, g_role_args))

        for p_arg in p_args:
            if p_arg in g_args:
                tp_arg += 1
                g_args.remove(p_arg)

        for p_role_arg in p_role_args:
            if p_role_arg in g_role_args:
                tp_rol += 1
                g_role_args.remove(p_role_arg)

    pre, rec = safe_div(tp_arg, num_p_arg), safe_div(tp_arg, num_g_arg)
    f1_ai = 2*pre*rec/(pre+rec)*100
    # if recorder:
    #     recorder.print("Argument I ---- tp/total {:4d}/{:4d}, precision: {:6.2f}, recall {:6.2f}, f1 {:6.2f}".format(tp_arg, num_g_arg, pre*100, rec*100, f1))
    # else:
    #     print("Argument I ---- tp/total {:4d}/{:4d}, precision: {:6.2f}, recall {:6.2f}, f1 {:6.2f}".format(tp_arg, num_g_arg, pre*100, rec*100, f1))
    print(tp_arg, num_p_arg, num_g_arg)
    if recorder:
        recorder.print("Argument I ---- tp/total {:4d}/{:4d}, {:.3f} | {:.3f} | {:.3f}".format(tp_arg, num_g_arg, pre*100, rec*100, f1_ai))
    else:
        print("Argument I ---- tp/total {:4d}/{:4d}, {:.3f} | {:.3f} | {:.3f}".format(tp_arg, num_g_arg, pre*100, rec*100, f1_ai))


    print(tp_rol, num_p_rol, num_g_rol)
    pre, rec = safe_div(tp_rol,num_p_rol), safe_div(tp_rol,num_g_rol)
    f1_ac = 2*pre*rec/(pre+rec)*100
    # if recorder:
    #     recorder.print("Argument C ---- tp/total {:4d}/{:4d}, precision: {:6.2f}, recall {:6.2f}, f1 {:6.2f}".format(tp_rol, num_g_rol, pre*100, rec*100, f1))
    # else:
    #     print("Argument C ---- tp/total {:4d}/{:4d}, precision: {:6.2f}, recall {:6.2f}, f1 {:6.2f}".format(tp_rol, num_g_rol, pre*100, rec*100, f1))
    if recorder:
        recorder.print("Argument C ---- tp/total {:4d}/{:4d},  {:.3f} | {:.3f} | {:.3f}".format(tp_rol, num_g_rol, pre*100, rec*100, f1_ai))
    else:
        print("Argument C ---- tp/total {:4d}/{:4d},  {:.3f} | {:.3f} | {:.3f}".format(tp_rol, num_g_rol, pre*100, rec*100, f1_ac))


    avg_a = (f1_ai+f1_ac)/2
    return avg_a


def eval_f1_score_multi(p_texts_list, g_texts, pred_event_types, ref_event_types, recorder=None, toprint=False):
    # if not single only, the test_batch_select_text will be a list of list
    # p_texts_list is a concatenated list
    # g_texts_list is a list of string, some of the string might be long -> multiple event
    
    # for single events
    # g_texts_list = [g_i.split(" ||sep|| ") for g_i in g_texts if " ||sep|| " not in g_i] 
    g_texts_list = [g_i.split(" ||sep|| ") for g_i in g_texts] 

    # concatenated list -> len(g_texts) == num_instances == 305 for dev

    # check whether the number of instances matched
    # assert len(p_texts_list) == len(g_texts_list)
    tp_arg, num_p_arg, num_g_arg = 0, 0, 0
    tp_rol, num_p_rol, num_g_rol = 0, 0, 0
    p_arg_list, g_arg_list = [], []
    p_role_list, g_role_list = [], []

    for p_text_list, g_text_list, p_event_list, g_event_list in zip(p_texts_list, g_texts_list, pred_event_types, ref_event_types):
        # p_text_list = p_text_list[:1]
        one_p_args, one_g_args = [], []
        one_p_role_args, one_g_role_args = [], []

        assert len(p_event_list) == len(p_text_list)
        assert len(g_event_list) == len(g_text_list)

        ##  this is for arguments
        for (g_text, g_event) in zip(g_text_list, g_event_list):
            g_roles = re.findall('<\|/[^/>][^>]*\|>', g_text)
            g_roles = [x[3:-2] for x in g_roles]
            for role_i in g_roles:
                tag = re.search("<\|{0}\|> (.*?) <\|/{0}\|>".format(role_i), g_text)
                if tag:
                    if tag.group(1) == "[None]":
                        continue
                    temp_args = tag.group(1).split(" [and] ")
                    for temp_arg in temp_args:
                        temp_arg = temp_arg.replace(" . ", ". ")
                        # one_g_args.append(tuple([g_event.split("_")[-1][:-1], temp_arg]))
                        # one_g_role_args.append(tuple([g_event.split("_")[-1][:-1], role_i, temp_arg]))
                        one_g_args.append(tuple([g_event.split("_")[-1], temp_arg]))
                        one_g_role_args.append(tuple([g_event.split("_")[-1], role_i, temp_arg]))
        
        num_g_arg += len(one_g_args)
        num_g_rol += len(one_g_role_args)
        g_arg_list.extend(one_g_args)
        g_role_list.extend(one_g_role_args)
        
        for (p_text, p_event) in zip(p_text_list, p_event_list):
            p_roles = re.findall('<\|/[^/>][^>]*\|>', p_text)
            p_roles = [x[3:-2] for x in p_roles]
            for role_i in p_roles:
                tag = re.search("<\|{0}\|> (.*?) <\|/{0}\|>".format(role_i), p_text)
                if tag:
                    if tag.group(1) == "[None]":
                        continue
                    temp_args = tag.group(1).split(" [and] ")
                    for temp_arg in temp_args:
                        one_p_args.append(tuple([p_event.split("_")[-1], temp_arg]))
                        one_p_role_args.append(tuple([p_event.split("_")[-1], role_i, temp_arg]))

        num_p_arg += len(one_p_args)
        num_p_rol += len(one_p_role_args)
        
        p_arg_list.extend(one_p_args)
        p_role_list.extend(one_p_role_args)
        
        one_g_args_copy = deepcopy(one_g_args)
        for p_arg in one_p_args:
            if p_arg in one_g_args_copy:
                tp_arg += 1
                one_g_args_copy.remove(p_arg)

        one_g_role_args_copy = deepcopy(one_g_role_args)
        for p_role_arg in one_p_role_args:
            if p_role_arg in one_g_role_args_copy:
                tp_rol += 1
                one_g_role_args_copy.remove(p_role_arg)

    avg_a, f1_ai, f1_ac = 0,0,0

    pre, rec = safe_div(tp_arg, num_p_arg), safe_div(tp_arg, num_g_arg)
    f1_ai = 2*safe_div(pre*rec, (pre+rec))*100
    pre, rec = safe_div(tp_rol, num_p_rol), safe_div(tp_rol, num_g_rol)
    f1_ac = 2*safe_div(pre*rec, (pre+rec))*100
    avg_a = (f1_ai+f1_ac)/2

    print("Argument I ---- tp/total_g/total_p: {:4d}/{:4d}/{:4d}, f1: {:6.2f}".format(tp_arg, num_g_arg, num_p_arg, f1_ai))
    print("Argument C ---- tp/total_g/total_p: {:4d}/{:4d}/{:4d}, f1: {:6.2f}".format(tp_rol, num_g_rol, num_p_rol, f1_ac))

    return avg_a, [f1_ai, f1_ac]


def main():
    args = parse_config()
    model, tokenizer, config = initialize_model(args)
    print("Done initlization")
    lc_arginputs, ref_args, pred_event_types, ref_event_types = load_data(args)
    lc_pred_list = argument_gen(args, config, model, tokenizer, lc_arginputs)
    texts_all = [(" ||sep|| ").join(i) for i in lc_pred_list]
    
    eval_f1_score_multi(lc_pred_list, ref_args, pred_event_types, ref_event_types)




# eval_f1_score(greedy_pred_list, lc_gold_list)

if __name__ ==  "__main__":
    main()

