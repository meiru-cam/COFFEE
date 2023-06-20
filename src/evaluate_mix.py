import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, T5Tokenizer
from model import GenerativeModel, Prefix_fn_cls
from data import EEDataset
from utils import cal_scores, get_span_idxs, get_span_idxs_zh
from argparse import ArgumentParser, Namespace
import re
from sklearn.metrics import f1_score

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('--constrained_decode', default=False, action='store_true')
parser.add_argument('--beam', type=int, default=4)
parser.add_argument('--num_return', type=int, default=1)
parser.add_argument('--type', type=str, default="sep")
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config = Namespace(**config)

# over write beam size
config.beam_size = args.beam

# import template file
if config.dataset == "ace05":
    from template_generate_ace import event_template_generator, IN_SEP, ROLE_LIST, NO_ROLE, AND
    TEMP_FILE = "template_generate_ace"
elif config.dataset == "ere":
    from template_generate_ere import event_template_generator, IN_SEP, ROLE_LIST, NO_ROLE, AND
    TEMP_FILE = "template_generate_ere"
else:
    raise NotImplementedError

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
torch.cuda.set_device(config.gpu_device)

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
        
# check valid styles
assert np.all([style in ['triggerword', 'template'] for style in config.input_style])
assert np.all([style in ['argument:roletype'] for style in config.output_style])

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
else:
    raise NotImplementedError

special_tokens = []
sep_tokens = []
if "triggerword" in config.input_style:
    sep_tokens += [IN_SEP["triggerword"]]
if "template" in config.input_style:
    sep_tokens += [IN_SEP["template"]]
if "argument:roletype" in config.output_style:
    special_tokens += [f"<|{r}|>" for r in ROLE_LIST]
    special_tokens += [f"<|/{r}|>" for r in ROLE_LIST]
    special_tokens += [NO_ROLE, AND]

# special_tokens += ['[PER]', '[ORG]', '[FAC]', '[LOC]', '[WEA]', '[GPE]', '[VEH]']
# special_tokens += ['[\PER]', '[\ORG]', '[\FAC]', '[\LOC]', '[\WEA]', '[\GPE]', '[\VEH]']

# tokenizer.add_tokens(sep_tokens+special_tokens)

# load data
dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

with open(config.vocab_file) as f:
    vocab = json.load(f)

# load model
logger.info(f"Loading model from {args.model}")
model = GenerativeModel(config, tokenizer)
model.load_state_dict(torch.load(args.model, map_location=f'cuda:{config.gpu_device}'))
model.cuda(device=config.gpu_device)
model.eval()

# output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.type == 'sep':
    for data_set, batch_num, data_type in zip([dev_set, test_set], [dev_batch_num, test_batch_num], ['dev', 'test']):
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc=data_type)
        gold_triggers, gold_roles, pred_roles = [], [], []
        pred_wnd_ids, gold_outputs, pred_outputs, inputs = [], [], [], []
        pred_trigs, pred_trig_event, gold_trigs, gold_trig_event = [], [], [], []
        count = 0
        for batch in DataLoader(data_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
            batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_inputs = [[] for _ in range(config.eval_batch_size)]
            batch_event_templates = []
            for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
                batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
            
            ## Stage1: Extract Trigger and Event_type
            # convert EE instances to EAE instances
            trig_inputs, trig_gold_outputs, trig_events, trig_bids = [], [], [], []
            eae_inputs, eae_gold_outputs, eae_events, eae_bids = [], [], [], []
            # create data inputs and output for trigger extraction
            for i, event_temp in enumerate(batch_event_templates):
                for data in event_temp.get_training_data():
                    # eae_inputs.append(data[0].split('<|triggerword|>')[0]+'<|triggerword|> [None] <|template'+data[0].split('<|triggerword|>')[1].split('<|template')[1])
                    # eae_inputs.append(data[0].split(' <|template')[0])
                    trig_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0] + " [" +data[4].replace(":", "_")+ "]")
                    # trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0])
                    trig_events.append(data[2])
                    trig_bids.append(i)
                    # batch_inputs[i].append(data[0].split(' <|template')[0])
                    batch_inputs[i].append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0]) # + " [" +data[4].replace(":", "_")+ "]")
            
            # if there is triggers in this batch, predict triggerword and event type
            if len(trig_inputs) > 0:
                trig_inputs = tokenizer(trig_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                enc_idxs = trig_inputs['input_ids']
                enc_idxs = enc_idxs.cuda()
                enc_attn = trig_inputs['attention_mask'].cuda()

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
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                #TODO: beam with multiple output
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                            num_beams=config.beam_size, max_length=config.max_output_length,
                            forced_bos_token_id=None, num_return_sequences=args.num_return, num_beam_groups=5, diversity_penalty=1.0) # diverse beam search
                trig_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                trig_pred_outputs = np.reshape(trig_pred_outputs, (len(trig_gold_outputs), -1))
                # extract triggerword and event type from the generated outputs
                for p_texts, g_text in zip(trig_pred_outputs, trig_gold_outputs):
                    tag_ge = re.search('\[[^ />][^>]*\]', g_text)
                    gold_event_type = tag_ge.group()[1:-1]
                    gold_trigs.append(g_text[16:tag_ge.start()-1])
                    gold_trig_event.append(g_text[16:])
                    flag = False
                    # loop to check if the ground truth exists in the returned four beams
                    for p_text in p_texts:
                        if not p_text.startswith("<|triggerword|>"):
                            continue
                        tag_pe = re.search('\[[^ />][^>]*\]', p_text)
                        if not tag_pe:
                            continue
                        pred_event_type = tag_pe.group()[1:-1]
                        if p_text[16:] == g_text[16:]:
                            flag = True
                            pred_trigs.append(p_text[16:tag_pe.start()-1])
                            pred_trig_event.append(p_text[16:])
                            break
                    if not flag:
                        tag_pe = re.search('\[[^ />][^>]*\]', p_texts[0])
                        pred_event_type = tag_pe.group()[1:-1]
                        pred_trigs.append(p_texts[0][16:tag_pe.start()-1])
                        pred_trig_event.append(p_texts[0][16:])
                    

            # create data inputs and output for argument extraction
            for i, event_temp in enumerate(batch_event_templates):
                for data in event_temp.get_training_data():
                    # eae_inputs.append(data[0].split('<|triggerword|>')[0]+'<|triggerword|> [None] <|template'+data[0].split('<|triggerword|>')[1].split('<|template')[1])
                    # eae_inputs.append(data[0].split(' <|template')[0])
                    # eae_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0])
                    # eae_inputs.append("EventExtract: " + data[0].split(' <|template')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    eae_inputs.append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "] <|template" + data[0].split(" <|template")[1])
                    eae_gold_outputs.append(data[1])
                    eae_events.append(data[2])
                    eae_bids.append(i)
                    # batch_inputs[i].append(data[0].split(' <|template')[0])
                    # batch_inputs[i].append("EventExtract: " + data[0].split(' <|template')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    batch_inputs[i].append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "] <|template" + data[0].split(" <|template")[1])

            # if there are triggers in this batch, predict argument roles
            if len(eae_inputs) > 0:
                eae_inputs = tokenizer(eae_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
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
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                            num_beams=config.beam_size, max_length=config.max_output_length,
                            forced_bos_token_id=None)

                eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                # extract argument roles from the generated outputs
                for p_text, g_text, info, bid in zip(eae_pred_outputs, eae_gold_outputs, eae_events, eae_bids):
                    if config.model_name.split("copy+")[-1] == 't5-base':
                        p_text = p_text.replace(" |", " <|")
                        if p_text and p_text[0] != "<":
                            p_text = "<" + p_text
                    theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'), False)
                    assert theclass
                    template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'], config.lang, info)            
                    pred_object = template.decode(p_text)

                    for span, role_type, _ in pred_object:
                        # convert the predicted span to the offsets in the passage
                        # Chinese uses a different function since there is no space between Chenise characters
                        if config.lang == "chinese":
                            sid, eid = get_span_idxs_zh(batch.tokens[bid], span, trigger_span=info['trigger span'])
                        else:
                            sid, eid = get_span_idxs(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer, trigger_span=info['trigger span'])

                        if sid == -1:
                            continue
                        batch_pred_roles[bid].append(((info['trigger span']+(info['event type'],)), (sid, eid, role_type)))

                    batch_gold_outputs[bid].append(g_text)
                    batch_pred_outputs[bid].append(p_text)

            batch_pred_roles = [sorted(set(role)) for role in batch_pred_roles]
            
            gold_triggers.extend(batch.triggers)
            gold_roles.extend(batch.roles)
            pred_roles.extend(batch_pred_roles)
            pred_wnd_ids.extend(batch.wnd_ids)
            gold_outputs.extend(batch_gold_outputs)
            pred_outputs.extend(batch_pred_outputs)
            inputs.extend(batch_inputs)

        progress.close()

        # calculate scores
        scores = cal_scores(gold_roles, pred_roles)

        print("---------------------------------------------------------------------")
        print('Trigger I {:6.2f}, Trigger C {:6.2f}'.format(f1_score(pred_trigs, gold_trigs, average='micro')*100, f1_score(pred_trig_event, gold_trig_event, average='micro')*100))
        print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            scores['arg_id'][3] * 100.0, scores['arg_id'][2], scores['arg_id'][1], 
            scores['arg_id'][4] * 100.0, scores['arg_id'][2], scores['arg_id'][0], scores['arg_id'][5] * 100.0))
        print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            scores['arg_cls'][3] * 100.0, scores['arg_cls'][2], scores['arg_cls'][1], 
            scores['arg_cls'][4] * 100.0, scores['arg_cls'][2], scores['arg_cls'][0], scores['arg_cls'][5] * 100.0))
        print("---------------------------------------------------------------------")


        # write outputs
        outputs = {}
        for (pred_wnd_id, gold_trigger, gold_role, pred_role, gold_output, pred_output, input) in zip(
            pred_wnd_ids, gold_triggers, gold_roles, pred_roles, gold_outputs, pred_outputs, inputs):
            outputs[pred_wnd_id] = {
                "input": input, 
                "triggers": gold_trigger,
                "gold_roles": gold_role,
                "pred_roles": pred_role,
                "gold_text": gold_output,
                "pred_text": pred_output,
            }

        with open(os.path.join(args.output_dir, f'{data_type}.pred.json'), 'w') as fp:
            json.dump(outputs, fp, indent=2)
elif args.type == 'e2e':
    # End-to-end inference
    for data_set, batch_num, data_type in zip([dev_set, test_set], [dev_batch_num, test_batch_num], ['dev', 'test']):
    # for data_set, batch_num, data_type in zip([test_set], [test_batch_num], ['test']):
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc=data_type)
        gold_triggers, gold_roles, pred_roles = [], [], []
        pred_wnd_ids, gold_outputs, pred_outputs, inputs = [], [], [], []
        pred_trigs, pred_trig_event, gold_trigs, gold_trig_event = [], [], [], []
        count = 0
        # evaluate batch
        for batch in DataLoader(data_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
            batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_inputs = [[] for _ in range(config.eval_batch_size)]
            batch_event_templates = []
            for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
                batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
            
            ## Stage1: Extract Trigger and Event_type
            # convert EE instances to EAE instances
            trig_inputs, trig_gold_outputs, trig_events, trig_bids = [], [], [], []
            eae_inputs, eae_gold_outputs, eae_events, eae_bids = [], [], [], []
            # create data inputs and output for trigger extraction
            for i, event_temp in enumerate(batch_event_templates):
                # if len(event_temp.get_training_data()) != 1:
                #     continue
                for data in event_temp.get_training_data():
                    # eae_inputs.append(data[0].split('<|triggerword|>')[0]+'<|triggerword|> [None] <|template'+data[0].split('<|triggerword|>')[1].split('<|template')[1])
                    # eae_inputs.append(data[0].split(' <|template')[0])
                    # trig_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]")
                    # for trig_arg_notemp_notype
                    trig_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0])
                    trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0] + " [" +data[4].replace(":", "_")+ "]")
                    # trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0])
                    trig_events.append(data[2])
                    trig_bids.append(i)
                    batch_inputs[i].append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0])
                    # batch_inputs[i].append(data[0].split(' <|template')[0])
                    # batch_inputs[i].append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]")
            # if there is triggers in this batch, predict triggerword and event type
            if len(trig_inputs) > 0:
                trig_inputs = tokenizer(trig_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
                enc_idxs = trig_inputs['input_ids']
                enc_idxs = enc_idxs.cuda()
                enc_attn = trig_inputs['attention_mask'].cuda()

                if config.beam_size == 1:
                    model.model._cache_input_ids = enc_idxs
                else:
                    expanded_return_idx = (
                        torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
                    )
                    input_ids = enc_idxs.index_select(0, expanded_return_idx)
                    model.model._cache_input_ids = input_ids
                
                # inference, generate outputs
                with torch.no_grad():
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                #TODO: beam with multiple output
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        # outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                        #     num_beams=config.beam_size, max_length=config.max_output_length,
                        #     forced_bos_token_id=None, num_return_sequences=args.num_return, num_beam_groups=config.beam_size, diversity_penalty=1.0) # diverse beam search
                        outputs = model.model.generate(
                            input_ids=enc_idxs, 
                            attention_mask=enc_attn,
                            num_beams=config.beam_size, 
                            max_length=config.max_output_length,
                            forced_bos_token_id=None, 
                            num_return_sequences=args.num_return)
                trig_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                trig_pred_outputs = np.reshape(trig_pred_outputs, (len(trig_gold_outputs), -1))
                # extract triggerword and event type from the generated outputs
                for p_texts, g_text in zip(trig_pred_outputs, trig_gold_outputs):
                    tag_ge = re.search('\[[^ />][^>]*\]', g_text)
                    gold_event_type = tag_ge.group()[1:-1]
                    gold_trigs.append(g_text[16:tag_ge.start()-1])
                    gold_trig_event.append(g_text[16:])
                    flag = False
                    ## loop to check if the ground truth exists in the returned four beams
                    # for p_text in p_texts:
                    #     if not p_text.startswith("<|triggerword|>"):
                    #         continue
                    #     tag_pe = re.search('\[[^ />][^>]*\]', p_text)
                    #     if not tag_pe:
                    #         continue
                    #     pred_event_type = tag_pe.group()[1:-1]
                    #     if p_text[16:] == g_text[16:]:
                    #         flag = True
                    #         pred_trigs.append(p_text[16:tag_pe.start()-1])
                    #         pred_trig_event.append(p_text[16:])
                    #         break
                    if not flag:
                        tag_pe = re.search('\[[^ />][^>]*\]', p_texts[0])
                        pred_event_type = tag_pe.group()[1:-1]
                        pred_trigs.append(p_texts[0][16:tag_pe.start()-1])
                        pred_trig_event.append(p_texts[0][16:])
                    

            # create data inputs and output for argument extraction
            # use the pred_trigs generated in the first stage to construct the EE input
            for i, event_temp in enumerate(batch_event_templates):
                # if len(event_temp.get_training_data()) != 1:
                #     continue
                for data in event_temp.get_training_data():
                    # eae_inputs.append(data[0].split('<|triggerword|>')[0]+'<|triggerword|> [None] <|template'+data[0].split('<|triggerword|>')[1].split('<|template')[1])
                    # eae_inputs.append(data[0].split(' <|template')[0])
                    
                    # for trig_arg_notemp_notype
                    eae_inputs.append("EventExtract: " + data[0].split(' <|triggerword|>')[0] + ' <|triggerword|> ' + pred_trigs[count])
                    batch_inputs[i].append("EventExtract: " + data[0].split(' <|triggerword|>')[0]+ ' <|triggerword|> ' + pred_trigs[count])

                    # for trig_arg_notemp
                    # eae_inputs.append("EventExtract: " + data[0].split(' <|triggerword|>')[0] + ' <|triggerword|> ' + pred_trig_event[count])
                    # batch_inputs[i].append("EventExtract: " + data[0].split(' <|triggerword|>')[0]+ ' <|triggerword|> ' + pred_trig_event[count])
                    
                    # eae_inputs.append("EventExtract: " + data[0].split(' <|template')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    # eae_inputs.append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "] <|template" + data[0].split(" <|template")[1])
                    # eae_inputs.append("EventExtract: " + data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]" + ' <|triggerword|> ' + pred_trig_event[count])
                    eae_gold_outputs.append(data[1])
                    eae_events.append(data[2])
                    eae_bids.append(i)

                    # batch_inputs[i].append(data[0].split(' <|template')[0])
                    # batch_inputs[i].append("EventExtract: " + data[0].split(' <|template')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    # batch_inputs[i].append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "] <|template" + data[0].split(" <|template")[1])
                    # batch_inputs[i].append("EventExtract: " + data[0].split(" <|template")[0] + " [" +data[4].replace(":", "_")+ "]" + pred_trig_event[count]) # + " <|template" + data[0].split(" <|template")[1])
                    # batch_inputs[i].append("EventExtract: " + data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]" + ' <|triggerword|> ' + pred_trig_event[count])
                    count += 1
            # if there are triggers in this batch, predict argument roles
            if len(eae_inputs) > 0:
                eae_inputs = tokenizer(eae_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
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
                    if args.constrained_decode:
                        prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                                num_beams=config.beam_size, 
                                max_length=config.max_output_length,
                                forced_bos_token_id=None,
                                prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                                )
                    else:
                        # outputs= model.model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
                        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                            num_beams=config.beam_size, max_length=config.max_output_length,num_return_sequences=1, 
                            forced_bos_token_id=None)
                        # outputs = model.model.generate(
                        #     input_ids=enc_idxs, 
                        #     attention_mask=enc_attn,
                        #     num_beams=config.beam_size, 
                        #     max_length=config.max_output_length,
                        #     forced_bos_token_id=None, 
                        #     num_return_sequences=args.num_return, 
                        #     num_beam_groups=config.beam_size, 
                        #     diversity_penalty=1.0)

                eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                # extract argument roles from the generated outputs
                for p_text, g_text, info, bid in zip(eae_pred_outputs, eae_gold_outputs, eae_events, eae_bids):
                    if config.model_name.split("copy+")[-1] == 't5-base':
                        p_text = p_text.replace(" |", " <|")
                        if p_text and p_text[0] != "<":
                            p_text = "<" + p_text
                    theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'), False)
                    assert theclass
                    template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'], config.lang, info)            
                    pred_object = template.decode(p_text)

                    for span, role_type, _ in pred_object:
                        # convert the predicted span to the offsets in the passage
                        # Chinese uses a different function since there is no space between Chenise characters
                        if config.lang == "chinese":
                            sid, eid = get_span_idxs_zh(batch.tokens[bid], span, trigger_span=info['trigger span'])
                        else:
                            sid, eid = get_span_idxs(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer, trigger_span=info['trigger span'])

                        if sid == -1:
                            continue
                        batch_pred_roles[bid].append(((info['trigger span']+(info['event type'],)), (sid, eid, role_type)))

                    batch_gold_outputs[bid].append(g_text)
                    batch_pred_outputs[bid].append(p_text)

            batch_pred_roles = [sorted(set(role)) for role in batch_pred_roles]
            
            gold_triggers.extend(batch.triggers)
            gold_roles.extend(batch.roles)
            pred_roles.extend(batch_pred_roles)
            pred_wnd_ids.extend(batch.wnd_ids)
            gold_outputs.extend(batch_gold_outputs)
            pred_outputs.extend(batch_pred_outputs)
            inputs.extend(batch_inputs)

        progress.close()

        # calculate scores
        scores = cal_scores(gold_roles, pred_roles)
        print(f"num pred_trig {len(pred_trigs)}, num gold_trig {len(gold_trigs)}")
        print(f"num pred_trig_event {len(pred_trig_event)}, num gold_trig_event {len(gold_trig_event)}")
        print("first five predictions: ", pred_trig_event[:5], "first five gold: ", gold_trig_event[:5])

        print("---------------------------------------------------------------------")
        c_temp = 0
        for p, g in zip(pred_trigs, gold_trigs):
            if p == g:
                c_temp += 1
        print("correct trigger I : ", c_temp)

        c_temp = 0
        for p, g in zip(pred_trig_event, gold_trig_event):
            if p == g:
                c_temp += 1
        print("correct trigger C : ", c_temp)

        print('Trigger I {:6.2f}, Trigger C {:6.2f}'.format(f1_score(pred_trigs, gold_trigs, average='micro')*100, f1_score(pred_trig_event, gold_trig_event, average='micro')*100))
        print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            scores['arg_id'][3] * 100.0, scores['arg_id'][2], scores['arg_id'][1], 
            scores['arg_id'][4] * 100.0, scores['arg_id'][2], scores['arg_id'][0], scores['arg_id'][5] * 100.0))
        print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            scores['arg_cls'][3] * 100.0, scores['arg_cls'][2], scores['arg_cls'][1], 
            scores['arg_cls'][4] * 100.0, scores['arg_cls'][2], scores['arg_cls'][0], scores['arg_cls'][5] * 100.0))
        print("---------------------------------------------------------------------")


        # write outputs
        outputs = {}
        for (pred_wnd_id, gold_trigger, gold_role, pred_role, gold_output, pred_output, input) in zip(
            pred_wnd_ids, gold_triggers, gold_roles, pred_roles, gold_outputs, pred_outputs, inputs):
            outputs[pred_wnd_id] = {
                "input": input, 
                "triggers": gold_trigger,
                "gold_roles": gold_role,
                "pred_roles": pred_role,
                "gold_text": gold_output,
                "pred_text": pred_output,
            }
        with open(os.path.join(args.output_dir, f'{data_type}.pred.json'), 'w') as fp:
            json.dump(outputs, fp, indent=2)
else:
    print("############ wrong evaluation type #############")
