import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, AdamW, get_linear_schedule_with_warmup, T5Tokenizer, BartTokenizer

# from transformers import LogitsProcessorList, MinLengthLogitsProcessor, ConstrainedBeamSearchScorer
from model import GenerativeModel, Prefix_fn_cls
from data import EEDataset
from utils import cal_scores, get_span_idxs, get_span_idxs_zh
from argparse import ArgumentParser, Namespace
import re
from sklearn.metrics import f1_score
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
    
# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('--constrained_decode', default=False, action='store_true')
parser.add_argument('--beam', type=int, default=4)
parser.add_argument('--beam_group', type=int, default=4)
parser.add_argument('--num_return', type=int, default=1)
parser.add_argument('--type', type=str, default="sep")
parser.add_argument('--single_only', default=False, action="store_true")
parser.add_argument('--trig_style', type=str, default="content-et2trig")
parser.add_argument('--arg_style', type=str, default="content-trig2arg")
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

assert torch.cuda.is_available()


# logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
# logger = logging.getLogger(__name__)
        
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
elif config.model_name.startswith("facebook/bart-") or config.model_name.startswith("bart-"):
    tokenizer = BartTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
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

train_set = EEDataset(tokenizer, config.train_file, max_length=config.max_length)
train_batch_num = len(train_set) // config.eval_batch_size + (len(train_set) % config.eval_batch_size != 0)

with open(config.vocab_file) as f:
    vocab = json.load(f)

# load model
# logger.info(f"Loading model from {args.model}")
model = GenerativeModel(config, tokenizer)
model.load_state_dict(torch.load(args.model, map_location=f'cuda:{config.gpu_device}'))
model.cuda(device=config.gpu_device)
model.eval()

# output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


if args.type == "ranktrig":
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    
    # compute bleu
    def compute_bleu(can, ref):
        smoothie = SmoothingFunction().method2
        bleu_s = sentence_bleu([ref.split()], can.split(), weights=(1,0,0,0), smoothing_function=smoothie)
        return bleu_s
    
    # compute rouge score
    def compute_rouge(hyp, ref):
        score = all_scorer.score(ref, hyp)
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3

    all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.Event(enable_timing=True)
    timings = []
    for data_set, batch_num, data_type in zip([train_set, dev_set, test_set], [train_batch_num, dev_batch_num, test_batch_num], ['train', 'dev', 'test']):
        output_dir = args.output_dir + data_type
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc=data_type)
        gold_triggers, gold_roles, pred_roles = [], [], []
        pred_wnd_ids, gold_outputs, pred_outputs, inputs = [], [], [], []
        pred_trigs, pred_trig_event, gold_trigs, gold_trig_event = [], [], [], []
        count = 0
        ref_event, candidates, trig_candidates, content = [], [], [], []
        # only considering single event examples
        # content: with event_type and without event_type
        # candidates: <|triggerword|> triggerword <role> arg <role> arg # combine the prediction from TE and EE
        # ref_event: ground truth 

        # evaluate batch
        for batch in DataLoader(data_set, batch_size=8, shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            # if count > 20: break
            batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
            batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
            batch_inputs = [[] for _ in range(config.eval_batch_size)]
            batch_event_templates = []
            for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
                batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
            
            ## Stage1: Extract Trigger and Event_type
            # convert EE instances to EAE instances
            trig_inputs, trig_gold_outputs= [], []
            eae_inputs, eae_gold_outputs = [], []
            # create data inputs and output for trigger extraction
            for i, event_temp in enumerate(batch_event_templates):
                if args.single_only:
                    if len(event_temp.get_training_data()) > 1:
                        # only considering the single event examples
                        continue
                for data in event_temp.get_training_data():
                    trig_inputs.append("TriggerExtract: " + data[0].split(' <|triggerword|>')[0]) # + " [" +data[4].replace(":", "_")+ "]")
                    trig_gold_outputs.append('<|triggerword|> ' + data[0].split('<|triggerword|> ')[1].split(" <|template|")[0] + " [" +data[4].replace(":", "_")+ "]")

                    ## generate data for ranking
                    content.append(data[0].split(' <|triggerword|>')[0]) # without event_type
                    # content.append(data[0].split(' <|triggerword|>')[0] + " [" +data[4].replace(":", "_")+ "]") # with event_type
                    count += 1
            
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
                        ## TODO: with copy mechanism? constrain on context words only?

                        # beam search
                        starter.record()
                        outputs = model.model.generate(
                            input_ids=enc_idxs, 
                            attention_mask=enc_attn,
                            num_beams=config.beam_size, 
                            max_length=config.max_output_length,
                            forced_bos_token_id=None, 
                            num_return_sequences=args.num_return,
                            output_scores=True,
                            min_length=0,
                            return_dict_in_generate=True)
                        ender.record()
                        torch.cuda.synchronize()
                        if data_type == "test":
                            timings.append(starter.elapsed_time(ender))

                seqs = outputs.sequences.detach().cpu()
                trig_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in seqs]
                
                pred_scores = outputs.sequences_scores.detach().cpu().numpy()
                trig_pred_outputs = np.reshape(trig_pred_outputs, (len(trig_gold_outputs), -1)).tolist()
                pred_scores = np.reshape(pred_scores, (len(trig_gold_outputs), -1)).tolist()

                trs_bs = torch.sum(model.model.compute_transition_beam_scores(
                    sequences=outputs.sequences,
                    scores=outputs.scores, 
                    beam_indices=outputs.beam_indices
                ), dim=1).cpu().numpy()
                trs_bs = np.reshape(trs_bs, (len(trig_gold_outputs), -1)).tolist()
                
                # extract triggerword and event type from the generated outputs
                for p_texts, g_text, p_scores in zip(trig_pred_outputs, trig_gold_outputs, trs_bs):
                    gold_trig_event.append(g_text[16:])
                    ref_event.append(g_text)
                    p_texts = [(x, str(score)) for x, score in zip(p_texts, p_scores)]
                    candidates.append(p_texts)
            assert np.shape(ref_event)[0] == np.shape(candidates)[0] == np.shape(content)[0]
            
        count = 0
        for cn_i, r_i, can_i in zip(content, ref_event, candidates):
            output = {
                "article": cn_i, 
                "abstract": r_i,
                "candidates": can_i,
                }
            # write outputs
            with open(os.path.join(output_dir, f'{count}.json'), 'w') as fp:
                json.dump(output, fp, indent=2)
                
            count += 1
        print("num sample: ", count)
    
    mean_syn = np.sum(timings) / len(timings)
    std_syn = np.std(timings)
    print('------------------------------------')
    print(f"num instances: {len(timings)}, mean_syn: {mean_syn}, std_syn: {std_syn}")
else:
    print("############ wrong evaluation type #############")

