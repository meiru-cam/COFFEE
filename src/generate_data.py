import os, json, pickle, logging, pprint, random
import numpy as np
from data import EEDataset
from argparse import ArgumentParser, Namespace
from utils import generate_vocabs
from transformers import MT5Tokenizer, T5Tokenizer

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)

# import template file
if config.dataset == "ace05":
    from template_generate_ace import event_template_generator
elif config.dataset == "ere":
    from template_generate_ere import event_template_generator
else:
    raise NotImplementedError

# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")

def generate_data(data_set, vocab, config):
    inputs = []
    targets = []
    events = []
    
    for data in data_set.data:
        event_template = event_template_generator(data.tokens, data.triggers, data.roles, 
                                                  config.input_style, config.output_style, 
                                                  vocab, config.lang)
        for data_ in event_template.get_training_data():
            inputs.append("TriggerExtract: " + data_[0].split(' <|triggerword|>')[0]) # input is the content information
            targets.append('<|triggerword|> '+data_[0].split('<|triggerword|> ')[1].split(" <|template|")[0] + " [" +data_[4].replace(":", "_")+ "]") # output for TriggerExtract is triggerword and Event_type

            inputs.append("EventExtract: " + data_[0].split(' <|template')[0]) # input is the content information + trigger, without template
            targets.append(data_[1]) # output is the arguments

            events.append(data_[2])
            events.append(data_[2])
    
    return inputs, targets, events

# check valid styles
assert np.all([style in ["triggerword", "template"] for style in config.input_style])
assert np.all([style in ["argument:roletype"] for style in config.output_style])

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

# create output directory
if not os.path.exists(config.finetune_dir):
    os.makedirs(config.finetune_dir)

# load data
train_set = EEDataset(tokenizer, config.train_file, max_length=config.max_length)
dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
vocab = generate_vocabs([train_set, dev_set, test_set])

# save vocabulary
with open('{}/vocab.json'.format(config.finetune_dir), 'w') as f:
    json.dump(vocab, f, indent=4)    

# generate finetuned data
train_inputs, train_targets, train_events = generate_data(train_set, vocab, config)
logger.info(f"Generated {len(train_inputs)} training examples from {len(train_set)} instance")

with open('{}/train_input.json'.format(config.finetune_dir), 'w') as f:
    json.dump(train_inputs, f, indent=4)

with open('{}/train_target.json'.format(config.finetune_dir), 'w') as f:
    json.dump(train_targets, f, indent=4)

with open('{}/train_all.pkl'.format(config.finetune_dir), 'wb') as f:
    pickle.dump({
        'input': train_inputs,
        'target': train_targets,
        'all': train_events
    }, f)
    
dev_inputs, dev_targets, dev_events = generate_data(dev_set, vocab, config)
logger.info(f"Generated {len(dev_inputs)} dev examples from {len(dev_set)} instance")

with open('{}/dev_input.json'.format(config.finetune_dir), 'w') as f:
    json.dump(dev_inputs, f, indent=4)

with open('{}/dev_target.json'.format(config.finetune_dir), 'w') as f:
    json.dump(dev_targets, f, indent=4)

with open('{}/dev_all.pkl'.format(config.finetune_dir), 'wb') as f:
    pickle.dump({
        'input': dev_inputs,
        'target': dev_targets,
        'all': dev_events
    }, f)
    
test_inputs, test_targets, test_events = generate_data(test_set, vocab, config)
logger.info(f"Generated {len(test_inputs)} test examples from {len(test_set)} instance")

with open('{}/test_input.json'.format(config.finetune_dir), 'w') as f:
    json.dump(test_inputs, f, indent=4)

with open('{}/test_target.json'.format(config.finetune_dir), 'w') as f:
    json.dump(test_targets, f, indent=4)

with open('{}/test_all.pkl'.format(config.finetune_dir), 'wb') as f:
    pickle.dump({
        'input': test_inputs,
        'target': test_targets,
        'all': test_events
    }, f)



