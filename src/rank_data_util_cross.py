import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
import os
from transformers import BertTokenizerFast, RobertaTokenizerFast

# on 01
SEP, EOS = '<sep>', '<eos>'
class Data:
    def __init__(self, model_name, train_path, test_path, train_candi_pool_size, train_negative_num, test_candi_span, 
        max_content_len, max_tgt_len):
        '''
            train_candi_pool_size: number of possible negative candidates for each content
            train_negative_num: number of negatives during training
            test_candi_span: number of candidates considered during testing
        '''
        print ('Loading tokenizer...')
        if model_name.startswith('bert'):
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        elif model_name.startswith('roberta'):
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        else:
            raise Exception('Wrong Tokenization Mode!')

        print ('original vocabulary Size %d' % len(self.tokenizer))
        self.special_token_list = [SEP, EOS]
        print ('original vocabulary Size %d' % len(self.tokenizer))
        self.tokenizer.add_tokens(self.special_token_list)
        print ('vocabulary size after extension is %d' % len(self.tokenizer))

        self.cls_token, self.cls_token_id, self.sep_token, self.sep_token_id = self.tokenizer.cls_token, \
        self.tokenizer.cls_token_id, self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.max_content_len, self.max_tgt_len = max_content_len, max_tgt_len
        self.train_candi_pool_size, self.train_negative_num, self.test_candi_span = \
        train_candi_pool_size, train_negative_num, test_candi_span

        print ('Loading training data...')
        self.train_content_id_list, self.train_reference_id_list, self.train_all_candidate_id_list, self.train_all_candidate_genscore_list, \
        self.train_content_text_list, self.train_reference_text_list, self.train_all_candidate_text_list = \
        self.load_data(train_path, mode='train')

        # print ('Loading test data...')
        self.test_content_id_list, self.test_reference_id_list, self.test_all_candidate_id_list, self.test_all_candidate_genscore_list, \
        self.test_content_text_list, self.test_reference_text_list, self.test_all_candidate_text_list = \
        self.load_data(test_path, mode='test')

        self.train_num, self.test_num = len(self.train_content_id_list), len(self.test_content_id_list)
        print ('train number is %d, test number is %d' % (self.train_num, self.test_num))
        self.train_idx_list = [i for i in range(self.train_num)]
        self.test_idx_list, self.test_current_idx = [i for i in range(self.test_num)], 0
        
    def load_one_text_id(self, text, max_len):
        # removed the constrain of max_len
        text_id_list = self.tokenizer.encode(text, max_length=512, truncation=True, add_special_tokens=False)[:max_len]
        return text_id_list


    def load_all(self, fdir, mode):
        # load from the file directory
        self.isdir = os.path.isdir(fdir)
        print("isdir", self.isdir)

        if self.isdir:
            self.fdir = fdir
            self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            self.num = len(self.files)

        data = []
        for idx in range(self.num):
            if self.isdir:
                with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
                    data.append(json.load(f))
            else:
                with open(self.files[idx]) as f:
                    data.append(json.load(f))
        
        if "t5" in fdir:
            for idx, di in enumerate(data):
                for candi in range(len(di['candidates'])):
                    di['candidates'][candi][0] = "<" + di['candidates'][candi][0]
        
        # in evaluation, split the reference by "\n "
        if mode == "test":
            appeared = {}
            new_data = {}
            for idx, di in enumerate(data):
                if di['article'] in appeared.keys():
                    new_data[appeared[di['article']]]['abstract'] += "\n " + di['abstract']
                else:
                    appeared[di['article']] = idx
                    new_data[idx] = di
            self.num = len(new_data)
            data = list(new_data.values())
        return data


    def load_content(self, data):
        res_id_list = []
        res_text_list = []

        p = progressbar.ProgressBar(self.num)
        p.start()
        idx = 0
        for l in data:
            p.update(idx)
            one_text = l["article"]
            one_id_list = self.load_one_text_id(one_text, self.max_content_len)
            
            res_id_list.append(one_id_list)
            res_text_list.append(one_text)
        p.finish()
        return res_id_list, res_text_list


    def load_reference(self, data, mode):
        res_id_list = []
        res_text_list = []
        p = progressbar.ProgressBar(self.num)
        p.start()
        idx = 0
        for l in data:
            p.update(idx)
            one_text = l["abstract"]
            one_id_list = self.load_one_text_id(one_text, self.max_tgt_len)
            res_id_list.append(one_id_list)
            res_text_list.append(one_text)
        p.finish()
        return res_id_list, res_text_list


    def load_candidates(self, candidate_text, mode, ref_text):
        if mode == 'train':
            select_num = self.train_candi_pool_size
        elif mode == 'test':
            select_num = self.test_candi_span
        else:
            raise Exception('Wrong Mode!!!')
        candidate_id_list = []
        if mode == 'train':
            candidate_text_list = [i[0] for i in candidate_text if i[0] != ref_text and i[0] != ""]
            candidates_gen_scores = [float(i[1]) for i in candidate_text if i[0] != ref_text and i[0] != ""]
            if candidate_text_list == []:
                return None, None, None
        else:
            candidate_text_list = [i[0] for i in candidate_text if i[0] != ""]
            candidates_gen_scores = [float(i[1]) for i in candidate_text if i[0] != ""]
        # candidate_text_list[:select_num]
        # print("num_candidates: ", len(candidate_text_list))
        # print("ref: ", ref_text, "first 3 cand: ", candidate_text_list[:3])
        for text in candidate_text_list:
            one_text_id = self.load_one_text_id(text, self.max_tgt_len)
            if len(one_text_id) <= 0:
                print("text", text)
                print("id", one_text_id)
                print("candidate text", candidate_text)
                print(candidate_text_list)

                assert len(one_text_id) > 0
            candidate_id_list.append(one_text_id)
        # assert len(candidate_id_list) == select_num
        return candidate_id_list, candidate_text_list, candidates_gen_scores


    def load_data(self, data_path, mode):
        # TODO: When loading the test data, filter the multi-event instances
        # for multi-event instance, there will be multiple positives, only take into input once
        # each reference trigger will be separated by "\n"  
        # candidates leave unchange

        data = self.load_all(data_path, mode)
        content_id_list, content_text_list = self.load_content(data)
        reference_id_list, reference_text_list = self.load_reference(data, mode)
        tmp_all_candidate_text_list = [l["candidates"] for l in data]

        assert len(tmp_all_candidate_text_list) == len(content_id_list)

        all_candidate_id_list, all_candidate_text_list, all_candidate_genscore_list = [], [], []
        print ('Loading candidates...')
        p = progressbar.ProgressBar(self.num)
        p.start()
        to_remove = []
        for idx in range(self.num):
            p.update(idx)
            ref_text = reference_text_list[idx] # get the reference text
            one_candidate_text = tmp_all_candidate_text_list[idx]
            one_candidate_text_id_list, one_candidate_text_list, one_candidate_gen_scores = self.load_candidates(one_candidate_text, mode, ref_text)
            if one_candidate_text_id_list == None:
                to_remove.append(idx)
                continue
            all_candidate_id_list.append(one_candidate_text_id_list)
            all_candidate_text_list.append(one_candidate_text_list)
            all_candidate_genscore_list.append(one_candidate_gen_scores)
        p.finish()
        for idx in to_remove[::-1]:
            reference_text_list.pop(idx)
            del reference_id_list[idx]
            del content_id_list[idx]
        return content_id_list, reference_id_list, all_candidate_id_list, all_candidate_genscore_list, content_text_list, reference_text_list, all_candidate_text_list


    def padding(self, batch_id_list):
        batch_tensor_list = [torch.LongTensor(one_id_list) for one_id_list in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_tensor_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask


    def padding_segment(self, batch_seg_id_list):
        res_seg_id_list = []
        max_len = max([len(item) for item in batch_seg_id_list])
        for one_seg_id_list in batch_seg_id_list:
            one_len_diff = max_len - len(one_seg_id_list)
            res_seg_id_list.append(one_seg_id_list + [1 for _ in range(one_len_diff)])
        seg_id_tensor = torch.LongTensor(res_seg_id_list)
        assert seg_id_tensor.size() == torch.Size([len(batch_seg_id_list), max_len])
        return seg_id_tensor


    def process_batch_data(self, batch_content_id_list, batch_reference_id_list):
        batch_token_id_list, batch_seg_id_list = [], []
        assert len(batch_content_id_list) == len(batch_reference_id_list)
        bsz = len(batch_content_id_list)
        for idx in range(bsz):
            one_content_id_list = [self.cls_token_id] + batch_content_id_list[idx] + [self.sep_token_id]
            one_content_seg_id_list = [0 for _ in one_content_id_list]
            one_reference_id_list = batch_reference_id_list[idx] + [self.sep_token_id]
            one_reference_seg_id_list = [1 for _ in one_reference_id_list]
            one_token_id_list = one_content_id_list + one_reference_id_list
            batch_token_id_list.append(one_token_id_list)
            one_seg_id_list = one_content_seg_id_list + one_reference_seg_id_list
            batch_seg_id_list.append(one_seg_id_list)
        batch_token_tensor, batch_token_mask = self.padding(batch_token_id_list)
        batch_seg_tensor = self.padding_segment(batch_seg_id_list)
        assert batch_token_tensor.size() == batch_token_mask.size()
        assert batch_seg_tensor.size() == batch_token_mask.size()
        return batch_token_tensor, batch_token_mask, batch_seg_tensor


    def get_one_train_negative_id_list(self, content_id):
        candi_id_list = self.train_all_candidate_id_list[content_id]
        candi_num = len(candi_id_list)
        if candi_num >= self.train_negative_num:
            neg_idx_list = random.sample([i for i in range(candi_num)], self.train_negative_num)
        else:
            neg_idx_list = random.choices([i for i in range(candi_num)], k=self.train_negative_num)
        neg_candi_id_list = [candi_id_list[idx] for idx in neg_idx_list]
        return neg_candi_id_list


    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_content_id_list, batch_reference_id_list, all_batch_neg_candi_list \
        = [], [], [[] for _ in range(self.train_negative_num)]
        for idx in batch_idx_list:
            one_content_id_list = self.train_content_id_list[idx]
            batch_content_id_list.append(one_content_id_list)
            one_reference_id_list = self.train_reference_id_list[idx]
            batch_reference_id_list.append(one_reference_id_list)
            one_candi_neg_id_list = self.get_one_train_negative_id_list(idx)
            # TODO: what is this negative set?, all the k-th candidate within this batch? 
            # shape of the batch_neg_candi_list will be (self.train_negative_num, batch_size)
            for k in range(self.train_negative_num):
                all_batch_neg_candi_list[k].append(one_candi_neg_id_list[k])
        all_batch_token_list, all_batch_mask_list, all_batch_seg_list = [], [], []
        # first process reference data
        batch_ref_token_tensor, batch_ref_mask_tensor, batch_ref_seg_tensor = self.process_batch_data(batch_content_id_list, batch_reference_id_list)
        all_batch_token_list.append(batch_ref_token_tensor)
        all_batch_mask_list.append(batch_ref_mask_tensor)
        all_batch_seg_list.append(batch_ref_seg_tensor)
        for neg_k in range(self.train_negative_num):
            batch_neg_token_tensor, batch_neg_mask_tensor, batch_neg_seg_tensor = \
            self.process_batch_data(batch_content_id_list, all_batch_neg_candi_list[neg_k])
            all_batch_token_list.append(batch_neg_token_tensor)
            all_batch_mask_list.append(batch_neg_mask_tensor)
            all_batch_seg_list.append(batch_neg_seg_tensor)
        return all_batch_token_list, all_batch_mask_list, all_batch_seg_list


    def get_next_test_batch(self, batch_size):
        batch_content_id_list, all_candi_id_list = [], [[] for _ in range(self.test_candi_span)]
        batch_reference_text_list, batch_candidate_text_list, batch_candidate_genscore_list = [], [], []
        if self.test_current_idx + batch_size < self.test_num - 1:
            for i in range(batch_size):
                curr_idx = self.test_current_idx + i
                one_content_id_list = self.test_content_id_list[curr_idx]
                batch_content_id_list.append(one_content_id_list)
                one_candi_id_list = self.test_all_candidate_id_list[curr_idx]
                for k in range(self.test_candi_span):
                    all_candi_id_list[k].append(one_candi_id_list[k])
                batch_reference_text_list.append(self.test_reference_text_list[curr_idx])
                batch_candidate_text_list.append(self.test_all_candidate_text_list[curr_idx][:self.test_candi_span])
                batch_candidate_genscore_list.append(self.test_all_candidate_genscore_list[curr_idx][:self.test_candi_span])
            self.test_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = self.test_current_idx + i
                if curr_idx > self.test_num - 1: 
                    curr_idx = 0
                    self.test_current_idx = 0
                else:
                    pass
                # if curr_idx > self.test_num - 1:
                #     self.test_current_idx = 0
                #     break
                one_content_id_list = self.test_content_id_list[curr_idx]
                batch_content_id_list.append(one_content_id_list)
                one_candi_id_list = self.test_all_candidate_id_list[curr_idx]
                for k in range(self.test_candi_span):
                    all_candi_id_list[k].append(one_candi_id_list[k])
                batch_reference_text_list.append(self.test_reference_text_list[curr_idx])
                batch_candidate_text_list.append(self.test_all_candidate_text_list[curr_idx][:self.test_candi_span])
                batch_candidate_genscore_list.append(self.test_all_candidate_genscore_list[curr_idx][:self.test_candi_span])
            self.test_current_idx = 0

        all_batch_token_list, all_batch_mask_list, all_batch_seg_list = [], [], []
        ## TODO: shouldn't the shape of all_batch_token_list be [batch_size, self.test_candi_span, seq_length]?
        ## The current shape is [self.test_candi_span, batch_size, seq_length]
        for k in range(self.test_candi_span):
            batch_token_tensor, batch_mask_tensor, batch_seg_tensor = \
            self.process_batch_data(batch_content_id_list, all_candi_id_list[k])
            all_batch_token_list.append(batch_token_tensor)
            all_batch_mask_list.append(batch_mask_tensor)
            all_batch_seg_list.append(batch_seg_tensor)
        return all_batch_token_list, all_batch_mask_list, all_batch_seg_list, batch_reference_text_list, batch_candidate_text_list, batch_candidate_genscore_list