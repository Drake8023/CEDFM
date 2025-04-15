import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW 
import matplotlib.pyplot as plt
from datetime import datetime

from models.ce_qca import CEQCA
from models.ce_qa import CEQA
from models.ce import CE
from models.bert import BERT


from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSAGCNData 
from prepare_vocab import VocabHelp 

from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup 


logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO) 
logger.addHandler(logging.StreamHandler(sys.stdout)) 


def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.losses = []
        self.test_losses = [] 
        print("train data ===========",opt.dataset_file['train'])

        tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name) # actually tokenizer.max_seq_len == opt.max_length 
        # bert = BertModel.from_pretrained('bert-base-uncased') 
        bert = BertModel.from_pretrained("./bert_path") # locally after downloading 
        for name, param in bert.named_parameters(): 
            print(name + ": ", param.requires_grad, sep='') 
          
        dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')      # deprel
        pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')

        opt.deprel_size = len(dep_vocab)

        self.model = opt.model_class(bert, opt).to(opt.device)
        
           
        trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer,pos_vocab, dep_vocab, opt=opt) 
        testset = ABSAGCNData(opt.dataset_file['test'], tokenizer,pos_vocab, dep_vocab, opt=opt) 
 
        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index))) 
        self._print_args() 
       
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    def _reset_params(self): 
        for p in self.model.parameters(): 
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)   # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0]**0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model): 
        # Prepare optimizer and schedule (linear warmup and decay) 
        no_decay = ['bias', 'LayerNorm.weight'] 
        diff_part = ["bert.embeddings", "bert.encoder"] 

        logger.info("layered learning rate on") 
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": self.opt.finetune_weight_decay,
                "lr": self.opt.bert_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.opt.bert_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.opt.learning_rate
            },
        ] 
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon) 
        # optimizer = AdamW(optimizer_grouped_parameters) 

        return optimizer 

    
    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = '' 
        self.loss = criterion
        train_losses_per_epoch = []
        if self.opt.scheduler == 'cosine': 
            scheduler = get_cosine_schedule_with_warmup( 
                optimizer, int(self.opt.warmup*len(self.train_dataloader)), self.opt.num_epoch*len(self.train_dataloader)) 
        elif self.opt.scheduler == 'linear': 
            scheduler = get_linear_schedule_with_warmup( 
                optimizer, int(self.opt.warmup*len(self.train_dataloader)), self.opt.num_epoch*len(self.train_dataloader)) 
        elif self.opt.scheduler == 'none': 
            scheduler = None 
        for epoch in range(self.opt.num_epoch): 
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            epoch_losses = []
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1

                self.model.train()
                optimizer.zero_grad()
                
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                # print("Inputs in batch {}: {}".format(i_batch, inputs))
                # for i, input_tensor in enumerate(inputs):
                #     print(f"Shape of input {i} in batch {i_batch}: {input_tensor.shape}")
                outputs = self.model(inputs) 
                targets = sample_batched['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets) 

                loss.backward()
                optimizer.step() 
                if scheduler: 
                    scheduler.step() 
                    
                epoch_losses.append(loss.item())  
                self.losses.append(loss.item())   
                
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total 
                    test_acc, f1 ,test_loss = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, test_acc, f1,test_loss)
                            self.best_model = copy.deepcopy(self.model) 
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}, test_loss: {:.4f}'.format(loss.item(), train_acc, test_acc, f1,test_loss))
  
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info('Epoch {} average loss: {:.4f}'.format(epoch, epoch_loss))
            train_losses_per_epoch.append(epoch_loss)
            
            _, _, epoch_test_loss = self._evaluate()
            self.test_losses.append(epoch_test_loss)  
            
 
    
        return max_test_acc, max_f1, model_path
    
    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        test_loss = 0.0
        targets_all, outputs_all = None, None
        
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
          
                outputs = self.model(inputs)
                
                loss = self.loss(outputs, targets)
                test_loss += loss.item() * len(targets)
                
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
             
        test_loss /= n_test_total
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1 , test_loss

    @torch.no_grad() 
    def _show_cases(self): # For case study 
        self.model.eval() 
        cases_result = open("target_predict.txt", 'w') 
        for sample_batched in self.test_dataloader: 
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            targets = sample_batched['polarity'].to(self.opt.device)
            outputs = self.model(inputs) 
            predict = torch.argmax(outputs, -1) 
            for i in range(targets.size()[0]): 
                cases_result.write(str(targets[i].item()) ) 
                cases_result.write(", ") 
                cases_result.write(str(predict[i].item()) ) 
                cases_result.write("\n") 
        cases_result.close() 

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion) 

        # For case study 
        # self._show_cases() 
        
    
    def run(self):
        label_weights = torch.tensor([1, 1, 1.], device=self.opt.device) 

        if self.opt.balance_loss: 
            if self.opt.dataset == 'restaurant': 
                label_weights = torch.tensor([1/2164, 1/807, 1/637], device=self.opt.device)
            elif self.opt.dataset == 'laptop': 
                label_weights = torch.tensor([1/976, 1/851, 1/455], device=self.opt.device) 
            elif self.opt.dataset == 'twitter': 
                label_weights = torch.tensor([1/1507, 1/1528, 1/3016], device=self.opt.device) 
            elif self.opt.dataset == 'rest16': 
                label_weights = torch.tensor([1/1240, 1/439, 1/69], device=self.opt.device) 
        
        criterion = nn.CrossEntropyLoss(weight=label_weights) 

        optimizer = self.get_bert_optimizer(self.model) 
        max_test_acc_overall = 0
        max_f1_overall = 0
        # if 'bert' not in self.opt.model_name: # use default initilization of every module 
        #     self._reset_params()
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path)) 
        logger.info('#' * 60) 
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall)) 
        logger.info('max_f1_overall:{}'.format(max_f1_overall)) 
        self._test() 


def main():
    model_classes = {
      
        'ce-qca': CEQCA,
        'ce-qa': CEQA,
        'ce': CE,
        'bert': BERT,       
     
    } 

    vocab_dirs = {
        'restaurant': './datasets/Restaurants_corenlp',
        'laptop': './datasets/Laptops_corenlp',
        'twitter': './datasets/Tweets_corenlp',
        'rest16': './datasets/Restaurants16', 
    } 
    
    dataset_files = {
        'restaurant': {
            'train': './datasets/Restaurants_corenlp/train.json',
            'test': './datasets/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './datasets/Laptops_corenlp/train.json',
            'test': './datasets/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './datasets/Tweets_corenlp/train.json',
            'test': './datasets/Tweets_corenlp/test.json',
        }, 
        'rest16': { 
            'train': './datasets/Restaurants16/train.json', 
            'test': './datasets/Restaurants16/test.json', 
        } 
    }
    
    input_colses = { 
   
        'bert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix', 'src_mask', 'aspect_mask'] 
    } 
    
    initializers = { 
        'xavier_uniform_': torch.nn.init.xavier_uniform_, 
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_name', default='ce-qca', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys())) 
    parser.add_argument('--learning_rate', default=0.001, type=float) 
    # 5e-5 0.001
    parser.add_argument('--num_epoch', default=15, type=int) 
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    #60 30
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.') 
    parser.add_argument('--deprel_dim', type=int, default=30, help='Dependent relation embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=60, help='bert dim.') 
    parser.add_argument('--num_layers', type=int, default=4, help='Num of layers.') 
    parser.add_argument('--polarities_dim', default=3, type=int, help='3') 

    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    #0.7 0.5
    parser.add_argument('--lower', default=True, help='Lowercase all words.')

    # no need to specified, may incur error 
    parser.add_argument('--directed', default=False, help='directed graph or undirected graph')
    parser.add_argument('--add_self_loop', default=True) 

    parser.add_argument('--use_rnn', action='store_true') 
    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=60, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.') 
    
    parser.add_argument('--max_length', default=85, type=int) 
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight deay if we apply some.") 
    parser.add_argument('--vocab_dir', type=str, default='./datasets/Restaurants_corenlp') 
    parser.add_argument('--pad_id', default=0, type=int) 

    parser.add_argument('--attn_dropout', type=float, default=0.1) 
    parser.add_argument('--ffn_dropout', type=float, default=0.3) 
    parser.add_argument('--norm', type=str, default='ln', choices=['ln', 'bn']) 
    parser.add_argument('--max_position', type=int, default=9) 

    parser.add_argument('--scheduler', type=str, default='none', choices=['linear', 'cosine', 'none']) 
    parser.add_argument('--warmup', type=float, default=2) 
    parser.add_argument('--balance_loss', action='store_true') 
    parser.add_argument('--cuda', default='0', type=str) 
    
    # BERT 
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # 2.8e-5 1e-8
    parser.add_argument('--bert_dim', type=int, default=768) 
    parser.add_argument('--bert_dropout', type=float, default=0.5, help='BERT dropout rate.')
    # 0.5 0.1
    parser.add_argument('--bert_lr', default=2e-5, type=float) 
    parser.add_argument("--finetune_weight_decay", default=0.01, type=float) 
    opt = parser.parse_args()
    	
    opt.model_class = model_classes[opt.model_name] 
    opt.dataset_file = dataset_files[opt.dataset]
    
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer] 

    opt.vocab_dir = vocab_dirs[opt.dataset] 


    opt.inputs_cols = input_colses['bert']
    opt.hidden_dim = 768 
    opt.max_length = 100 
    opt.num_epoch = 15 
 


    opt.device = torch.device('cuda:'+opt.cuda) 
    
    # set random seed 
    setup_seed(opt.seed) 

    if not os.path.exists('./logging'): 
        os.makedirs('./logging', mode=0o777) 
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H%M%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./logging', log_file)))

    # logger.info(' '.join(sys.argv)) 

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__': 
    main()
