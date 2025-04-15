
import sys
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from .Quantum import PositionEmbedding,ComplexLinear,CDropout,QDropout,my_attention1_1, ComplexMultiply, QOuter,QOuterC, QMixture,QMixtureC, QMeasurement,QMeasurementC,linear_c

class ComplexReLU(torch.nn.Module):
    def forward(self, x):
        real_part = x.real.relu()
        imag_part = x.imag.relu()
        return torch.complex(real_part, imag_part)
class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(ComplexLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.gamma_real = nn.Parameter(torch.ones(normalized_shape))
            self.beta_real = nn.Parameter(torch.zeros(normalized_shape))
            self.gamma_imag = nn.Parameter(torch.ones(normalized_shape))
            self.beta_imag = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.gamma_real = None
            self.beta_real = None
            self.gamma_imag = None
            self.beta_imag = None

    def forward(self, x):
        real_part = x.real
        imag_part = x.imag
        
        mean_real = real_part.mean(dim=-1, keepdim=True)
        var_real = real_part.var(dim=-1, keepdim=True, unbiased=False)
        
        mean_imag = imag_part.mean(dim=-1, keepdim=True)
        var_imag = imag_part.var(dim=-1, keepdim=True, unbiased=False)
        
        real_hat = (real_part - mean_real) / torch.sqrt(var_real + self.eps)
        imag_hat = (imag_part - mean_imag) / torch.sqrt(var_imag + self.eps)
        
        if self.elementwise_affine:
            real_hat = self.gamma_real * real_hat + self.beta_real
            imag_hat = self.gamma_imag * imag_hat + self.beta_imag
        
        return torch.complex(real_hat, imag_hat)

class L2NormC(torch.nn.Module):
    def __init__(self, dim=1, keep_dims=True, eps=1e-10):
        super(L2NormC, self).__init__()
        self.dim = dim
        self.keepdim = keep_dims
        self.eps = eps

    def forward(self, inputs):
        real_part = inputs.real
        imag_part = inputs.imag
        
        norm_squared = torch.sum(real_part**2 + imag_part**2, dim=self.dim, keepdim=self.keepdim)
        output = torch.sqrt(self.eps + norm_squared)
        
        return output
class L2Norm(torch.nn.Module):

    def __init__(self, dim=1, keep_dims=True, eps = 1e-10):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.keepdim = keep_dims
        self.eps = eps

    def forward(self, inputs):

        output = torch.sqrt(self.eps+ torch.sum(inputs**2, dim=self.dim, keepdim=self.keepdim))

        return output


class CE(nn.Module): 

    def __init__(self, bert, opt): 
        super().__init__() 
        self.opt = opt 
        self.bert = bert 
        
        self.seq_len = 100
        self.dim = 100
        self.emb_dim = 100
        self.catdim = self.emb_dim+opt.hidden_dim
        self.liner = nn.Linear(self.seq_len, self.dim)
        self.norm = L2Norm(dim=-1)
        self.projections = nn.Linear(opt.hidden_dim, self.emb_dim)
        self.att_embeddings = PositionEmbedding(self.emb_dim, input_dim=1)
        self.multiply = ComplexMultiply()
        self.mixture = QMixtureC()
        self.outer = QOuterC()
        self.measurement = QMeasurementC(self.emb_dim)
        self.att =my_attention1_1(self.emb_dim,self.emb_dim,1,0.01,16,self.emb_dim)

    
        self.deprel_embedding_layer = nn.Embedding(opt.deprel_size + 1, opt.deprel_dim, padding_idx=0) if opt.deprel_dim > 0 else None    
        self.deprel_embedding_layer_i = nn.Embedding(opt.deprel_size + 1, opt.deprel_dim, padding_idx=0) if opt.deprel_dim > 0 else None    
        
        dim = opt.bert_dim+self.emb_dim+self.emb_dim
        self.linear_in = ComplexLinear(dim, self.emb_dim)
        self.linear_out = nn.Linear(self.catdim, opt.polarities_dim) 
        self.bert_drop = CDropout(opt.bert_dropout) 
        self.pooled_drop = nn.Dropout(opt.bert_dropout) 
        self.ffn_dropout = opt.ffn_dropout 

 

     

    def forward(self, inputs): 
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs 

        outputs = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids) 
        tagger_input, pooled_output = outputs.last_hidden_state, outputs.pooler_output 
        
       
        pooled_output = self.pooled_drop(pooled_output) 

        seq_rep = nn.ReLU()(self.projections(tagger_input))
        mask = self.att_embeddings(attention_mask)
        seq_rep_norm = F.normalize(seq_rep, dim=-1)
        pure = self.multiply([mask, seq_rep_norm])
        matrices = self.outer(pure)
        weights = self.norm(seq_rep)
        weights = F.softmax(weights, dim=-1)
        in_states = self.mixture([[matrices], weights])
        quantum_output = []
        for _h in in_states:
            measurement_probs = self.measurement(_h)
            quantum_output.append(measurement_probs)
        quantum_output = torch.stack(quantum_output, dim=-2)
        tagger_input =self.projections(tagger_input)
        tagger_input = tagger_input @ torch.transpose(tagger_input, -1, -2)
    
        tagger_input = nn.ReLU()(self.liner(tagger_input))

 
       
        
        tagger_input = torch.cat([outputs.last_hidden_state, tagger_input, quantum_output], dim=-1)
        
        tagger_input = self.bert_drop(tagger_input)

        
        
        h = self.linear_in(tagger_input) 
 
        er = self.deprel_embedding_layer(adj_dep) 
        ei= self.deprel_embedding_layer_i(adj_dep)

        er = er.mean(dim=3)
        ei = ei.mean(dim=3)
        
        e = torch.complex(er,ei)
        h=torch.add(h,e)

        aspect_words_num = aspect_mask.sum(dim=1).unsqueeze(-1) # (B, L) -> (B, 1) 
        aspect_mask = aspect_mask.unsqueeze(-1) # (B, L, 1) 
        out = (h * aspect_mask).sum(dim=1) / aspect_words_num # (B, F) / (B, 1) 
        out = torch.cat([out, pooled_output], dim=-1) 
        out = torch.abs(out)
        out = self.linear_out(out) 
      
        return out 
