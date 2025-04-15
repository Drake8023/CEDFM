import torch
from torch import nn
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def clear_dia(x):
        real=x.real
        imag=x.imag
        eye=torch.eye(real.shape[-1],real.shape[-1]).to(x.device)
        eye=eye*-1+1
        imag=imag*eye
        r=torch.complex(real,imag)
        return r
class Attention_1(nn.Module):
    def __init__(self, KV_size, num_attention_heads, attention_probs, Q_size, batchsize=16, trans_dim=100):
        super().__init__()
        if KV_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (KV_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(KV_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        self.query = linear_c(Q_size, batchsize, trans_dim)
        self.key = linear_c(KV_size, batchsize, trans_dim)
        self.value = linear_c(KV_size, batchsize, trans_dim)

        self.dropout = nn.Dropout(attention_probs)
  
    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def cal_score(self, x, y):

        result=torch.matmul(x,y.transpose(-1,-2))

        result=result.real
        x_imag=x.imag
        y_imag=y.imag
        result_bu=torch.matmul(x_imag,y_imag.transpose(-1,-2))
        output_real=result+2*result_bu

        return output_real
    def forward(self, KV, Q, K_mask, Q_mask):

        mixed_query_layer = self.query(Q)
        mixed_key_layer = self.key(KV)
        mixed_value_layer = self.value(KV)
        batch_size = mixed_key_layer.shape[0]
        seq_len = mixed_key_layer.shape[1]
        embedding_size = mixed_key_layer.shape[2]
        attention_scores = self.cal_score(mixed_query_layer, mixed_key_layer) 

        if K_mask is not None:
    
            K_mask = (1-K_mask)*(-100000)
            K_mask = K_mask.unsqueeze(1)
            attention_scores = attention_scores + K_mask
      
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        
  
        value_layer1 = mixed_value_layer.reshape(batch_size,seq_len,embedding_size)
        V_real=value_layer1.real
        V_imag=value_layer1.imag

        context_layer_real = torch.matmul(attention_probs, V_real)
        context_layer_imag = torch.matmul(attention_probs, V_imag)
        if Q_mask is not None:
    
            Q_mask = Q_mask.unsqueeze(-1)
        
            context_layer_real = context_layer_real * Q_mask
            context_layer_imag = context_layer_imag * Q_mask
        context_layer1 = torch.complex(context_layer_real,context_layer_imag)
        output=context_layer1
        return output    
class my_attention1_1(nn.Module):
    def __init__(self, KV_size, Q_size, nhead=1, dropout=0.01, batchsize=16,trans_dim=100):
        super(my_attention1_1, self).__init__()
        self.context_cross_attention = Attention_1(KV_size, nhead, dropout, Q_size=Q_size, batchsize=batchsize, trans_dim=trans_dim)
     
        self.fc1 =linear_c(Q_size,batchsize,trans_dim)
        self.dropout1 = QDropout(dropout)

    def forward(self, KV, Q, K_mask, Q_mask):
        new_src = self.context_cross_attention(KV, Q, K_mask, Q_mask)
        cross_src = 0.5*new_src+0.5*Q
        cross_src_1 = self.fc1(cross_src)
        # print(cross_src_1.shape)
        cross_src2 = 0.5*cross_src+0.5*self.dropout1(cross_src_1)

        return cross_src2
class PositionEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, input_dim=1, zero_phase=False, device=torch.device('cuda')):
        super(PositionEmbedding, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.zero_phase = zero_phase


        frequency_inits = 1 / torch.pow(10000, torch.true_divide(torch.arange(embed_dim), embed_dim))
        frequency_matrix = frequency_inits.repeat(self.input_dim, 1)
        self.frequency_embedding = nn.Embedding.from_pretrained(frequency_matrix)

        phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        self.phase_embedding = nn.Embedding.from_pretrained(phase_matrix)


    def forward(self, x):

        # No speaker embedding
        if self.input_dim == 1:
            x = torch.zeros_like(x)
        phases = self.phase_embedding(x)
        phases = 2 * 3.14 * nn.Sigmoid()(phases)

        time_stamps = x.shape[1]

        positions = torch.arange(time_stamps).unsqueeze(-1).to(self.device)
        pos_embed = positions.repeat(1, self.embed_dim) * self.frequency_embedding(x) + phases
        if self.zero_phase:
            pos_embed = torch.zeros_like(pos_embed)

        return pos_embed
class QNorm(torch.nn.Module):
    def __init__(self, embed_dim):
        super(QNorm, self).__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self,x):                  

        x_real = self.norm(x.real)
        x_imag = self.norm(x.imag)

        result = torch.complex(x_real, x_imag)
        
        return result
class CDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            real_part = nn.functional.dropout(x.real, self.p, self.training)
            imag_part = nn.functional.dropout(x.imag, self.p, self.training)
            return torch.complex(real_part, imag_part)
        else:
            return x
class QDropout(torch.nn.Module):
    def __init__(self, p=0.2):
        super(QDropout, self).__init__()
        self.p = p
    def forward(self, x):
        batch_size = len(x)
        seq_len = x.shape[1]
        dimension = x.shape[-1]
        eye=torch.eye(dimension).to(x.device)

        x_eye=x*eye
        b=torch.triu(torch.bernoulli(torch.ones(batch_size,seq_len,dimension)*(1-self.p))).to(x.device)
        b=b+b.permute(0,2,1)
        eye_fan=(eye==0)
        b=b*eye_fan
        y=x*b
        y=y+x_eye        
        return y

class ComplexMultiply(torch.nn.Module):
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        phase = inputs[0]
        amplitude = inputs[1]

        if amplitude.dim() == phase.dim() + 1:  # Assigning each dimension with same phase
            cos = torch.unsqueeze(torch.cos(phase), dim=-1)
            sin = torch.unsqueeze(torch.sin(phase), dim=-1)

        elif amplitude.dim() == phase.dim():  # Each dimension has different phases
            cos = torch.cos(phase)
            sin = torch.sin(phase)


        else:
            raise ValueError('input dimensions of phase and amplitude do not agree to each other.')

        real_part = cos * amplitude
        imag_part = sin * amplitude

        return [real_part, imag_part]
class linear_c(nn.Module):
    def __init__(self, embed_dim,batch_size,seq_len):
        super(linear_c, self).__init__()
        self.unitary = torch.nn.Parameter(torch.stack([torch.eye(embed_dim),torch.zeros(embed_dim, embed_dim)],dim = -1))
        self.batch_size = batch_size
        self.seq_len = seq_len
    def forward(self, x):#batch_size*seq_len*embedding*embedding

        U_real = self.unitary[:,:,0]
  
        U_imag = self.unitary[:,:,1]
   
        U = torch.complex(U_real,U_imag)

        U_H = torch.conj(U).permute(1,0)

        p = torch.matmul(U,x)

        output = torch.matmul(p,U_H)
        return output

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    
        self.real_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.randn(out_features, in_features))

       
        self.real_bias = nn.Parameter(torch.randn(out_features))
        self.imag_bias = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        real_input = input.real
        imag_input = input.imag

      
        real_output = torch.matmul(real_input, self.real_weight.t()) - torch.matmul(imag_input, self.imag_weight.t())
        imag_output = torch.matmul(real_input, self.imag_weight.t()) + torch.matmul(imag_input, self.real_weight.t())

    
        real_output = real_output + self.real_bias
        imag_output = imag_output + self.imag_bias

        return torch.complex(real_output, imag_output)
class QOuter(torch.nn.Module):
    def __init__(self):
        super(QOuter, self).__init__()

    def forward(self, x):

        if not isinstance(x, list):
            raise ValueError('xr should be called '
                             'on a list of 2 inputs.')

        if len(x) != 2:
            raise ValueError('x should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(x)) + ' inputs.')
        print(x[0].shape)
        print(x[1].shape)
        real = x[0].transpose(0, 1)
        imag = x[1].transpose(0, 1)
        output = []
        for r, i in zip(real, imag):
            output_rr = []
            output_ii = []
            for rr, ii in zip(r, i):
                unsqueezed_rr = torch.unsqueeze(rr, dim=-1)
                unsqueezed_ii = torch.unsqueeze(ii, dim=-1)
                _r = torch.mm(unsqueezed_rr, unsqueezed_rr.t()) + torch.mm(unsqueezed_ii, unsqueezed_ii.t())
                _i = -torch.mm(unsqueezed_rr, unsqueezed_ii.t()) + torch.mm(unsqueezed_ii, unsqueezed_rr.t())

                output_rr.append(_r)
                output_ii.append(_i)

            output_rr = torch.stack(output_rr, dim=0)
            output_ii = torch.stack(output_ii, dim=0)
            output.append([output_rr, output_ii])
        print(f"output length: {len(output)}")
        if len(output) > 0:
            print(f"First element shapes: {output[0][0].shape}, {output[0][1].shape}")
        return output

class QOuterC(torch.nn.Module):
    def __init__(self):
        super(QOuterC, self).__init__()

    def forward(self, x):

        if not isinstance(x, list):
            raise ValueError('xr should be called '
                             'on a list of 2 inputs.')

        if len(x) != 2:
            raise ValueError('x should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(x)) + ' inputs.')

        real = x[0].transpose(0, 1)
        imag = x[1].transpose(0, 1)
        output = []
        for r, i in zip(real, imag):
            
            output_complex = []
            for rr, ii in zip(r, i):
             
                complex_tensor = torch.complex(rr, ii)
                unsqueezed_c = torch.unsqueeze(complex_tensor, dim=-1)
                
                outer_product = torch.matmul(unsqueezed_c, unsqueezed_c.conj().transpose(-1, -2))
                output_complex.append(outer_product)

            output_complex = torch.stack(output_complex, dim=0)
            output.append(output_complex)
     
  
        return output

class QMeasurementC(torch.nn.Module):
    def __init__(self, embed_dim, device=torch.device('cuda')):
        super(QMeasurementC, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.kernel = torch.nn.Parameter(
            torch.stack([torch.eye(embed_dim).to(self.device), torch.zeros(embed_dim, embed_dim).to(self.device)],
                        dim=-1))

    def forward(self, inputs):

    
        real_kernel = self.kernel[:, :, 0]
        imag_kernel = self.kernel[:, :, 1]

        real_kernel = real_kernel.unsqueeze(-1)
        imag_kernel = imag_kernel.unsqueeze(-1)

        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
                            + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
                            - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))

        projector =torch.complex(projector_real,projector_imag)  

        
        
        output = torch.matmul(torch.flatten(inputs, start_dim=-2, end_dim=-1),
                                   torch.flatten(projector, start_dim=-2, end_dim=-1).t())

  
       

        return output
class QMeasurement(torch.nn.Module):
    def __init__(self, embed_dim, device=torch.device('cuda')):
        super(QMeasurement, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.kernel = torch.nn.Parameter(
            torch.stack([torch.eye(embed_dim).to(self.device), torch.zeros(embed_dim, embed_dim).to(self.device)],
                        dim=-1))

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        input_real = inputs[0]
        input_imag = inputs[1]

        real_kernel = self.kernel[:, :, 0]
        imag_kernel = self.kernel[:, :, 1]

        real_kernel = real_kernel.unsqueeze(-1)
        imag_kernel = imag_kernel.unsqueeze(-1)

        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
                         + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
                         - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
      
     
        output_real = torch.matmul(torch.flatten(input_real, start_dim=-2, end_dim=-1),
                                   torch.flatten(projector_real, start_dim=-2, end_dim=-1).t()) \
                      - torch.matmul(torch.flatten(input_imag, start_dim=-2, end_dim=-1),
                                     torch.flatten(projector_imag, start_dim=-2, end_dim=-1).t())

        return output_real


class QMixture(torch.nn.Module):

    def __init__(self, use_weights=True, device=torch.device('cuda')):
        super(QMixture, self).__init__()
        self.use_weights = use_weights
        self.device = device

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        in_modalities = inputs[0]  # [modal_1,...modal_n], each being a list of [real, imag] arrays

        weights = inputs[1].transpose(0, 1)  # (time_stamps, batch_size, num_modalities)
        embed_dim = in_modalities[0][0][0].shape[-1]
     
        outputs = []

        for reps_t in zip(*in_modalities, weights):
           
            multimodal_rep = [torch.stack(rep_field, dim=-1) for rep_field in zip(*reps_t[:-1])]
      
         
            w = reps_t[-1].unsqueeze(dim=1).unsqueeze(dim=-1).expand(-1, embed_dim, -1, -1)
           #([16, 102, 1, 1])
            output_rep = [torch.matmul(_rep, w).squeeze(dim=-1) for _rep in multimodal_rep]
           
            
            outputs.append(output_rep)

        return outputs

class QMixtureC(torch.nn.Module):

    def __init__(self, use_weights=True, device=torch.device('cuda')):
        super(QMixtureC, self).__init__()
        self.use_weights = use_weights
        self.device = device

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        in_modalities = inputs[0]  # [modal_1,...modal_n], each being a list of [real, imag] arrays
   
        weights = inputs[1].transpose(0, 1)  # (time_stamps, batch_size, num_modalities)
        embed_dim = in_modalities[0][0].shape[-1]

        outputs = []

        for reps_t in zip(*in_modalities, weights):

            multimodal_rep = torch.stack(reps_t[:-1], dim=-1)
       
            w = reps_t[-1].unsqueeze(dim=1).unsqueeze(dim=-1).expand(-1, embed_dim, -1, -1).to(torch.complex64)
   
            output_rep = torch.matmul(multimodal_rep, w).squeeze(dim=-1)
     
            outputs.append(output_rep)

        return outputs



class PositionEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, input_dim=1, zero_phase=False, device=torch.device('cuda')):
        super(PositionEmbedding, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.zero_phase = zero_phase

        # Vaswani et al.
        frequency_inits = 1 / torch.pow(10000, torch.true_divide(torch.arange(embed_dim), embed_dim))
        frequency_matrix = frequency_inits.repeat(self.input_dim, 1)
        self.frequency_embedding = nn.Embedding.from_pretrained(frequency_matrix)

        self.frequency_embedding.weight.requires_grad = True
        phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        self.phase_embedding = nn.Embedding.from_pretrained(phase_matrix)
        self.phase_embedding.weight.requires_grad = True


    def forward(self, x):

        # No speaker embedding
        if self.input_dim == 1:
            x = torch.zeros_like(x)
        phases = self.phase_embedding(x)
        phases = 2 * 3.14 * nn.Sigmoid()(phases)

        time_stamps = x.shape[1]

        positions = torch.arange(time_stamps).unsqueeze(-1).to(self.device)
        pos_embed = positions.repeat(1, self.embed_dim) * self.frequency_embedding(x) + phases
        if self.zero_phase:
            pos_embed = torch.zeros_like(pos_embed)
     

        return pos_embed
