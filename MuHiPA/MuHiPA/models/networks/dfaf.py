
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

from MuHiPA.datasets.block.models.networks.vqa_net import factory_text_enc





class DFAF(nn.Module):
    """
    Implementation of Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering (DFAF)
    Based on code from https://github.com/Cyanogenoid/vqa-counting
    """
    def __init__(self, txt_enc={}, 
                 output_size=100,
                 output_features=2400,
                 question_features = 2400,
                 vision_features = 2400,
                 hidden_features = 512,
                 num_inter_head = 8,
                 num_intra_head = 8,
                 num_block = 2,
                 visual_normalization = True,
                 wid_to_word={},
                 word_to_wid={},
                 aid_to_ans=[],
                 ans_to_aid={},
                 max_answers=3000,
                 t_emb = False
                 ):
        super(DFAF, self).__init__()
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        self.question_features = question_features
        self.vision_features = vision_features
        self.hidden_features = hidden_features
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block
        self.visual_normalization = visual_normalization
        self.max_answers = max_answers
        # self.output_features = output_features
        # self.output_size = output_size
        self.t_emb = t_emb
        
        
        if self.t_emb:
            self.tag_embedding = torch.nn.Embedding(1601, 1240)

        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        
        assert(self.hidden_features % self.num_inter_head == 0)
        assert(self.hidden_features % self.num_intra_head == 0)
        # words_list = list(words_list)
        # words_list.insert(0, '__unknown__')

        # self.text = word_embedding.TextProcessor(
        #     classes=words_list,
        #     embedding_features=300,
        #     lstm_features=self.question_features,
        #     drop=0.1,
        # )

        self.interIntraBlocks = SingleBlock(
            num_block=self.num_block,
            v_size=self.vision_features,
            q_size=self.question_features,
            output_size=self.hidden_features,
            num_inter_head=self.num_inter_head,
            num_intra_head=self.num_intra_head,
            drop=0.1,
        )
        
        
        # self.interIntraBlocks = MultiBlock(
        #     num_block=self.num_block, v_size, q_size, output_size, num_head, drop=0.0):

        self.classifier = Classifier(
            in_features=self.hidden_features,
            mid_features=1024,
            out_features=self.max_answers,
            drop=0.1,)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
                    
    def process_question(self, q, l):
        q_emb = self.txt_enc.embedding(q)
        q_rnn, q_hidden = self.txt_enc.rnn(q_emb) #(bs, 10, 2400)
        return q_rnn
    
    
    # def max_question_length(self):
    #     if not hasattr(self, '_max_length'):
    #         self._max_length = max(map(len, self.questions))
    #     return self._max_length
    
    def get_mask(self, l, n_regions):
        
        
        # q_mask = torch.from_numpy((np.arange(self._max_length) < l).astype(int))
        
        
        q_mask = torch.zeros((l.shape[0], n_regions), device='cuda')
#        q_mask = torch.cuda.FloatTensor(l.shape[0], n_regions).fill_(0)
        for i in range(l.shape[0]):
            for j in range(l[0]):
                q_mask[i,j] = 1

        return q_mask 
    
    def forward(self, batch):
        '''
        v: visual feature      [batch, 2048, num_obj]
        b: bounding box        [batch, 4, num_obj]
        q: question            [batch, max_q_len]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        answer: predict logits [batch, config.max_answers]
        '''
        
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        b = batch['norm_coord']
        
        
        
        n_regions = v.shape[1]
        bsize = q.shape[0]   
        num_words = q.shape[1]
        missing_words = n_regions - num_words
        zeros = torch.cuda.LongTensor(bsize, missing_words).fill_(0)
        q = torch.cat((q, zeros), dim = 1)
        q_mask = self.get_mask(l, n_regions)
        

        #process question
        q = self.process_question(q, l) #(20,4800)
        # q_mask : q_mask[bsize, n_regions: None].expand(v.size(0), n_regions, q.size(1))
        #original
        # q = self.text(q)  # [batch, max_len, 1280]
        
        cls_score = None
        cls_text = None
        cls_id = None
        if self.t_emb:
            cls_score = batch['cls_scores']
            cls_text = batch['cls_text']
            cls_id = batch['cls']
            cls_emb = self.tag_embedding(cls_id)
            # cls_score = cls_score[:,:,None,None].expand(cls_score.size(0), cls_score.size(1), cls_emb.size(2), cls_emb.size(3))
            cls_score = cls_score[:,:,None].expand(cls_score.size(0), cls_score.size(1), cls_emb.size(2))
            cls_emb *= cls_score
            t = cls_emb
            
            v = torch.cat((t, v), 2)
            
            q2_shape = v.shape[2] - q.shape[2]
            q2 = torch.zeros((q.shape[0], q.shape[1], q2_shape), device='cuda')
            q = torch.cat((q, q2), 2)
            
        else:
            
            v2_shape = q.shape[2] - v.shape[2]
            v2 = torch.zeros((v.shape[0], v.shape[1], v2_shape), device='cuda')
            v = torch.cat((v, v2), 2)
            # q_mask = self.get_mask(l, n_regions)
        # prepare v & q features
        # v = v.transpose(1,2).contiguous()
        # b = b.transpose(1,2).contiguous()
        
        
        if self.visual_normalization:
            v = v / (v.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(v) # [batch, max_obj, 2048]
            
        v_mask = torch.ones((l.shape[0], n_regions), device='cuda')
        v_mask = v_mask.float()
        q_mask = q_mask.float()
        
        # print("-------------------------- ", q.shape)
        # print("-------------------------- ", v.shape)
        
        
        v, q = self.interIntraBlocks(v, q, q_mask, v_mask)
        v = F.normalize(v,p=2) 
        q = F.normalize(q,p=2) 
        
        answer = self.classifier(v, q, q_mask, v_mask)
        # print("-------------------------- ", answer.shape)
        
        
        out = {'logits': answer}
        
        out = self.process_answers(out)
        return out
    
    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        _, pred = out['logits'].data.max(1)
        pred.squeeze_()
        out['answers'] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out['answer_ids'] = [pred[i] for i in range(batch_size)]
        return out

class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)

class ReshapeBatchNorm(nn.Module):
    def __init__(self, feat_size, affine=True):
        super(ReshapeBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(feat_size, affine=affine)

    def forward(self, x):
        assert(len(x.shape) == 3)
        batch_size, num, _ = x.shape
        x = x.view(batch_size * num, -1)
        x = self.bn(x)
        return x.view(batch_size, num, -1)

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        #self.fusion = Fusion()
        self.lin1 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.bn = nn.BatchNorm1d(mid_features)

    def forward(self, v, q, q_mask, v_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        out = self.lin1(self.drop(v_mean * q_mean))
        out = self.lin2(self.drop(self.relu(self.bn(out))))
        return out

class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = nn.Linear(v_size, output_size)
        self.q_lin = nn.Linear(q_size, output_size)

        self.interBlock = InterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

        self.drop = nn.Dropout(drop)

    def forward(self, v, q, q_mask, v_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        # transfor features
        v = self.v_lin(self.drop(v))
        q = self.q_lin(self.drop(q))
        for i in range(self.num_block):
            v, q = self.interBlock(v, q, v_mask, q_mask)
            v, q = self.intraBlock(v, q, v_mask, q_mask)
            v = F.normalize(v,p=2) 
            q = F.normalize(q,p=2) 

        return v,q

class MultiBlock(nn.Module):
    """
    Multi Block Inter-/Intra-modality
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_head, drop=0.0):
        super(MultiBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.num_block = num_block

        blocks = []
        blocks.append(InterModalityUpdate(v_size, q_size, output_size, num_head, drop))
        blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_head, drop))
        for i in range(num_block - 1):
            blocks.append(InterModalityUpdate(output_size, output_size, output_size, num_head, drop))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_head, drop))
        self.multi_blocks = nn.ModuleList(blocks)

    def forward(self, v, q, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        for i in range(self.num_block):
            v, q = self.multi_blocks[i*2+0](v, q, v_mask, q_mask)
            v, q = self.multi_blocks[i*2+1](v, q, v_mask, q_mask)
        return v,q

class InterModalityUpdate(nn.Module):
    """
    Inter-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_output = nn.Linear(output_size + v_size, output_size)
        self.q_output = nn.Linear(output_size + q_size, output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        # transfor features
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        # mask all padding object/word features
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply multi-head
        vk_set = torch.split(v_k, v_k.size(2) // self.num_head, dim=2)
        vq_set = torch.split(v_q, v_q.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(q_k, q_k.size(2) // self.num_head, dim=2)
        qq_set = torch.split(q_q, q_q.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
            # inner product & set padding object/word attention to negative infinity & normalized by square root of hidden dimension
            q2v = (vq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            v2q = (qq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, max_len, num_obj]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            # softmax attention
            interMAF_q2v = F.softmax(q2v, dim=2) #[batch, num_obj, max_len]
            interMAF_v2q = F.softmax(v2q, dim=2) #[batch, max_len, num_obj]
            # calculate update input (each head of multi-head is calculated independently and concatenate together)
            v_update = interMAF_q2v @ qv_slice if (i==0) else torch.cat((v_update, interMAF_q2v @ qv_slice), dim=2)
            q_update = interMAF_v2q @ vv_slice if (i==0) else torch.cat((q_update, interMAF_v2q @ vv_slice), dim=2)
        # update new feature
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)
        updated_v = self.v_output(self.drop(cat_v))
        updated_q = self.q_output(self.drop(cat_q))
        return updated_v, updated_q

class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = nn.Linear(v_size, output_size)
        self.q4v_gate_lin = nn.Linear(q_size, output_size)

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_output = nn.Linear(output_size, output_size)
        self.q_output = nn.Linear(output_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
    
    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        # conditioned gating vector
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        v4q_gate = self.sigmoid(self.v4q_gate_lin(self.drop(self.relu(v_mean)))).unsqueeze(1) #[batch, 1, feat_size]
        q4v_gate = self.sigmoid(self.q4v_gate_lin(self.drop(self.relu(q_mean)))).unsqueeze(1) #[batch, 1, feat_size]

        # key, query, value
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        # mask all padding object/word features
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply conditioned gate
        new_vq = (1 + q4v_gate) * v_q
        new_vk = (1 + q4v_gate) * v_k
        new_qq = (1 + v4q_gate) * q_q
        new_qk = (1 + v4q_gate) * q_k

        # apply multi-head
        vk_set = torch.split(new_vk, new_vk.size(2) // self.num_head, dim=2)
        vq_set = torch.split(new_vq, new_vq.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(new_qk, new_qk.size(2) // self.num_head, dim=2)
        qq_set = torch.split(new_qq, new_qq.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
            # calculate attention
            v2v = (vq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, num_obj, num_obj]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            q2q = (qq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, max_len, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            dyIntraMAF_v2v = F.softmax(v2v, dim=2)
            dyIntraMAF_q2q = F.softmax(q2q, dim=2)
            # calculate update input
            v_update = dyIntraMAF_v2v @ vv_slice if (i==0) else torch.cat((v_update, dyIntraMAF_v2v @ vv_slice), dim=2)
            q_update = dyIntraMAF_q2q @ qv_slice if (i==0) else torch.cat((q_update, dyIntraMAF_q2q @ qv_slice), dim=2)
        # update
        updated_v = self.v_output(self.drop(v + v_update))
        updated_q = self.q_output(self.drop(q + q_update))
        return updated_v, updated_q
