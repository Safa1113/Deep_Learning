from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options, OptionsDict
from bootstrap.lib.logger import Logger
import MuHiPA.datasets.block as block
from MuHiPA.datasets.block.models.networks.vqa_net import factory_text_enc
from MuHiPA.datasets.block.models.networks.vqa_net import mask_softmax
from MuHiPA.datasets.block.models.networks.mlp import MLP
from .reasoning import MuHiPAReasoning
from .visualize import Visualize
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import time 
import json

#import block
#from block.models.networks.vqa_net import factory_text_enc
#from block.models.networks.vqa_net import mask_softmax
#from block.models.networks.mlp import MLP


class MuHiPA(nn.Module):

    def __init__(self,
            txt_enc={},
            self_q_att=False,
            self_q_att_2=False,
            self_v_att={},
            self_t_att=True,
            n_step=3,
            shared=False,
            cell={},
            agg={},
            # t_embd=True,
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={}):
        super(MuHiPA, self).__init__()
        self.self_q_att = self_q_att
        self.self_q_att_2 = self_q_att_2
        self.n_step = n_step
        self.shared = shared
        self.self_v_att = self_v_att
        # self.t_embd = t_embd
        
        if self_t_att:
            self_t_att ={ 'output_dim': 1, 'mlp_glimpses': 2,
          'fusion':{
            'type': 'block',
            'input_dims': [4800, 620],
            'output_dim': 1,
            'mm_dim': 1600,
            'chunks': 20,
            'rank': 10,
            'dropout_input': 0.1,
            'dropout_pre_lin': 0.0}}
        
        self.self_t_att = self_t_att
        self.cell = cell
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean', 'none', 'sum']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        # Modules
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        
        # self.outputm = nn.Linear(2048 * 36, 2048)
        # self.outputq = nn.Linear(2400 * 36, 2400)
        # 
#        self.qqlast = nn.Linear(4800, 4800)
#        self.mmlast = nn.Linear(4096, 4096)


        #same size
        # self.vtoq_size = nn.Linear(2048, 2400)
        # vtoq_size_fusion_dic = {'type': 'block',
        # 'input_dims': [4800, 4096],
        # 'output_dim': 4800,
        # 'mm_dim': 1600,
        # 'chunks': 20,
        # 'rank': 10,
        # 'dropout_input': 0.1,
        # 'dropout_pre_lin': 0.}
        # self.vtoq_size_fusion = block.factory_fusion(vtoq_size_fusion_dic)
        
        #self.q_classif_linear = nn.Linear(2480, 30)
        #self.classif_embedding = torch.nn.Embedding(30, 1000)
        
        
        
        self.tag_embedding = torch.nn.Embedding(1601, 1240)
        
        
        
        #self.DyIntraModalityUpdate = DyIntraModalityUpdate(2048, 620, 2048, 8, drop=0.0)
        
        # concept ----------------------------------------------
        #self.concept_no = 100
        #self.concept_size = 1000
        #self.index = nn.Linear(2048, self.concept_no)
        #self.concept_embedding = torch.nn.Embedding(self.concept_no, self.concept_size)
        
        
        if self.self_q_att:
            # self.q_linear0 = nn.Linear(2400, 1200)
            # self.q_att_linear0 = nn.Linear(768, 512)
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        if self.self_v_att:
            self.mlp_glimpses = 2
            # self.mlp_glimpses = self_v_att['mlp_glimpses']
            self.fusion = block.factory_fusion(self_v_att['fusion'])
            self.linear0 = nn.Linear(self_v_att['output_dim'], 512)
            self.linear1 = nn.Linear(512, self.mlp_glimpses)
            
        if self.self_t_att:
            # self.mlp_glimpses = 2
            self.mlp_glimpses = self_t_att['mlp_glimpses']
            self.tfusion = block.factory_fusion(self_t_att['fusion'])
            self.tlinear0 = nn.Linear(self_t_att['output_dim'], 512)
            self.tlinear1 = nn.Linear(512, self.mlp_glimpses)
            
            
        #--------------------------newww
        if self.self_q_att_2:
            self.linear0_q = nn.Linear(1, 512)
            self.linear1_q = nn.Linear(512, self.mlp_glimpses)
            self.fusion_q_att = block.factory_fusion({'type': 'block', 
                                                          'input_dims': [2400, 6576], #[4800, 2048],5336  6576 4716
                                                          'output_dim': 1,
                                                          'mm_dim': 1600,
                                                          'chunks': 20,
                                                          'rank': 10,
                                                          'dropout_input': 0.1,
                                                          'dropout_pre_lin': 0.})
            
            

        if self.shared:
            self.cell = MuHiPAReasoning(**cell)
        else:
            self.cells = nn.ModuleList([MuHiPAReasoning(**cell) for i in range(self.n_step)])
        

        if 'fusion' in self.classif:
            self.classif_module = block.factory_fusion(self.classif['fusion'])
        elif 'mlp' in self.classif:
            self.classif_module = MLP(self.classif['mlp'])
        else:
            raise ValueError(self.classif.keys())

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        Logger().log_value('nparams_txt_enc',
            self.get_nparams_txt_enc(),
            should_print=True)
        
        Logger().log_value('nparams_q_attention',
            self.get_nparams_qattention(),
            should_print=True)
        
        Logger().log_value('nparams_v_attention',
            self.get_nparams_vattention(),
            should_print=True)
        
        Logger().log_value('nparams_class',
            self.get_nparams_class(),
            should_print=True)
        
        Logger().log_value('nparams_classifyer',
            self.get_nparams_classifyer(),
            should_print=True)

        self.buffer = None
        self.vis = Visualize(True, wid_to_word =self.wid_to_word)
        #self.BERT = BERT()
        
        #self.fusion_module_vt = block.factory_fusion({'type': 'block', 
#                                                          'input_dims': [2048, 310], #[4800, 2048],
#                                                          'output_dim': 2048,
#                                                          'mm_dim': 1600,
#                                                          'chunks': 20,
#                                                          'rank': 10,
#                                                          'dropout_input': 0.1,
#                                                          'dropout_pre_lin': 0.})

        self.representations = []
        self.data = []

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
#        if self.self_q_att:
#            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
#            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)


    def get_nparams_qattention(self):
        params = []
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def get_nparams_vattention(self):
        
        params = []
        if self.self_v_att:
            params += [p.numel() for p in self.linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.linear1.parameters() if p.requires_grad]
            params += [p.numel() for p in self.fusion.parameters() if p.requires_grad]
        return sum(params)    

    def get_nparams_class(self):
        params = []

        params += [p.numel() for p in self.tag_embedding.parameters() if p.requires_grad]
        return sum(params)
    
    def get_nparams_classifyer(self):
        params = []

        params += [p.numel() for p in self.classif_module.parameters() if p.requires_grad]
        return sum(params)
    
    def set_buffer(self):
        self.buffer = {}
        if self.shared:
            self.cell.pairwise.set_buffer()
        else:
            for i in range(self.n_step):
                self.cell[i].pairwise.set_buffer()

    def set_pairs_ids(self, n_regions, bsize, device='cuda'):
        if self.shared and self.cell.pairwise:
            self.cell.pairwise_module.set_pairs_ids(n_regions, bsize, device=device)
        else:
            for i in self.n_step:
                if self.cells[i].pairwise:
                    self.cells[i].pairwise_module.set_pairs_ids(n_regions, bsize, device=device)

    def forward(self, batch):

        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        # if 'norm_coord' in batch.keys():
        c = batch['norm_coord']
        
        cls_score = batch['cls_scores']
        cls_text = batch['cls_text']
        cls_id = batch['cls']
        # print(batch.keys())
        batch_size = v.shape[0]
        # print("----------------------", v.shape)
        
        # for k, v in batch.items():
        #      print(k)
        
        
        # print("tttttttttttttttttttttttt", l.shape)
        
        # print(batch['question'][0])
        # print(batch['question_type'][0])
        # print(batch['question_id'][0])
        # print(batch['image_name'][0])
        # print(batch['index'][0])
        # print(batch['answer_id'][0])
        # print(batch['answer'][0])
        # print(self.vis.get_question(batch['question'][0]))
        
        # print("HHHHHHHHHHHHHHHHH", len(self.ans_to_aid.keys()))
        
        
        
  
    
        
        #tag processing
        cls_emb = self.tag_embedding(cls_id)
        cls_score = cls_score[:,:,None].expand(cls_score.size(0), cls_score.size(1), cls_emb.size(2))
        cls_emb *= cls_score
        t = cls_emb
        
        
        
        # for key, value in batch.items() :
            # print (key)
        # print(len(cls_text))
        # print(batch['cls_scores'].shape)
        # print(batch['cls'].shape)
        # print(batch['cls'][0])

        # l = l+2
        # bert = self.vis.get_question(q)
        # bert = self.BERT(bert)
        # print("bbbbbbbrrt", bert.requires_grad)

        n_regions = v.shape[1]
        bsize = q.shape[0]   
        num_words = q.shape[1]
        missing_words = n_regions - num_words
        zeros = torch.cuda.LongTensor(bsize, missing_words).fill_(0)
        q = torch.cat((q, zeros), dim = 1)
        
        
        # q_mask = self.get_mask(l, n_regions)
    
    
        
        # q_classif = self.process_classif(q)
        # # self.vis.question_type(batch, q_classif, "q_classif------------------")
        # _, q_classif_idx = torch.max(q_classif, 1)
        # q_classif_emb = self.classif_embedding(q_classif_idx) 


        #process question
        q = self.process_question(q, l) #(20,4800)
        # q = Variable(bert)
        # q = bert
        # q_visualize = q
        # qq_mask = q_mask[:,:,None].expand(q_mask.shape[0], q_mask.shape[1], q.shape[2])
        # qq_mask = q_mask
        # qq_mask = 0
        
        # -------------------------------------- Visualization
        self.vis.setBatch(batch)
        # self.vis.showJustImage(batch)
        #self.vis.images(visual =v, title = "Image original values")
        #self.vis.images(batch)
        # self.vis.question(batch, q)
        # -------------------------------------- Visualization
    

        # v = self.vtoq_size(F.dropout(v, 0.1))
        
        if self.self_q_att:
            # print("bqqqqbbrrt", q.requires_grad)
            q = self.question_attention(q, l)
            # q = q.sum(1)
            # print("bqqqqbbrrt", q.requires_grad)
            # print("bqqqefergrrt", q.shape)
            
        q1 = q    
            
        
        # print("====================", q.shape)
            
            
            
            
            
        #concept
        # index = torch.FloatTensor([0]).cuda()
        # print(v.shape)
        # index = self.index(F.dropout(v, p=0.1))
        # index = F.sigmoid(index)
        # concept = index @ self.concept_embedding.weight
        # t = concept
        # print(concept.shape)
        # v = concept
        if self.self_v_att:
            # for i in range (3):
               # v = self.DyIntraModalityUpdate(v,t)

            # v = self.fusion_module_vt([v.contiguous().view(batch_size*n_regions, -1),
            # t.contiguous().view(batch_size*n_regions, -1)])
            # v = v.view(batch_size, n_regions, -1)
            # v = torch.cat((concept, v), 2)
            v = torch.cat((t, v), 2)
            
            v = self.image_attention(q,v)

            # v = torch.cat((concept, v), 1)

        # if self.self_t_att:
        #     # t = self.tag_attention(q,t)
        #     pass
            # print(t.shape)
            # print(t)
        
        # v = torch.cat((q_classif_emb, v), 1)
        # t = torch.cat((q_classif_emb, t), 1)

        # cell
        qq = q #* qq_mask
        mm = v
        # mm = torch.cat((t, v), dim=1) 
        # tt = t
        # joint1 = torch.cat((qq, mm), dim=1)
        # joint2 = torch.cat((qq, mm), dim=1)
        # joint2 = self.vtoq_size_fusion([q,v])
        # result = []
        # all_vvv = []
        vvv = []
        vvv.append(self.classif_module([qq, mm]))
        # print("fwwwwwwwwwwwwwwwwwww", vvv[0].shape)
        
        # mm = F.normalize(mm, p=2.0, dim=1, eps=1e-12, out=None)
        # qq = F.normalize(qq, p=2.0, dim=1, eps=1e-12, out=None)
        
        buff = []
        if self.cell:
            for i in range(self.n_step):
                
#                self.vis.argmax_change_image(features = mm, title="cell number" + str(i))
#                self.vis.argmax_change_question(features = qq, title="cell number" + str(i))
  
#                tempm = mm
                
                cell = self.cell if self.shared else self.cells[i]
                
                # -------------------------------------- Visualization
                # self.vis.seperator("cccccccccccccccccccc"+str(i))
                # -------------------------------------- Visualization
                # result.append([qq,mm])
                # print("bbbbbbbrrt", qq.requires_grad)
                mm, qq, b = cell(qq, mm)
                buff.append(b)
                # self.vis.argmax_change_image(features1 = mm, features2 = tempm, title="cell number" + str(i))
                
                # self.vis.images(batch,mm)
                # self.vis.question(batch, q)
        mm = mm + v * 0.2
        qq = qq + q * 0.2
        # self.vis.argmax_change_image(features = mm, title="cell end")
        # self.vis.argmax_change_question(features = qq, title="cell end")
        vvv.append(self.classif_module([qq, mm]))
        # all_vvv.append(vvv)
        
        # for i in range(len(batch['index'])): 
        #     self.representations[i]['q_reas'] = qq[i]
        #     self.representations[i]['v_reas'] = mm[i]
            
        

        
        
        
       
       

        if self.buffer is not None: # for visualization
                self.buffer[i] = deepcopy(cell.pairwise.buffer)

        if self.agg['type'] == 'max':
#            for i in range(len(result)):
#                result[i][0] = torch.max(result[i][0], 1)
#                result[i][1] = torch.max(result[i][1], 1)
            mm, mm_argmax = torch.max(mm, 1)
            qq, qq_argmax = torch.max(qq, 1)

        elif self.agg['type'] == 'mean':
            #normalization
            mm = torch.sqrt(F.relu(mm)) - torch.sqrt(F.relu(-mm))
            mm = F.normalize(mm,p=2)  
            mm = mm.view(bsize, -1)
            mm = self.outputm(mm)
            qq = torch.sqrt(F.relu(qq)) - torch.sqrt(F.relu(-qq))
            qq = F.normalize(qq,p=2)  
            qq = qq.view(bsize, -1)
            qq = self.outputq(qq)
#            mm = mm.mean(1)
#            qq = qq.mean(1)

        elif self.agg['type'] == 'sum':
            mm = mm.sum(1)
            qq = qq.sum(1)
#            for i in range(len(result)):
#                result[i][0] = torch.sum(result[i][0], 1)
#                result[i][1] = torch.sum(result[i][1], 1)
        elif self.agg['type'] == 'none':
            
            pass
            


        if 'fusion' in self.classif:

#            qq = self.qqlast(qq)
#            mm = self.mmlast(mm)
            
            logits = self.classif_module([qq, mm])
            
            # for i in range(len(result)):
            #     result[i][0] = self.classif_module([result[i][0], result[i][1]])
                
        elif 'mlp' in self.classif:
            logits = self.classif_module(mm)

        # print(logits.shape)
        # print(logits.shape)
        vis_ques = [self.vis.get_question(batch['question'][i]) for i in range(len(batch['question']))]
        out = {'logits': logits, 'representations': self.representations,
               'v_agg': v,
               'q_agg': q, 
               'v_reas': mm, 
               'q_reas': qq,
               'question': vis_ques, 
               'vvv': vvv,
               'cellq': buff[2][2],
               'cellv': buff[2][0],
               'cell': buff
               }
        
        out = self.process_answers(out)
        
        # out = {'logits': logits, 'representations': self.representations,
        #        'v_agg': v,
        #        'q_agg': q, 
        #        'v_reas': mm, 
        #        'q_reas': qq,
        #        'predicted': anw_vis
        #        }
      
        
        #------------------------------------ visualize
        vis_ids = batch['index']
        vis_imgs = batch['image_name']
        # vis_anws = batch['answer']
        
        # vis_type = batch['question_type']
        # print(q[i])
        # print(qq[i])
        # for i in range(len(batch['index'])): 
        #     self.representations.append({'id': vis_ids[i], 'image_name': vis_imgs[i], 'question':vis_ques[i], 
        #                                'answer': vis_anws[i], 
        #                                "type" : vis_type[i],
        #                                "predicted" : anw_vis['answers'][i],
        #                                'v_agg': v[i].cpu().detach().numpy().tolist(), 
        #                                'q_agg': q[i].cpu().detach().numpy().tolist(), 
        #                                'v_reas': mm[i].cpu().detach().numpy().tolist(), 
        #                                'q_reas': qq[i].cpu().detach().numpy().tolist()})
        # self.vis.multi_image(vis_imgs)
        # self.vis.cosine_similarity_q(qq, q, vis_ques)
        # self.vis.cosine_similarity_m(mm, v)
        # for i in range(len(result)):
#            result[i][0] = {'logits': result[i][0]}
            # result[i][0] = self.process_answers2(result[i][0])
        
        
#        self.Visualize.images(batch)
#        self.Visualize.obj(batch)
        # for i in range(len(result)):
            # self.vis.answers(batch, result[i][0])
        # for i in range (10):
            # self.vis.answers(batch, anw_vis)
#        self.Visualize.question(batch, self.wid_to_word, q_visualize)
        

        return out

    def process_question(self, q, l):
        q_emb = self.txt_enc.embedding(q)
        q_rnn, q_hidden = self.txt_enc.rnn(q_emb) #(bs, 10, 2400)

        return q_rnn
    
    def process_classif(self, q):
        q_classif = q[:,0:4]
        q_emb = self.txt_enc.embedding(q_classif)
        q_classif = q_emb.contiguous().view(q_emb.shape[0], -1)
        # print(q_classif.shape)
        q_classif = self.q_classif_linear(q_classif)
        return q_classif
    
    def process_cls(self,cls_text):
        # self.word_to_wid[''] = 0
        # self.word_to_wid["<unk>"] = 0
        # self.word_to_wid.setdefault('')
        # country_dict.get('Japan', 'Not Found')
        # ids = [torch.transpose(torch.FloatTensor([[self.word_to_wid.get(cls_text[i][j][k], 0) for k in range (len(cls_text[i][j]))]  for j in range (len(cls_text[i])) ]), 0, 1) for i in range (len(cls_text))]
        ids = []
        for i in range (len(cls_text)):
            text = [torch.LongTensor([self.word_to_wid.get(cls_text[i][j][k], 0) for k in range (len(cls_text[i][j]))])  for j in range (len(cls_text[i])) ]
            # print(len(text))
            # print(text[0].shape)
            # print(text[0])
            # text = torch.Tensor(text)
            text = pad_sequence(text, padding_value=0, batch_first=True)#.squeeze(dim=2)
            # print(text.shape)
            text = torch.transpose(text, 0, 1)
            ids.append(text)
        # print(ids)
        # ids = torch.transpose(ids, 0, 1)
        input_cls = pad_sequence(ids, padding_value=0, batch_first=True)#.squeeze(dim=2)
        input_cls = torch.transpose(input_cls, 1, 2).cuda()
        # print(input_cls.shape)
        cls_emb = self.txt_enc.embedding(input_cls)
        return input_cls, cls_emb

    def process_answers2(self, out):
        batch_size = out.shape[0]
        _, pred = out.data.max(1)
        pred.squeeze_()
#        print("---------------------------------------------")
#        print(pred.shape)
#        print(pred[1])
#        print(out['logits'].shape)
#        print(len(self.aid_to_ans))
        out= [self.aid_to_ans[pred[i]] for i in range(batch_size)]
#        for i in range(batch_size):
#            print("--------------")
#            print(pred[i])
#            print(self.aid_to_ans[pred[i]])          
#            out['answers'] = self.aid_to_ans[pred[i]]
        out = [pred[i] for i in range(batch_size)]
        return out


    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        _, pred = out['logits'].data.max(1)
        pred.squeeze_()
#        print("---------------------------------------------")
#        print(pred.shape)
#        print(pred[1])
#        print(out['logits'].shape)
#        print(len(self.aid_to_ans))
        out['answers'] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
#        for i in range(batch_size):
#            print("--------------")
#            print(pred[i])
#            print(self.aid_to_ans[pred[i]])          
#            out['answers'] = self.aid_to_ans[pred[i]]
        out['answer_ids'] = [pred[i] for i in range(batch_size)]
        return out
    
    def get_mask(self, l, n_regions):
        q_mask = torch.zeros((l.shape[0], n_regions), device='cuda')
#        q_mask = torch.cuda.FloatTensor(l.shape[0], n_regions).fill_(0)
        for i in range(l.shape[0]):
            for j in range(l[0]):
                q_mask[i,j] = 1

        return q_mask  
    
    def question_attention(self, q, l):

        if self.self_q_att:
#            q = self.q_linear0(F.dropout(q, p=0.1))
#            q = torch.sqrt(F.relu(q)) - torch.sqrt(F.relu(-q))
#            q = F.normalize(q,p=2)  
            q_att = self.q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = self.q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            #self.q_att_coeffs = q_att
            buffer_whole_q = []
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    # self.vis.question(features=q, seperator="before q_out attention")
                    q_out = q_att*q
                    buffer_whole_q.append(q_out)
                    # self.vis.question(features=q_out, seperator="after q_out attention")
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
                buffer_whole_q = torch.cat(buffer_whole_q, dim=2)
                # self.vis.question(features=buffer_whole_q, seperator="Final q_out")
                buffer_argmax = buffer_whole_q.max(1)[1]
                # self.vis.set_argmax_question(buffer_argmax)
#                q = q_outs[0] + q_outs[1]
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)

#        print("--------------------------", q.shape)
        return q
    
    def question_attention_2(self, q, l, v):
        batch_size = q.size(0)
        n_regions = q.size(1)
        if self.self_q_att_2:
            
            v = v[:,None,:].expand(v.size(0), n_regions, v.size(1))
            alpha = self.fusion_q_att([
                q.contiguous().view(batch_size*n_regions, -1),
                v.contiguous().view(batch_size*n_regions, -1)
            ])
            alpha = alpha.view(batch_size, n_regions, -1)
        
#            q = self.q_linear0(F.dropout(q, p=0.1))
#            q = torch.sqrt(F.relu(q)) - torch.sqrt(F.relu(-q))
#            q = F.normalize(q,p=2)  

            if self.mlp_glimpses > 0:
                alpha = self.linear0_q(alpha)
                alpha = F.relu(alpha)
                alpha = self.linear1_q(alpha)
            
            alpha = mask_softmax(alpha, l)
           
    
            buffer_whole_q = []
            if alpha.size(2) > 1: # nb_glimpses > 1
                alphas = torch.unbind(alpha, dim=2)
                q_outs = []
                for alpha in alphas:
                        
                    alpha = alpha.unsqueeze(2).expand_as(q)
                    # self.vis.images(visual = v, title="before v_out")
                    
                    q_out = alpha*q
                    q_mask = self.get_mask(l, n_regions)
                    q_mask = q_mask[:,:,None].expand(q_out.size(0), n_regions, q_out.size(2))
                    q_out = q_out * q_mask
                    ####################### MMMMAAAAAAASSSSSSSSKKKKKKKKK
                    # mean = alpha.mean(dim=1)
                    # mean = mean.unsqueeze(1)
                    # v_out = v_out.masked_fill(v_out <= mean, 0)
                    # self.vis.images(visual = v_out, title="v_out")
                    buffer_whole_q.append(q_out)
                    
                    # self.vis.showEarlyFusion(visual=v_out, title="average attention before sum")
                    q_out = q_out.sum(1)
                    # v_out = v_out.max(1)[0]
                    q_outs.append(q_out)
                buffer_whole_q = torch.cat(buffer_whole_q, dim=2)
                # self.vis.images(visual = buffer_whole_v, title="Final v_out")
                buffer_argmax = buffer_whole_q.max(1)[1]
    #            self.vis.set_argmax_image(buffer_argmax)
                q_out = torch.cat(q_outs, dim=1)
            else:
                # if mask:
                #     mean = alpha.mean(dim=1)
                #     mean = mean.unsqueeze(1)
                #     alpha = alpha.masked_fill(alpha <= mean, 0)
                alpha = alpha.expand_as(q)
                q_out = alpha*q
                
                # vis.question(features=y_k, seperator="----K")
                # vis.question(features=y_v, seperator="----V")
                q_out = q_out.sum(1)

        return q_out

    
    def image_attention(self, q, v, mask=False):

        
        batch_size = q.size(0)
        n_regions = v.size(1)
        q = q[:,None,:].expand(q.size(0), n_regions, q.size(1))
        alpha = self.fusion([
            q.contiguous().view(batch_size*n_regions, -1),
            v.contiguous().view(batch_size*n_regions, -1)
        ])
        alpha = alpha.view(batch_size, n_regions, -1)
        if self.mlp_glimpses > 0:
            alpha = self.linear0(alpha)
            alpha = F.relu(alpha)
            alpha = self.linear1(alpha)

        alpha = F.softmax(alpha, dim=1)

        buffer_whole_v = []
        if alpha.size(2) > 1: # nb_glimpses > 1
            alphas = torch.unbind(alpha, dim=2)
            v_outs = []
            for alpha in alphas:

                # if mask:
                # # if False:
                # mean,_ = alpha.max(dim=1)
                # mean = mean.unsqueeze(1)
                # alpha = alpha.masked_fill(alpha <= mean, 0)
                
                alpha = alpha.unsqueeze(2).expand_as(v)
                # self.vis.images(visual = v, title="before v_out")
                
                v_out = alpha*v
                ####################### MMMMAAAAAAASSSSSSSSKKKKKKKKK
                # mean = alpha.mean(dim=1)
                # mean = mean.unsqueeze(1)
                # v_out = v_out.masked_fill(v_out <= mean, 0)
                # self.vis.images(visual = v_out, title="v_out")
                buffer_whole_v.append(v_out)
                
                # self.vis.showEarlyFusion(visual=v_out, title="average attention before sum")
                v_out = v_out.sum(1)
                # v_out = v_out.max(1)[0]
                v_outs.append(v_out)
            buffer_whole_v = torch.cat(buffer_whole_v, dim=2)
            # self.vis.images(visual = buffer_whole_v, title="Final v_out")
            buffer_argmax = buffer_whole_v.max(1)[1]
#            self.vis.set_argmax_image(buffer_argmax)
            v_out = torch.cat(v_outs, dim=1)
        else:
            # if mask:
            #     mean = alpha.mean(dim=1)
            #     mean = mean.unsqueeze(1)
            #     alpha = alpha.masked_fill(alpha <= mean, 0)
            alpha = alpha.expand_as(v)
            v_out = alpha*v
            
            # vis.question(features=y_k, seperator="----K")
            # vis.question(features=y_v, seperator="----V")
            v_out = v_out.sum(1)
        return v_out  
    
    def tag_attention(self, q, t, mask=False):

        # t = t[:,:,0,:]
        # print(t.shape)
        batch_size = q.size(0)
        n_regions = t.size(1)
        q = q[:,None,:].expand(q.size(0), n_regions, q.size(1))
        alpha = self.tfusion([
            q.contiguous().view(batch_size*n_regions, -1),
            t.contiguous().view(batch_size*n_regions, -1)
        ])
        alpha = alpha.view(batch_size, n_regions, -1)
        if self.mlp_glimpses > 0:
            alpha = self.tlinear0(alpha)
            alpha = F.relu(alpha)
            alpha = self.tlinear1(alpha)

        alpha = F.softmax(alpha, dim=1)

        buffer_whole_t = []
        if alpha.size(2) > 1: # nb_glimpses > 1
            alphas = torch.unbind(alpha, dim=2)
            t_outs = []
            for alpha in alphas:

                
                alpha = alpha.unsqueeze(2).expand_as(t)
                # self.vis.tag(features = t, seperator="before t_out")
                
                t_out = alpha*t
                ####################### MMMMAAAAAAASSSSSSSSKKKKKKKKK
                # mean = alpha.mean(dim=1)
                # mean = mean.unsqueeze(1)
                # v_out = v_out.masked_fill(v_out <= mean, 0)
                # self.vis.tag(features = t_out, seperator="t_out")
                buffer_whole_t.append(t_out)
                t_out = t_out.sum(1)
                # v_out = v_out.max(1)[0]
                t_outs.append(t_out)
            buffer_whole_t = torch.cat(buffer_whole_t, dim=2)
            # self.vis.tag(features = buffer_whole_t, seperator="Final t_out")
            buffer_argmax = buffer_whole_t.max(1)[1]
            # self.vis.set_argmax_tag(buffer_argmax)
            t_out = torch.cat(t_outs, dim=1)
        else:
            alpha = alpha.expand_as(t)
            t_out = alpha*t
            
            # vis.question(features=y_k, seperator="----K")
            # vis.question(features=y_v, seperator="----V")
            t_out = t_out.sum(1)
        return t_out