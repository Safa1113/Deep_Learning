from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import MuHiPA.datasets.block as block
from .pairwise import Pairwise
from torch.nn.utils.weight_norm import weight_norm
from MuHiPA.datasets.block.models.networks.vqa_net import mask_softmax
import math
from bootstrap.lib.logger import Logger
import time 




class MuHiPAReasoning(nn.Module):

    def __init__(self,
            residual=False,
            fusion={},
            pairwiseMQ_m = {},
            pairwiseMQ_q = {},
            fusion_module_v = {},
            fusion_module_v2 = {},
            fusion_module_t = {},
            fusion_module_tv = {},
            fusion_module_q = {},
            q_attention = False,
            pairwise={},
            coAttention_Q={},
            coAttention_M={},
            selfAttention_Q = {},
            selfAttention_M = {}):
        super(MuHiPAReasoning, self).__init__()
        self.residual = residual
        self.fusion = fusion
        self.coAttention_Q = coAttention_Q
        self.coAttention_M = coAttention_M
        self.selfAttention_Q = selfAttention_Q
        self.selfAttention_M = selfAttention_M
        self.pairwiseMQ_m = pairwiseMQ_m
        self.pairwiseMQ_q = pairwiseMQ_q
        self.fusion_module_v = fusion_module_v
        self.fusion_module_v2 = fusion_module_v2
        self.fusion_module_t = fusion_module_t
        self.fusion_module_tv = fusion_module_tv
        self.fusion_module_q = fusion_module_q
        self.pairwise = pairwise
        
        # self.feedforwardQ = nn.Linear(4800, 4800)
        # self.feedforwardM = nn.Linear(4096, 4096)
        
        # self.feedforwardQM = nn.Linear(4800, 4096)
        # self.feedforwardMQ = nn.Linear(4096, 4800)
        
        if self.coAttention_Q:
            self.coAttention_Q = CoAttention(**coAttention_Q)
        if self.coAttention_M:
            self.coAttention_M = CoAttention(**coAttention_M)
            
        if self.selfAttention_Q:
            self.selfAttention_Q = SelfAttention(**selfAttention_Q)
        if self.selfAttention_M:
            self.selfAttention_M = SelfAttention(**selfAttention_M)

 
        if self.pairwiseMQ_m:
            self.pairwiseMQ_m = PairwiseMQ(**pairwiseMQ_m)
        if self.pairwiseMQ_q:
            self.pairwiseMQ_q = PairwiseMQ(**pairwiseMQ_q)
            self.update_q_with_new_m  = pairwiseMQ_q['new_m']
            
        if self.fusion_module_v:
            self.v_conv = nn.Conv1d(1, 10, 5, stride=2, padding=2)
            self.v_att_linear0 = nn.Linear(2048, 512)  #(1024,512) #(2048, 512) 
            self.v_att_linear1 = nn.Linear(512, 2)
            self.fusion_module_v = block.factory_fusion(self.fusion_module_v)
            self.fusion_module_v2 = block.factory_fusion(self.fusion_module_v2)
                
                # {'type': 'block', 
                #                                           'input_dims': [6576, 6576], #[4800, 2048],5336  6576 4716
                #                                           'output_dim': 6576,
                #                                           'mm_dim': 1600,
                #                                           'chunks': 20,
                #                                           'rank': 10,
                #                                           'dropout_input': 0.1,
                #                                           'dropout_pre_lin': 0.})
        if self.fusion_module_t:
            self.fusion_module_t = block.factory_fusion(self.fusion_module_t)
        if self.fusion_module_tv:
            self.fusion_module_tv = block.factory_fusion(self.fusion_module_tv)

        if self.fusion_module_q:
            self.fusion_module_q = block.factory_fusion(self.fusion_module_q)
            self.fusion_module_q2 = block.factory_fusion({'type': 'block', 
                                                          'input_dims': [4800, 4800], #[4800, 2048], 5336
                                                          'output_dim': 4800,
                                                          'mm_dim': 1600,
                                                          'chunks': 20,
                                                          'rank': 10,
                                                          'dropout_input': 0.1,
                                                          'dropout_pre_lin': 0.})
        self.q_attention = q_attention
        if q_attention:
#            self.linear = 
            
            self.q_attention = q_attention
            self.q_att_linear0 = nn.Linear(2400, 4096)
            
            
            # self.q_att_linear1 = nn.Linear(512, 1)
            # mlp_glimpses = 2
            # self.mlp_glimpses = mlp_glimpses
            # self.fusion = block.factory_fusion(self.fusion)
            # self.linear0 = nn.Linear(1, 512)
            # self.linear1 = nn.Linear(512, mlp_glimpses)
        
        if self.pairwise:
            self.pairwise = Pairwise(**pairwise)
            
        Logger().log_value('nparams_vfusion',
            self.get_nparams_vfusion(),
            should_print=True)
        Logger().log_value('nparams_qfusion',
            self.get_nparams_qfusion(),
            should_print=True)
        
        Logger().log_value('nparams of big cell',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)
        
        
    def get_nparams_vfusion(self):
        params = []
        if self.fusion_module_v:
            params += [p.numel() for p in self.fusion_module_v.parameters() if p.requires_grad]
            params += [p.numel() for p in self.fusion_module_v2.parameters() if p.requires_grad]
        return sum(params)
    
    def get_nparams_qfusion(self):
        params = []
        if self.fusion_module_q:
            params += [p.numel() for p in self.fusion_module_q.parameters() if p.requires_grad]
        return sum(params)    

    def forward(self, qq, mm):
        
        #bsize = mm.shape[0]
        #n_regions = mm.shape[1]
        mm_new = mm
        qq_new = qq
        # tt_new = tt


       




       
#         if self.coAttention_M:
#             # print("-----------------------------------------------")
#             # print(qq_mask.shape)
#             # print(qq_new.shape)
            
#             mm_new = self.coAttention_M(mm_new, qq_new, qq_mask, True, vis)
#             # mm_new = self.coAttention_M(mm_new, qq_new, q2=True)
            
#         if self.selfAttention_M:
#             mm_new = self.selfAttention_M(mm_new)  
            
#         if self.pairwiseMQ_m:
#             mm_new, _ = self.pairwiseMQ_m(mm_new, qq_new, update_mm = True)
        
#         # if self.selfAttention_M:
#         #     qq_mask = self.selfAttention_M(qq_mask)    
        
#         if self.coAttention_Q:    
# #            if self.update_q_with_new_m:
#             qq_new = self.coAttention_Q(qq_new, mm_new, qq_mask, False, vis)
#             # qq_new = self.coAttention_Q(qq_new, mm_new, q2=False)
# #            else:
# #                qq_new = self.coAttention_Q(qq_new, mm)
 
        buff = []
      
        
        # if self.selfAttention_Q:
        #     qq_new = self.selfAttention_Q(qq_new,qq_mask, True)        
        

            
            
        # if self.pairwiseMQ_q:
        #     if self.update_q_with_new_m:
        #         _, qq_new = self.pairwiseMQ_q(mm_new, qq_new, update_qq = True, vis=vis)
        #         qq_new = qq * qq_new
        #     else:
        #         _, qq_new = self.pairwiseMQ_q(mm, qq_new, update_qq = True, vis=vis)
        #         qq_new = qq * qq_new
        
       
        # if self.fusion_module_t:
        #     tt_new = self.fusion_module_t([qq_new, tt_new])
        # if self.fusion_module_tv:
        #     tt_new = self.fusion_module_tv([mm_new, tt_new])
       
        if self.fusion_module_v:
            # start_time = time.time() 
            # vis.density_plot(mm_new[0], title="before mm_new")
            # vis.statics(mm_new, title="before mm_new")
            # vis.argmax_change_image(features = mm_new*2, title="before mm_new self fusion")
            
            # print("mm_newmm_newmm_newmm_newmm_newmm_newmm_newmm_newmm_newmm_new", mm_new.shape)
            
            mm_new = self.fusion_module_v2([mm_new, mm_new])
            # vis.argmax_change_image(features = mm_new*2, title="after mm_new self fusion")
            buff.append(mm_new)
            # vis.density_plot(mm_new[0], title=" mm_new")
            # vis.statics(mm_new, title="mm_new")

            mm_new = self.fusion_module_v([qq_new, mm_new])
            # mm_new = F.normalize(mm_new, p=2.0, dim=1, eps=1e-12, out=None)
            buff.append(mm_new)
            # vis.argmax_change_image(features = mm_new*2, title="after mm_new fusion with question")
            # vis.density_plot(mm_new[0], title="after interaction with question mm_new")
            # vis.statics(mm_new, title="after interaction with question mm_new")
            # print( "- Time- %s seconds ---" % (time.time() - start_time))
            

            # input = mm_new.unsqueeze(1)
            # output = self.v_conv(input)
            
            # self.image_attention(qq_new, output)
            
            # # print(output.shape)
            # v_att = self.v_att_linear0(output)
            # v_att = F.relu(v_att)
            # v_att = self.v_att_linear1(v_att)
            # v_att = F.softmax(v_att, dim=1)
            # v_atts = torch.unbind(v_att, dim=2)
            # v_outs = []
            # for v_att in v_atts:
            #   v_att = v_att.unsqueeze(2)
            #   v_att = v_att.expand_as(output)
            #   v_out = v_att*output
            #   v_out = v_out.sum(1)
            #   v_outs.append(v_out)
            # mm_new = torch.cat(v_outs, dim=1)
            # vis.argmax_change_image(features = mm_new, title="after mm_new convolution and attention")
            
            # vis.density_plot(mm_new[0], title="after new 1 conv attention")
            # vis.statics(mm_new, title="after new 1 conv attention")

        
        if self.fusion_module_q:
#            qq_new = self.fusion_module_q2([qq_new, qq_new])
            # vis.density_plot(qq_new[0], title="before qq_new")
            qq_new = self.fusion_module_q([qq_new, mm_new])
            buff.append(qq_new)
            pass
            
        if self.q_attention:
            
#            qq_new = F.softmax(qq_new, dim=1)
#            qq_new = qq_new.squeeze()
#            qq_new = self.question_attention(qq, l)
#            mm_new = self.image_attention(qq_new,mm)
#            qq_new = qq_new.unsqueeze(dim = 1).expand_as(qq)
#            m, _ = mm_new.max(1) #test33333333333333
#            qq_new = m.unsqueeze(1).expand_as(qq_new) * qq_new  #test3333333
#            qq_new = self.fusion_module_q2([qq_new, qq]) #test4
            # vis.argmax_change_question(features = qq, title="before qq_new")
            qq_new = qq * qq_new
            # qq_new = F.normalize(qq_new, p=2.0, dim=1, eps=1e-12, out=None)
            buff.append(qq_new)
            # vis.argmax_change_question(features = qq_new, title="after qq_new")
            # vis.density_plot(qq_new[0], title="after qq_new")
#            mm_new = mm_new.unsqueeze(dim = 1).expand_as(mm)
#            mm_new = mm * mm_new
            pass
        
        
        
        
        # if self.pairwise:
        #     mm_new = self.pairwise(mm)
     

        #FeedForward and Layer Normalization
#        qq_new = self.feedforwardQ(qq_new)
        # mm_new = self.feedforwardM(F.dropout(mm_new,0.1))
        # m, _ = mm.max(1)
        # m = m.unsqueeze(1).expand_as(mm_new)
        # mm_new = m * mm_new
        
#        qq_new = F.normalize(qq_new,p=2)  
        # mm_new = F.normalize(mm_new,p=2)  
        
#        print("--------------------------------------")
#        print(mm_new.shape)
#        print(qq_new.shape)
        
        if self.residual:
            mm_new = mm_new + mm 
            buff.append(mm_new)
            # vis.argmax_change_image(features = mm_new, title="final image")
            # vis.density_plot(mm_new[0], title="Final mm_new")
            # vis.statics(mm_new, title="Final mm_new")
            qq_new = qq_new + qq 
            buff.append(qq_new)
            # tt_new = tt_new + tt 
            # qq_mask = qq_mask + qq_mask

        return mm_new, qq_new, buff#tt_new#, qq_mask
        # return mm_new, mm_new
    
    def question_attention(self, q, l):

       
#            q = self.q_linear0(F.dropout(q, p=0.1))
#            q = torch.sqrt(F.relu(q)) - torch.sqrt(F.relu(-q))
#            q = F.normalize(q,p=2)  
        q_att = self.q_att_linear0(q)
        q_att = F.relu(q_att)
        q_att = self.q_att_linear1(q_att)
        q_att = mask_softmax(q_att, l)
        #self.q_att_coeffs = q_att
        if q_att.size(2) > 1:
            q_atts = torch.unbind(q_att, dim=2)
            q_outs = []
            for q_att in q_atts:
                q_att = q_att.unsqueeze(2)
                q_att = q_att.expand_as(q)
                q_out = q_att*q
                q_out = q_out.sum(1)
                q_outs.append(q_out)
            q = torch.cat(q_outs, dim=1)
#                q = q_outs[0] + q_outs[1]
        else:
            q_att = q_att.expand_as(q)
            q = q_att * q
            q = q.sum(1)

#        print("--------------------------", q.shape)
        return q
    
    
    def image_attention(self, q, v):

        
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
        
        if alpha.size(2) > 1: # nb_glimpses > 1
            alphas = torch.unbind(alpha, dim=2)
            v_outs = []
            for alpha in alphas:
                alpha = alpha.unsqueeze(2).expand_as(v)
                v_out = alpha*v
                v_out = v_out.sum(1)
                v_outs.append(v_out)
            v_out = torch.cat(v_outs, dim=1)
        else:
            alpha = alpha.expand_as(v)
            v_out = alpha*v
            v_out = v_out.sum(1)
        return v_out    

    
class PairwiseMQ(nn.Module):

    def __init__(self,
            residual=True,
            fusion_feat={},
            linear_output={},
            new_m={},
            agg={}):
        super(PairwiseMQ, self).__init__()
        self.residual = residual
        self.fusion_feat = fusion_feat
        self.linear_output = linear_output
        self.linear = nn.Linear(self.fusion_feat['output_dim'] * self.linear_output['no_of_regions'], self.linear_output['output_dim'])
        self.agg = agg
        if self.fusion_feat:
            self.f_feat_module = block.factory_fusion(self.fusion_feat)

        self.buffer = None

    def set_buffer(self):
        self.buffer = {}

    def forward(self, mm, qq, coords = None, update_mm=False, update_qq=False, vis=None):
        bsize = mm.shape[0]
        nregion = mm.shape[1]
        
#        qq_mask = q_mask[:,:,None].expand(q_mask.shape[0], q_mask.shape[1], qq.shape[-1])
#        qq = qq * qq_mask

        Rij = 0

            
    
        mmm = mm[:,:,None,:]
        mmm = mmm.expand(bsize,nregion,nregion,mm.shape[-1])
        mmm = mmm.contiguous()
        mmm = mmm.view(bsize*nregion*nregion,mm.shape[-1])
        qqq = qq[:,None,:,:]
        qqq = qqq.expand(bsize,nregion,nregion,qq.shape[-1])
        qqq = qqq.contiguous()
        qqq = qqq.view(bsize*nregion*nregion,qq.shape[-1])
        Rij += self.f_feat_module([qqq, mmm])


#        qq_mask = q_mask[:,:,None].expand(q_mask.shape[0], q_mask.shape[1], Rij.shape[-1])
#        qq_mask = qq_mask[:,None,:,:].expand(bsize,nregion,nregion, Rij.shape[-1])
        Rij = Rij.view(bsize,nregion,nregion,-1)
#        Rij *= qq_mask
       
        qq_new = None
        mm_new = None

        if self.agg['type'] == 'max':
            
            if update_mm:
                Rij = Rij.view(bsize,nregion,-1)
                mm_new = self.linear(Rij)
#                mm_new, argmax = Rij.max(2)
            if update_qq:
                Rij = torch.transpose(Rij, 1, 2).contiguous()
                Rij = Rij.view(bsize,nregion,-1)
                qq_new = self.linear(Rij)
                vis.question(features=qq_new,seperator="---- PairwiseMQ q")  
#                qq_mask = q_mask[:,:,None].expand(q_mask.shape[0], q_mask.shape[1], qq_new.shape[-1])
#                qq_new *= qq_mask
#                qq_new, argmax = Rij.max(1)
                
                
        else:
            if update_mm:
                mm_new = getattr(Rij, self.agg['type'])(2)
            if update_qq:
                mm_new = getattr(Rij, self.agg['type'])(1)

        if self.buffer is not None:
            self.buffer['mm'] = mm.data.cpu() # bx36x2048
            self.buffer['qq'] = qq.data.cpu() # bx36x2048
            self.buffer['mm_new'] = mm.data.cpu() # bx36x2048
            self.buffer['qq_new'] = qq.data.cpu() # bx36x2048
            self.buffer['argmax'] = argmax.data.cpu() # bx36x2048
            L1_regions = torch.norm(mm_new.data, 1, 2) # bx36
            L2_regions = torch.norm(mm_new.data, 2, 2) # bx36
            self.buffer['L1_max'] = L1_regions.max(1)[0].cpu() # b
            self.buffer['L2_max'] = L2_regions.max(1)[0].cpu() # b

        if self.residual: 
            if update_mm:
                mm_new += mm
            if update_qq:
                qq_new += qq

        return mm_new, qq_new
    
#    def image_attention(self, alpha, v):
#
#     
#        alpha = alpha.view(batch_size, n_regions, -1)
#        if self.mlp_glimpses > 0:
#            alpha = self.linear0(alpha)
#            alpha = F.relu(alpha)
#            alpha = self.linear1(alpha)
#
#        alpha = F.softmax(alpha, dim=1)
#        
#        if alpha.size(2) > 1: # nb_glimpses > 1
#            alphas = torch.unbind(alpha, dim=2)
#            v_outs = []
#            for alpha in alphas:
#                alpha = alpha.unsqueeze(2).expand_as(v)
#                v_out = alpha*v
#                v_out = v_out.sum(1)
#                v_outs.append(v_out)
#            v_out = torch.cat(v_outs, dim=1)
#        else:
#            alpha = alpha.expand_as(v)
#            v_out = alpha*v
#            v_out = v_out.sum(1)
#        return v_out    
