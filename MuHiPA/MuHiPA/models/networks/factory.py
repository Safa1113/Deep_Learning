import sys
import copy
import torch
import torch.nn as nn
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from MuHiPA.datasets.block.models.networks.vqa_net import VQANet as AttentionNet
from .muhipa import MuHiPA
from .emurelpa_vizwiz import EMuRelPA_VizWiz
from .murel_net import MuRelNet
from .vrd_net import VRDNet
from .vrd_net_block import VRDNetBlock
from .dfaf import DFAF

def factory(engine):
    mode = list(engine.dataset.keys())[0]
    dataset = engine.dataset[mode]
    opt = Options()['model.network']

    if opt['name'] == 'MuHiPA':
        net = MuHiPA(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            self_v_att=opt['self_v_att'],
            # self_t_att=opt['self_t_att'],
            n_step=opt['n_step'],
            shared=opt['shared'],
            cell=opt['cell'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid)

    elif opt['name'] == 'EMuRelPA_VizWiz':
        net = EMuRelPA_VizWiz(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            self_v_att=opt['self_v_att'],
            # self_t_att=opt['self_t_att'],
            n_step=opt['n_step'],
            shared=opt['shared'],
            cell=opt['cell'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid)
        
    elif opt['name'] == 'murel_net':
        net = MuRelNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            n_step=opt['n_step'],
            shared=opt['shared'],
            cell=opt['cell'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid)

    elif opt['name'] == 'vrd_net':
        net = VRDNet(opt)
        
    elif opt['name'] == 'vrd_net_block':
        net = VRDNetBlock(opt)        
        
    elif opt['name'] == 'DFAF':
        # print(opt['txt_enc'])
        # print(opt['output_size'])
        net = DFAF(
            txt_enc=opt['txt_enc'],
            output_size=opt['output_size'],
            output_features=opt['output_features'],
            question_features=opt['question_features'],
            vision_features=opt['vision_features'],
            hidden_features=opt['hidden_features'],
            num_inter_head=opt['num_inter_head'],
            num_intra_head=opt['num_intra_head'],
            num_block=opt['num_block'],
            visual_normalization=opt['visual_normalization'],
            max_answers=opt['max_answers'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            t_emb = opt['t_emb']
            )
    else:
        raise ValueError(opt['name'])

    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net


if __name__ == '__main__':
    factory()
