from module import HeterGConv_Edge, HeterGConvLayer, SenShift_Feat
import torch.nn as nn
import torch
from utils import batch_to_all_tva
from commonsense_model import CommonsenseRNN

class GraphSmile_COSMIC(nn.Module):
    def __init__(self, args, embedding_dims, n_classes_emo):
        super(GraphSmile_COSMIC, self).__init__()
        self.textf_mode = args.textf_mode
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.modals = args.modals
        self.shift_win = args.shift_win
        self.role_emb = nn.Embedding(2, args.hidden_dim)  # 2 roles: speaker (1), listener (0)

        self.comet_encoder = CommonsenseRNN(
            D_m=384,      # GRU memory dim
            D_s=768,      # RoBERTa input
            D_g=150,      # GRU hidden
            D_p=150,
            D_r=150,
            D_i=150,
            D_e=384,      # output of COMET encoder
            listener_state=True,
            context_attention='simple',
            D_a=100,
            dropout=args.drop,
            emo_gru=True,
        )


        self.dim_layer_v = nn.Sequential(
            nn.Linear(embedding_dims[1], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )
        self.dim_layer_a = nn.Sequential(
            nn.Linear(embedding_dims[2], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )

        self.text_encoder_roberta = nn.Sequential(
            nn.Linear(4 * embedding_dims[0], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop)
        )

        self.text_gate = nn.Sequential(
            nn.Linear(2 * args.hidden_dim, args.hidden_dim),
            nn.Sigmoid()
        )

        self.hetergconv_tv = HeterGConv_Edge(
            args.hidden_dim,
            HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda),
            args.heter_n_layers[0],
            args.drop,
            args.no_cuda
        )
        self.hetergconv_ta = HeterGConv_Edge(
            args.hidden_dim,
            HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda),
            args.heter_n_layers[1],
            args.drop,
            args.no_cuda
        )
        self.hetergconv_va = HeterGConv_Edge(
            args.hidden_dim,
            HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda),
            args.heter_n_layers[2],
            args.drop,
            args.no_cuda
        )

        self.modal_fusion = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LeakyReLU(),
        )

        self.emo_output = nn.Linear(args.hidden_dim, n_classes_emo)
        self.sen_output = nn.Linear(args.hidden_dim, 3)
        self.senshift = SenShift_Feat(args.hidden_dim, args.drop, args.shift_win)

    def forward(self, feature_t0, feature_t1, feature_t2, feature_t3,
                feature_c0, feature_c1, feature_c2, feature_c3, feature_c4,
                feature_c5, feature_c6, feature_c7, feature_c8,
                feature_v, feature_a, umask, qmask, dia_lengths):

        roberta_concat = torch.cat([feature_t0, feature_t1, feature_t2, feature_t3], dim=-1)
        feat_roberta = self.text_encoder_roberta(roberta_concat)

        # Derive role index: 1 if speaker, 0 if listener
        speaker_mask = qmask.argmax(dim=-1).long()  # [seq_len, batch_size]
        seq_len, batch_size = speaker_mask.size()
        flat_roles = speaker_mask.view(-1)
        role_embeddings = self.role_emb(flat_roles)

        flat_roberta = feat_roberta.view(-1, feat_roberta.size(-1))
        flat_roberta += role_embeddings
        feat_roberta = flat_roberta.view(seq_len, batch_size, -1)

        # COMET features before encoder
        xIntent = feature_c0
        xAttr   = feature_c1
        xNeed   = feature_c2
        xWant   = feature_c3
        xEffect = feature_c4
        xReact  = feature_c5
        oWant   = feature_c6
        oEffect = feature_c7
        oReact  = feature_c8

        comet_encoded, _ = self.comet_encoder(feat_roberta, xEffect, xReact, xIntent, oEffect, oReact, qmask)

        gate_input = torch.cat([feat_roberta, comet_encoded], dim=-1)
        gate = self.text_gate(gate_input)
        feat_text = gate * feat_roberta + (1 - gate) * comet_encoded

        feat_v = self.dim_layer_v(feature_v)
        feat_a = self.dim_layer_a(feature_a)

        emo_t, emo_v, emo_a = feat_text, feat_v, feat_a
        emo_t, emo_v, emo_a = batch_to_all_tva(emo_t, emo_v, emo_a, dia_lengths, self.no_cuda)

        featheter_tv, heter_edge_index = self.hetergconv_tv((emo_t, emo_v), dia_lengths, self.win_p, self.win_f)
        featheter_ta, heter_edge_index = self.hetergconv_ta((emo_t, emo_a), dia_lengths, self.win_p, self.win_f, heter_edge_index)
        featheter_va, heter_edge_index = self.hetergconv_va((emo_v, emo_a), dia_lengths, self.win_p, self.win_f, heter_edge_index)

        feat_fusion = (self.modal_fusion(featheter_tv[0]) + self.modal_fusion(featheter_ta[0]) +
                       self.modal_fusion(featheter_tv[1]) + self.modal_fusion(featheter_va[0]) +
                       self.modal_fusion(featheter_ta[1]) + self.modal_fusion(featheter_va[1])) / 6

        logit_emo = self.emo_output(feat_fusion)
        logit_sen = self.sen_output(feat_fusion)
        logit_shift = self.senshift(feat_fusion, feat_fusion, dia_lengths)

        return logit_emo, logit_sen, logit_shift, feat_fusion