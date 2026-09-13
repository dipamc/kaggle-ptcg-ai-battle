// REPLACED ocean.cu: registers the PTCG custom model (the only env we build).
// Original nmmo3/nethack encoders removed (and cudnn dependency with them).
#include "ptcg_model.cu"

static void create_custom_encoder(const std::string& env_name, Encoder* enc) {
    if (env_name != "ptcg") return;
    enc->forward = ptcg_enc_forward;
    enc->backward = ptcg_enc_backward;
    enc->init_weights = ptcg_enc_init_weights;
    enc->reg_params = ptcg_enc_reg_params;
    enc->reg_train = ptcg_enc_reg_train;
    enc->reg_rollout = ptcg_enc_reg_rollout;
    enc->create_weights = ptcg_enc_create_weights;
    enc->free_weights = ptcg_enc_free_weights;
    enc->free_activations = ptcg_enc_free_activations;
    // in_dim = obs (10944); out_dim = fused logits+value = hidden_size (65)
    enc->activation_size = sizeof(PtcgActX);
}

static void create_custom_decoder(const std::string& env_name, Decoder* dec) {
    if (env_name != "ptcg") return;
    dec->forward = ptcg_dec_forward;
    dec->backward = ptcg_dec_backward;
    dec->init_weights = ptcg_dec_init_weights;
    dec->reg_params = ptcg_dec_reg_params;
    dec->reg_train = ptcg_dec_reg_train;
    dec->reg_rollout = ptcg_dec_reg_rollout;
    dec->create_weights = ptcg_dec_create_weights;
    dec->free_weights = ptcg_dec_free_weights;
    dec->free_activations = ptcg_dec_free_activations;
    dec->activation_size = (int)sizeof(PtcgDecActs);
}

static void create_custom_network(const std::string& env_name, Network* net) {
    if (env_name != "ptcg") return;
    net->forward = ptcg_net_forward;
    net->forward_train = ptcg_net_forward_train;
    net->backward = ptcg_net_backward;
    net->init_weights = ptcg_net_init_weights;
    net->reg_params = ptcg_net_reg_params;
    net->reg_train = ptcg_net_reg_train;
    net->reg_rollout = ptcg_net_reg_rollout;
    net->create_weights = ptcg_net_create_weights;
    net->free_weights = ptcg_net_free_weights;
    net->free_activations = ptcg_net_free_activations;
}
