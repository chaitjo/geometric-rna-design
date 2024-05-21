import tools.rhofold.model.rna_fm as rna_esm
from argparse import Namespace

rna_fm_args = {
    'arch': 'roberta_large',
    'layers': 12,
    'embed_dim': 640,
    'ffn_embed_dim': 5120,
    'attention_heads': 20,
    'max_positions': 1024,
    'sample_break_mode': 'eos',
    'tokens_per_sample': 1023,
    'mask_prob': 0.15,
    'pad': 1, 'eos': 2, 'unk': 3, 'dropout': 0.1,
    'no_seed_provided': False,
    '_name': 'ESM-1b'
}

def load_esm1b_rna_t12(theme="protein"):

    alphabet = rna_esm.Alphabet.from_architecture('roberta_large', theme=theme)
    model_type = rna_esm.ProteinBertModel
    model = model_type(
        Namespace(**rna_fm_args), alphabet,
    )
    return model, alphabet

def esm1b_rna_t12():
    return load_esm1b_rna_t12(theme="rna")
