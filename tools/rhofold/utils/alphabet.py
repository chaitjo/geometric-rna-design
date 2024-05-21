
from typing import Sequence, Union
import torch
import string
import itertools
from Bio import SeqIO
from typing import List, Tuple

from tools.rhofold.model.rna_fm.data import Alphabet, get_rna_fm_token, BatchConverter, RawMSA

rna_msaseq_toks = {'toks': ['A','U','G','C','-']}

class RNAAlphabet(Alphabet):

    def get_batch_converter(self):
        if self.use_msa:
            return RNAMSABatchConverter(self)
        else:
            return BatchConverter(self)

    @classmethod
    def from_architecture(cls, name: str, ) -> "RNAAlphabet":
        if name in ("RNA MSA Transformer", "rna_msa_transformer", "RNA"):
            standard_toks = rna_msaseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
            use_msa = True
        else:
            raise ValueError("Unknown architecture selected")
        return cls(
            standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa
        )

class RNAMSABatchConverter(BatchConverter):

    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            # replace T with U
            msa = [ [dec, nastr.replace('T', 'U')] for dec, nastr in msa]

            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, :msa_tokens.size(0), :msa_tokens.size(1)] = msa_tokens

        return labels, strs, tokens

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
deletekeys[" "] = None # add remove
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description,
           remove_insertions(str(record.seq).replace('T', 'U'))) for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def read_fas(filename: str):
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description,
             remove_insertions(str(record.seq).replace('T', 'U'))) for record in itertools.islice(SeqIO.parse(filename, "fasta"), 1)]

def get_msa_feature(msa_path,
                    msa_depth,
                    batch_converter = RNAAlphabet.from_architecture('RNA').get_batch_converter()):

    msa_data = [read_msa(msa_path, msa_depth)]

    _, _, msa_batch_tokens = batch_converter(msa_data)

    # remove [cls] token in msa_batch_tokens
    fea1d = msa_batch_tokens.squeeze(0).data.cpu().numpy().transpose((1, 0))[1:, :]

    return torch.LongTensor(fea1d[:, :].transpose((1, 0)))

def get_features(fas_fpath, msa_fpath, msa_depth = 128):
    '''
    Get features from MSA
    '''

    seq = read_fas(fas_fpath)[0][1]

    msa_tokens = get_msa_feature(msa_path=msa_fpath, msa_depth=msa_depth)

    rna_fm_tokens = get_rna_fm_token(fas_fpath)

    return {
        'seq': seq,
        'tokens': msa_tokens.unsqueeze(0),
        'rna_fm_tokens': rna_fm_tokens.unsqueeze(0),
    }

