import os
import copy
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_matthews_corrcoef

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd

from src.data.data_utils import pdb_to_tensor, get_c4p_coords
from src.data.sec_struct_utils import (
    predict_sec_struct,
    dotbracket_to_paired,
    dotbracket_to_adjacency
)
from src.constants import (
    NUM_TO_LETTER, 
    PROJECT_PATH,
    RMSD_THRESHOLD,
    TM_THRESHOLD,
    GDT_THRESHOLD
)


def evaluate(
        model, 
        dataset, 
        n_samples, 
        temperature, 
        device, 
        model_name="eval",
        metrics=[
            'recovery', 'perplexity', 'sc_score_eternafold', 
            'sc_score_ribonanzanet', 'sc_score_rhofold'
        ],
        save_designs=False
    ):
    """
    Run evaluation suite for trained RNA inverse folding model on a dataset.

    The following metrics can be computed along with metadata per sample per residue:
    1. (recovery) Sequence recovery per residue (taking mean gives per sample recovery)
    2. (perplexity) Perplexity per sample
    3. (sc_score_eternafold) Secondary structure self-consistency score per sample, 
        using EternaFold for secondary structure prediction and computing MCC between
        the predicted and groundtruth 2D structures as adjacency matrices.
    4. (sc_score_ribonanzanet) Chemical modification self-consistency score per sample,
        using RibonanzaNet for chemical modification prediction of the groundtruth and
        designed sequences, and measuring MAE between them.
    5. (sc_score_rhofold) Tertiary structure self-consistency scores per sample,
        using RhoFold for tertiary structure prediction and measuring RMSD, TM-score,
        and GDT_TS between the predicted and groundtruth C4' 3D coordinates.
    6. (rmsd_within_thresh) Percentage of samples with RMSD within threshold (<=2.0A)
    7. (tm_within_thresh) Percentage of samples with TM-score within threshold (>=0.45)
    8. (gddt_within_thresh) Percentage of samples with GDT_TS within threshold (>=0.50)

    Args:
        model: trained RNA inverse folding model
        dataset: dataset to evaluate on
        n_samples: number of predicted samples/sequences per data point 
        temperature: sampling temperature
        device: device to run evaluation on
        model_name: name of model/dataset for plotting (default: 'eval')
        metrics: list of metrics to compute
        save_designs: whether to save designs as fasta with metrics
    
    Returns: Dictionary with the following keys:
        df: DataFrame with metrics and metadata per residue per sample for analysis and plotting
        samples_list: list of tensors of shape (n_samples, seq_len) per data point 
        recovery_list: list of mean recovery per data point
        perplexity_list: list of mean perplexity per data point
        sc_score_eternafold_list: list of 2D self-consistency scores per data point
        sc_score_ribonanzanet_list: list of 1D self-consistency scores per data point
        sc_score_rmsd_list: list of 3D self-consistency RMSDs per data point
        sc_score_tm_list: list of 3D self-consistency TM-scores per data point
        sc_score_gddt_list: list of 3D self-consistency GDTs per data point
        rmsd_within_thresh_list: list of % scRMSDs within threshold per data point
        tm_within_thresh_list: list of % scTMs within threshold per data point
        gddt_within_thresh_list: list of % scGDDTs within threshold per data point
    """
    assert 'recovery' in metrics, 'Sequence recovery must be computed for evaluation'

    #######################################################################
    # Optionally initialise other models used for self-consistency scoring
    #######################################################################

    if 'sc_score_ribonanzanet' in metrics:
        from tools.ribonanzanet.network import RibonanzaNet
        
        # Initialise RibonanzaNet for self-consistency score
        ribonanza_net = RibonanzaNet(
            os.path.join(PROJECT_PATH, 'tools/ribonanzanet/config.yaml'),
            os.path.join(PROJECT_PATH, 'tools/ribonanzanet/ribonanzanet.pt'),
            device
        )
        # Transfer model to device in eval mode
        ribonanza_net = ribonanza_net.to(device)
        ribonanza_net.eval()
    
    if 'sc_score_rhofold' in metrics:
        from tools.rhofold.rf import RhoFold
        from tools.rhofold.config import rhofold_config
        
        # Initialise RhoFold for 3D self-consistency score
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
        print(f"Loading RhoFold checkpoint: {rhofold_path}")
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device('cpu'))['model'])
        # Transfer model to device in eval mode
        rhofold = rhofold.to(device)
        rhofold.eval()
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    ####################################################
    # Evaluation loop over each data point sequentially
    ####################################################

    # per sample metric lists for storing evaluation results
    samples_list = []               # list of tensors of shape (n_samples, seq_len) per data point 
    recovery_list = []              # list of mean recovery per data point
    perplexity_list = []            # list of mean perplexity per data point
    sc_score_ribonanzanet_list = [] # list of 1D self-consistency scores per data point
    sc_score_eternafold_list = []   # list of 2D self-consistency scores per data point
    sc_score_rmsd_list = []         # list of 3D self-consistency RMSDs per data point
    rmsd_within_thresh_list = []    # list of % scRMSDs within threshold per data point
    sc_score_tm_list = []           # list of 3D self-consistency TM-scores per data point
    tm_within_thresh_list = []      # list of % scTMs within threshold per data point
    sc_score_gddt_list = []         # list of 3D self-consistency GDTs per data point
    gddt_within_thresh_list = []    # list of % scGDDTs within threshold per data point

    # DataFrame to store metrics and metadata per residue per sample for analysis and plotting
    df = pd.DataFrame(columns=['idx', 'recovery', 'sasa', 'paired', 'rmsds', 'model_name'])

    model.eval()
    if device.type == 'xpu':
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model)
        if 'sc_score_ribonanzanet' in metrics:
            ribonanza_net = ipex.optimize(ribonanza_net)
        if 'sc_score_rhofold' in metrics:
            rhofold = ipex.optimize(rhofold)
    
    with torch.no_grad():
        for idx, raw_data in tqdm(
            enumerate(dataset.data_list),
            total=len(dataset.data_list)
        ):
            # featurise raw data
            data = dataset.featurizer(raw_data).to(device)

            # sample n_samples from model for single data point: n_samples x seq_len
            samples, logits = model.sample(data, n_samples, temperature, return_logits=True)
            samples_list.append(samples.cpu().numpy())
            
            # perplexity per sample: n_samples x 1
            n_nodes = logits.shape[1]
            perplexity = torch.exp(F.cross_entropy(
                logits.view(n_samples * n_nodes, model.out_dim), 
                samples.view(n_samples * n_nodes).long(), 
                reduction="none"
            ).view(n_samples, n_nodes).mean(dim=1)).cpu().numpy()
            perplexity_list.append(perplexity.mean())

            ###########
            # Metadata
            ###########

            # per residue average SASA: seq_len x 1
            mask_coords = data.mask_coords.cpu().numpy()
            sasa = np.mean(raw_data['sasa_list'], axis=0)[mask_coords]

            # per residue indicator for paired/unpaired: seq_len x 1
            paired = np.mean(
                [dotbracket_to_paired(sec_struct) for sec_struct in raw_data['sec_struct_list']], axis=0
            )[mask_coords]

            # per residue average RMSD: seq_len x 1
            if len(raw_data["coords_list"]) == 1:
                rmsds = np.zeros_like(sasa)
            else:
                rmsds = []
                for i in range(len(raw_data["coords_list"])):
                    for j in range(i+1, len(raw_data["coords_list"])):
                        coords_i = get_c4p_coords(raw_data["coords_list"][i])
                        coords_j = get_c4p_coords(raw_data["coords_list"][j])
                        rmsds.append(torch.sqrt(torch.sum((coords_i - coords_j)**2, dim=1)).cpu().numpy())
                rmsds = np.stack(rmsds).mean(axis=0)[mask_coords]

            ##########
            # Metrics
            ##########

            # sequence recovery per residue across all samples: n_samples x seq_len 
            recovery = samples.eq(data.seq).float().cpu().numpy()
            recovery_list.append(recovery.mean())

            # update per residue per sample dataframe
            df = pd.concat([
                df, 
                pd.DataFrame({
                    'idx': [idx] * len(recovery.mean(axis=0)),
                    'recovery': recovery.mean(axis=0),
                    'sasa': sasa,
                    'paired': paired,
                    'rmsds': rmsds,
                    'model_name': [model_name] * len(recovery.mean(axis=0))
                })
            ], ignore_index=True)

            # global 2D self consistency score per sample: n_samples x 1
            if 'sc_score_eternafold' in metrics:
                sc_score_eternafold, pred_sec_structs = self_consistency_score_eternafold(
                    samples.cpu().numpy(), 
                    raw_data['sec_struct_list'], 
                    mask_coords,
                    return_sec_structs = True
                )
                sc_score_eternafold_list.append(sc_score_eternafold.mean())

            # global 1D self consistency score per sample: n_samples x 1
            if 'sc_score_ribonanzanet' in metrics:
                sc_score_ribonanzanet, pred_chem_mods = self_consistency_score_ribonanzanet(
                    samples.cpu().numpy(), 
                    raw_data['sequence'],
                    mask_coords, 
                    ribonanza_net,
                    return_chem_mods = True
                )
                sc_score_ribonanzanet_list.append(sc_score_ribonanzanet.mean())
            
            # global 3D self consistency scores per sample: n_samples x 1, each
            if 'sc_score_rhofold' in metrics:
                try:
                    output_dir = os.path.join(
                        wandb.run.dir, f"designs_{model_name}/{current_datetime}/sample{idx}/")
                except AttributeError:
                    output_dir = os.path.join(
                        PROJECT_PATH, f"designs_{model_name}/{current_datetime}/sample{idx}/")

                sc_score_rmsd, sc_score_tm, sc_score_gdt = self_consistency_score_rhofold(
                    samples.cpu().numpy(), 
                    raw_data,
                    mask_coords,
                    rhofold,
                    output_dir,
                    save_designs = save_designs
                )
                sc_score_rmsd_list.append(sc_score_rmsd.mean())
                sc_score_tm_list.append(sc_score_tm.mean())
                sc_score_gddt_list.append(sc_score_gdt.mean())

                rmsd_within_thresh_list.append((sc_score_rmsd <= RMSD_THRESHOLD).sum() / n_samples)
                tm_within_thresh_list.append((sc_score_tm >= TM_THRESHOLD).sum() / n_samples)
                gddt_within_thresh_list.append((sc_score_gdt >= GDT_THRESHOLD).sum() / n_samples)

                if save_designs:
                    # collate designed sequences in fasta format
                    sequences = [SeqRecord(
                        Seq(raw_data["sequence"]), id=f"input_sequence,", 
                        description=f"pdb_id={raw_data['id_list'][0]} rfam={raw_data['rfam_list'][0]} eq_class={raw_data['eq_class_list'][0]} cluster={raw_data['cluster_structsim0.45']}"
                    )]
                    for idx, zipped in enumerate(zip(
                        samples.cpu().numpy(),
                        perplexity,
                        recovery.mean(axis=1),
                        sc_score_eternafold,
                        pred_sec_structs,
                        sc_score_ribonanzanet,
                        pred_chem_mods,
                        sc_score_rmsd,
                        sc_score_tm,
                        sc_score_gdt
                    )):
                        seq, perp, rec, sc, pred_ss, sc_ribo, pred_cm, sc_rmsd, sc_tm, sc_gdt = zipped
                        seq = "".join([NUM_TO_LETTER[num] for num in seq])
                        edit_dist = edit_distance(seq, raw_data['sequence'])
                        sequences.append(SeqRecord(
                            Seq(seq), id=f"sample={idx},",
                            description=f"temperature={temperature} perplexity={perp:.4f} recovery={rec:.4f} edit_dist={edit_dist} sc_score={sc:.4f} sc_score_ribonanzanet={sc_ribo:.4f} sc_score_rmsd={sc_rmsd:.4f} sc_score_tm={sc_tm:.4f} sc_score_gdt={sc_gdt:.4f}"
                        ))
                    # write all designed sequences to output filepath
                    SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")

    out = {
        'df': df,
        'samples_list': samples_list,
        'recovery_list': recovery_list,
        'perplexity_list': perplexity_list
    }
    if 'sc_score_eternafold' in metrics:
        out['sc_score_eternafold'] = sc_score_eternafold_list
    if 'sc_score_ribonanzanet' in metrics:
        out['sc_score_ribonanzanet'] = sc_score_ribonanzanet_list
    if 'sc_score_rhofold' in metrics:
        out['sc_score_rmsd'] = sc_score_rmsd_list
        out['sc_score_tm'] = sc_score_tm_list
        out['sc_score_gddt'] = sc_score_gddt_list
        out['rmsd_within_thresh'] = rmsd_within_thresh_list
        out['tm_within_thresh'] = tm_within_thresh_list
        out['gddt_within_thresh'] = gddt_within_thresh_list
    return out


def self_consistency_score_eternafold(
        samples, 
        true_sec_struct_list, 
        mask_coords,
        n_samples_ss = 1,
        num_to_letter = NUM_TO_LETTER,
        return_sec_structs = False
    ):
    """
    Compute self consistency score for an RNA, given its true secondary structure(s)
    and a list of designed sequences. 
    EternaFold is used to 'forward fold' the designs.
    
    Args:
        samples: designed sequences of shape (n_samples, seq_len)
        true_sec_struct_list: list of true secondary structures (n_true_ss, seq_len)
        mask_coords: mask for missing sequence coordinates to be ignored during evaluation
        n_samples_ss: number of predicted secondary structures per designed sample
        num_to_letter: lookup table mapping integers to nucleotides
        return_sec_structs: whether to return the predicted secondary structures
    
    Workflow:
        
        Input: For a given RNA molecule, we are given:
        - Designed sequences of shape (n_samples, seq_len)
        - True secondary structure(s) of shape (n_true_ss, seq_len)
        
        For each designed sequence:
        - Predict n_sample_ss secondary structures using EternaFold
        - For each pair of true and predicted secondary structures:
            - Compute MCC score between their adjacency matrix representations
        - Take the average MCC score across all n_sample_ss predicted structures
        
        Take the average MCC score across all n_samples designed sequences
    """
    
    n_true_ss = len(true_sec_struct_list)
    sequence_length = mask_coords.sum()
    # map all entries from dotbracket to numerical representation
    true_sec_struct_list = np.array([dotbracket_to_adjacency(ss) for ss in true_sec_struct_list])
    # mask out missing sequence coordinates
    true_sec_struct_list = true_sec_struct_list[:, mask_coords][:, :, mask_coords]
    # reshape to (n_true_ss * n_samples_ss, seq_len, seq_len)
    true_sec_struct_list = torch.tensor(
        true_sec_struct_list
    ).unsqueeze(1).repeat(1, n_samples_ss, 1, 1).reshape(-1, sequence_length, sequence_length)

    mcc_scores = []
    pred_sec_structs = []
    for _sample in samples:
        # convert sample to string
        pred_seq = ''.join([num_to_letter[num] for num in _sample])
        # predict secondary structure(s) for each sample
        pred_sec_struct_list = predict_sec_struct(pred_seq, n_samples=n_samples_ss)
        if return_sec_structs:
            pred_sec_structs.append(copy.copy(pred_sec_struct_list))
        # map all entries from dotbracket to numerical representation
        pred_sec_struct_list = np.array([dotbracket_to_adjacency(ss) for ss in pred_sec_struct_list])
        # reshape to (n_samples_ss * n_true_ss, seq_len, seq_len)
        pred_sec_struct_list = torch.tensor(
            pred_sec_struct_list
        ).unsqueeze(0).repeat(n_true_ss, 1, 1, 1).reshape(-1, sequence_length, sequence_length)

        # compute mean MCC score between pairs of true and predicted secondary structures
        mcc_scores.append(
            binary_matthews_corrcoef(
                pred_sec_struct_list,
                true_sec_struct_list,
            ).float().mean()
        )

    if return_sec_structs:
        return np.array(mcc_scores), pred_sec_structs
    else:
        return np.array(mcc_scores)


def self_consistency_score_ribonanzanet(
    samples,
    true_sequence,
    mask_seq,
    ribonanza_net,
    num_to_letter=NUM_TO_LETTER,
    return_chem_mods=False,
):
    """Compute self consistency score for an RNA, given the (predicted) chemical modifications for
    the original RNA and a list of designed sequences. RibonanzaNet is used to 'forward fold' the
    designs.

    Args:
        samples: designed sequences of shape (n_samples, seq_len)
        true_sequence: true RNA sequence used to predict chemical modifications
        mask_seq: mask for missing sequence coordinates to be ignored during evaluation
        ribonanza_net: RibonanzaNet model
        num_to_letter: lookup table mapping integers to nucleotides
        return_chem_mods: whether to return the predicted chemical modifications

    Workflow:

        Input: For a given RNA molecule, we are given:
        - Designed sequences of shape (n_samples, seq_len)
        - Predicted chemical modifications for original sequence,
          of shape (n_samples, seq_len, 2), predicted via RibonanzaNet, of which we take
          the index 0 from the last channal --> 2A3/SHAPE.

        For each designed sequence:
        - Predict chemical modifications using RibonanzaNet
        - Compute mean absolute error between prediction and chemical modifications for
          the original sequence

        Take the average mean absolute error across all n_samples designed sequences
    """
    # Compute original sequence's chemical modifications using RibonanzaNet
    true_sequence = np.array([char for char in true_sequence])
    true_sequence = "".join(true_sequence[mask_seq])
    true_chem_mod = ribonanza_net.predict(true_sequence).unsqueeze(0).cpu().numpy()[:,:,0]

    _samples = np.array([[num_to_letter[num] for num in seq] for seq in samples])
    pred_chem_mod = ribonanza_net.predict(_samples[:, mask_seq]).cpu().numpy()[:,:,0]
    if return_chem_mods:
        return (np.abs(pred_chem_mod - true_chem_mod).mean(1)), pred_chem_mod
    else:
        return np.abs(pred_chem_mod - true_chem_mod).mean(1)


def self_consistency_score_ribonanzanet_sec_struct(
        samples, 
        true_sec_struct, 
        mask_coords, 
        ribonanza_net_ss,
        num_to_letter = NUM_TO_LETTER,
        return_sec_structs = False
    ):
    # map from dotbracket to numerical representation
    true_sec_struct = np.array(dotbracket_to_adjacency(true_sec_struct, keep_pseudoknots=True))
    # mask out missing sequence coordinates
    true_sec_struct = true_sec_struct[mask_coords][:, mask_coords]
    # (n_samples, seq_len, seq_len)
    true_sec_struct = torch.tensor(true_sec_struct)

    _samples = np.array([[num_to_letter[num] for num in seq] for seq in samples])
    _, pred_sec_structs = ribonanza_net_ss.predict(_samples)  # (n_samples, seq_len, seq_len)
    
    mcc_scores = []
    for pred_sec_struct in pred_sec_structs:
        # map from dotbracket to numerical representation
        pred_sec_struct = torch.tensor(dotbracket_to_adjacency(pred_sec_struct, keep_pseudoknots=True))
        # compute mean MCC score between pairs of true and predicted secondary structures
        mcc_scores.append(
            binary_matthews_corrcoef(
                pred_sec_struct,
                true_sec_struct,
            ).float().mean()
        )

    if return_sec_structs:
        return np.array(mcc_scores), pred_sec_structs
    else:
        return np.array(mcc_scores)


def self_consistency_score_rhofold(
        samples,
        true_raw_data,
        mask_coords,
        rhofold,
        output_dir,
        num_to_letter = NUM_TO_LETTER,
        save_designs = False,
        save_pdbs = False,
        use_relax = False,
    ):
    """
    Compute self consistency score for an RNA, given its true 3D structure(s)
    for the original RNA and a list of designed sequences.
    RhoFold is used to 'forward fold' the designs.

    Credit: adapted from Rishabh Anand

    Args:
        samples: designed sequences of shape (n_samples, seq_len)
        true_raw_data: Original RNA raw data with 3D structure(s) in `coords_list`
        mask_coords: mask for missing sequence coordinates to be ignored during evaluation
        rhofold: RhoFold model
        output_dir: directory to save designed sequences and structures
        num_to_letter: lookup table mapping integers to nucleotides
        save_designs: whether to save designs as fasta to output directory
        save_pdbs: whether to save PDBs of forward-folded designs to output directory
        use_relax: whether to perform Amber relaxation on designed structures

    Workflow:
            
        Input: For a given RNA molecule, we are given:
        - Designed sequences of shape (n_samples, seq_len)
        - True 3D structure(s) of shape (n_true_structs, seq_len, 3)
        
        For each designed sequence:
        - Predict the tertiary structure using RhoFold
        - For each pair of true and predicted 3D structures:
            - Compute RMSD, TM-score & GDT between their C4' coordinates
        
        Take the average self-consistency scores across all n_samples designed sequences

    Returns:
        sc_rmsds: array of RMSD scores per sample
        sc_tms: array of TM-score scores per sample
        sc_gddts: array of GDT scores per sample
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collate designed sequences in fasta format
    # first record: input sequence and model metadata
    input_seq = SeqRecord(
        Seq(true_raw_data["sequence"]),
        id=f"input_sequence,",
        description=f"input_sequence"
    )
    # SeqIO.write(input_seq, os.path.join(output_dir, "input_seq.fasta"), "fasta")
    sequences = [input_seq]
    
    # remaining records: designed sequences and metrics
    sc_rmsds = []
    sc_tms = []
    sc_gddts = []
    for idx, seq in enumerate(samples):
        # Save designed sequence to fasta file (temporary)
        seq = SeqRecord(
            Seq("".join([num_to_letter[num] for num in seq])), 
            id=f"sample={idx},",
            description=f"sample={idx}"
        )
        sequences.append(seq)
        design_fasta_path = os.path.join(output_dir, f"design{idx}.fasta")
        SeqIO.write(seq, design_fasta_path, "fasta")
        
        # Forward fold designed sequence using RhoFold
        design_pdb_path = os.path.join(output_dir, f"design{idx}.pdb")
        rhofold.predict(design_fasta_path, design_pdb_path, use_relax)
        
        # Load C4' coordinates of designed structure
        _, coords, _, _ = pdb_to_tensor(
            design_pdb_path,
            return_sec_struct=False,
            return_sasa=False,
            keep_insertions=False,
        )
        coords = get_c4p_coords(coords)
        # zero-center coordinates
        coords = coords - coords.mean(dim=0)

        # Compute self-consistency between designed and groundtruth structures
        _sc_rmsds = []
        _sc_tms = []
        _sc_gddts = []
        for other_coords in true_raw_data["coords_list"]:
            _other = get_c4p_coords(other_coords)[mask_coords, :]
            # zero-center other coordinates
            _other = _other - _other.mean(dim=0)
            # globally align coordinates
            R_hat = rotation_matrix(
                _other,  # mobile set
                coords # reference set
            )[0]
            _other = _other @ R_hat.T
            # compute metrics
            _sc_rmsds.append(get_rmsd(
                coords, _other, superposition=True, center=True))
            _sc_tms.append(get_tmscore(coords, _other))
            _sc_gddts.append(get_gddt(coords, _other))

        sc_rmsds.append(np.mean(_sc_rmsds))
        sc_tms.append(np.mean(_sc_tms))
        sc_gddts.append(np.mean(_sc_gddts))

        # remove temporary files
        os.unlink(design_fasta_path)
        if save_pdbs is False:
            os.unlink(design_pdb_path)
    
    if save_designs is False:
        # remove output directory        
        shutil.rmtree(output_dir)
    else:
        # write all designed sequences to output filepath
        SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")

    return np.array(sc_rmsds), np.array(sc_tms), np.array(sc_gddts)


def get_tmscore(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Template Modelling score (TM-score). 
    
    Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

    https://en.wikipedia.org/wiki/Template_modeling_score

    TM-score is a measure of similarity between two protein structures.
    The TM-score is intended as a more accurate measure of the global
    similarity of full-length protein structures than the often used RMSD
    measure. The TM-score indicates the similarity between two structures
    by a score between ``[0, 1]``, where 1 indicates a perfect match
    between two structures (thus the higher the better). Generally scores
    below 0.20 corresponds to randomly chosen unrelated proteins whereas
    structures with a score higher than 0.5 assume roughly the same fold.
    A quantitative study shows that proteins of TM-score = 0.5 have a
    posterior probability of 37% in the same CATH topology family and of
    13% in the same SCOP fold family. The probabilities increase rapidly
    when TM-score > 0.5. The TM-score is designed to be independent of
    protein lengths.
    
    We have adapted the implementation to RNA (TM-score threshold = 0.45).
    Requires aligned C4' coordinates as input.
    """
    l_target = y.shape[0]
    d0_l_target = 1.24 * np.power(l_target - 15, 1 / 3) - 1.8
    di = torch.pairwise_distance(y_hat, y)
    out = torch.sum(1 / (1 + (di / d0_l_target) ** 2)) / l_target
    if torch.isnan(out):
        return torch.tensor(0.0)
    return out


def get_gddt(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Global Distance Deviation Test metric (GDDT).

    Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

    https://en.wikipedia.org/wiki/Global_distance_test

    The GDT score is calculated as the largest set of amino acid residues'
    alpha carbon atoms in the model structure falling within a defined
    distance cutoff of their position in the experimental structure, after
    iteratively superimposing the two structures. By the original design the
    GDT algorithm calculates 20 GDT scores, i.e. for each of 20 consecutive distance
    cutoffs (``0.5 Å, 1.0 Å, 1.5 Å, ... 10.0 Å``). For structure similarity assessment
    it is intended to use the GDT scores from several cutoff distances, and scores
    generally increase with increasing cutoff. A plateau in this increase may
    indicate an extreme divergence between the experimental and predicted structures,
    such that no additional atoms are included in any cutoff of a reasonable distance.
    The conventional GDT_TS total score in CASP is the average result of cutoffs at
    ``1``, ``2``, ``4``, and ``8`` Å.

    Random predictions give around 20; getting the gross topology right gets one to ~50; 
    accurate topology is usually around 70; and when all the little bits and pieces, 
    including side-chain conformations, are correct, GDT_TS begins to climb above 90.

    We have adapted the implementation to RNA.
    Requires aligned C4' coordinates as input.
    """
    # Get distance between points
    dist = torch.norm(y - y_hat, dim=1)

    # Return mean fraction of distances below cutoff for each cutoff (1, 2, 4, 8)
    count_1 = (dist < 1).sum() / dist.numel()
    count_2 = (dist < 2).sum() / dist.numel()
    count_4 = (dist < 4).sum() / dist.numel()
    count_8 = (dist < 8).sum() / dist.numel()
    out = torch.mean(torch.tensor([count_1, count_2, count_4, count_8]))
    if torch.isnan(out):
        return torch.tensor(0.0)
    return out


def edit_distance(s: str, t: str) -> int:
    """
    A Space efficient Dynamic Programming based Python3 program 
    to find minimum number operations to convert str1 to str2

    Source: https://www.geeksforgeeks.org/edit-distance-dp-5/
    """
    n = len(s)
    m = len(t)

    prev = [j for j in range(m+1)]
    curr = [0] * (m+1)

    for i in range(1, n+1):
        curr[0] = i
        for j in range(1, m+1):
            if s[i-1] == t[j-1]:
                curr[j] = prev[j-1]
            else:
                mn = min(1 + prev[j], 1 + curr[j-1])
                curr[j] = min(mn, 1 + prev[j-1])
        prev = curr.copy()

    return prev[m]
