"""Evaluation metrics and self-consistency scoring for gRNAde RNA inverse folding models.

This module provides evaluation infrastructure for assessing the quality of RNA 
sequences designed by structure-conditioned generative models. The evaluation suite
computes multiple metrics at different levels of structural hierarchy:

    1. Sequence-level: Recovery rate and perplexity
    2. Secondary structure (2D): Self-consistency via structure prediction
    3. Chemical reactivity (1D): Self-consistency via SHAPE-like profiles
    4. Tertiary structure (3D): Self-consistency via 3D structure prediction

Self-Consistency Scoring:
    Self-consistency measures whether designed sequences are predicted to fold into the
    target structure by independent forward-folding models. This provides an in silico
    assessment of design quality without experimental validation:
        - EternaFold: Secondary structure prediction (dot-bracket notation)
        - RibonanzaNet: Chemical reactivity profiles (SHAPE-like)
        - RibonanzaNetSS: Secondary structure with pseudoknots
        - RhoFold: 3D structure prediction (coordinates)

Key Functions:
    evaluate: Main evaluation loop computing all requested metrics
    self_consistency_score_eternafold: 2D structure self-consistency (no pseudoknots)
    self_consistency_score_ribonanzanet: Chemical reactivity self-consistency
    self_consistency_score_ribonanzanet_sec_struct: 2D structure with pseudoknots
    openknot_score_ribonanzanet: OpenKnot score for Eterna competition
    self_consistency_score_rhofold: 3D structure self-consistency
    get_tmscore: TM-score for structure alignment quality
    get_gddt: GDT_TS score for structure accuracy
    edit_distance: Levenshtein distance between sequences

Metrics Computed:
    - Recovery: Fraction of native sequence recovered in designs
    - Perplexity: Sequence likelihood under the model
    - sc_score_eternafold: MCC between predicted and target 2D structure
    - sc_score_ribonanzanet: MAE between predicted and target reactivity
    - sc_score_ribonanzanet_ss: MCC for pseudoknotted structures
    - sc_score_rhofold: RMSD, TM-score, GDT_TS for 3D structure
"""

import copy
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd
from torchmetrics.functional.classification import binary_matthews_corrcoef
from tqdm import tqdm

import wandb
from src.constants import (
    GDT_THRESHOLD,
    NUM_TO_LETTER,
    PROJECT_PATH,
    RMSD_THRESHOLD,
    TM_THRESHOLD,
)
from src.data.data_utils import get_c4p_coords, pdb_to_tensor
from src.data.sec_struct_utils import (
    dotbracket_to_adjacency,
    dotbracket_to_paired,
    predict_sec_struct,
)
from src.openknot_score import get_openknot_score


##################################
# Main Evaluation Function
##################################

def evaluate(
    model,
    dataset,
    n_samples,
    temperature,
    device,
    model_name="eval",
    metrics=[
        "recovery",
        "perplexity",
        "sc_score_eternafold",
        "sc_score_ribonanzanet",
        "sc_score_ribonanzanet_ss",
        "sc_score_rhofold",
    ],
    save_designs=False,
):
    """Evaluation suite for RNA inverse folding models.

    This function evaluates a trained RNA sequence design model by sampling multiple
    sequences for each target structure and computing a comprehensive set of metrics
    across different levels of structural hierarchy. Evaluation includes sequence-level
    metrics (recovery, perplexity) and multiple self-consistency scores that assess
    whether designed sequences are predicted to adopt the target structure by independent
    forward-folding models.

    Available Metrics:
        1. recovery: Sequence recovery per residue (mean gives per-sample recovery).
           Measures how well the model recovers the native sequence.
        
        2. perplexity: Sequence likelihood under the model, measuring prediction
           confidence. Lower perplexity indicates higher confidence.
        
        3. sc_score_eternafold: Secondary structure self-consistency using EternaFold
           for structure prediction. Computes MCC between predicted and target 2D
           structures (no pseudoknots).
        
        4. sc_score_ribonanzanet: Chemical reactivity self-consistency using RibonanzaNet
           to predict SHAPE-like reactivity profiles. Computes MAE between predicted
           reactivity of designed and native sequences.
        
        5. sc_score_ribonanzanet_ss: Secondary structure self-consistency with
           pseudoknot support using RibonanzaNetSS. Computes MCC between predicted
           and target structures with pseudoknots.
        
        6. sc_score_rhofold: Tertiary structure self-consistency using RhoFold for
           3D structure prediction. Computes RMSD, TM-score, and GDT_TS between
           predicted and target C4' coordinates.

    Args:
        model (torch.nn.Module): Trained RNA inverse folding model for sequence design
        dataset (RNADesignDataset): Dataset containing target structures to evaluate on
        n_samples (int): Number of sequences to sample per target structure. More samples
            provide better statistics but increase computation time.
        temperature (float): Sampling temperature for sequence generation. Lower values
            (e.g., 0.1) produce more confident, less diverse samples; higher values
            (e.g., 1.0) increase diversity.
        device (torch.device or str): Device for model inference ('cpu', 'cuda')
        model_name (str): Identifier for the model/dataset, used in logging and file
            naming. Defaults to 'eval'.
        metrics (list): List of metric names to compute. Must include 'recovery'.
            Options: 'recovery', 'perplexity', 'sc_score_eternafold', 'sc_score_ribonanzanet',
            'sc_score_ribonanzanet_ss', 'sc_score_rhofold'. Defaults to all metrics.
        save_designs (bool): Whether to save designed sequences to FASTA files with
            associated metrics. Useful for downstream experimental validation. Defaults to False.

    Returns:
        dict: Dictionary containing evaluation results with keys:
            - df (pd.DataFrame): Per-residue aggregated metrics (mean across samples)
            - samples_list (list): Designed sequences, list of tensors [n_samples, seq_len]
            - recovery_list (list): Mean sequence recovery per structure
            - perplexity_list (list): Perplexity per structure
            - sc_score_eternafold (list): 2D self-consistency scores (optional)
            - sc_score_ribonanzanet (list): Reactivity self-consistency (optional)
            - sc_score_ribonanzanet_ss (list): 2D with pseudoknots (optional)
            - sc_score_rmsd (list): 3D RMSD self-consistency (optional)
            - sc_score_tm (list): 3D TM-score self-consistency (optional)
            - sc_score_gddt (list): 3D GDT_TS self-consistency (optional)
            - rmsd_within_thresh (list): % designs with RMSD ≤ 2.0Å (optional)
            - tm_within_thresh (list): % designs with TM-score ≥ 0.45 (optional)
            - gddt_within_thresh (list): % designs with GDT_TS ≥ 0.50 (optional)

    Workflow:
        1. Initialize optional forward-folding models (EternaFold, RibonanzaNet, RhoFold)
        2. For each structure in the dataset:
            a. Sample n_samples sequences using the model
            b. Compute sequence recovery and perplexity
            c. Compute requested self-consistency scores
            d. Optionally save designs to FASTA
        3. Aggregate results into DataFrame and metric lists
        4. Return comprehensive results dictionary
    """
    assert "recovery" in metrics, "Sequence recovery must be computed for evaluation"

    #######################################################################
    # Optionally initialise other models used for self-consistency scoring
    #######################################################################

    if "sc_score_ribonanzanet" in metrics:
        from tools.ribonanzanet.network import RibonanzaNet

        # Initialise RibonanzaNet for self-consistency score
        ribonanza_net = RibonanzaNet(
            os.path.join(PROJECT_PATH, "tools/ribonanzanet/config.yaml"),
            os.path.join(PROJECT_PATH, "checkpoints/ribonanzanet/ribonanzanet.pt"),
            device,
        )
        # Transfer model to device in eval mode
        ribonanza_net = ribonanza_net.to(device)
        ribonanza_net.eval()

    if "sc_score_ribonanzanet_ss" in metrics:
        from tools.ribonanzanet_sec_struct.network import RibonanzaNetSS

        # Initialise RibonanzaNetSS for 2D self-consistency score w/ pseudoknots
        ribonanza_net_ss = RibonanzaNetSS(
            os.path.join(PROJECT_PATH, "tools/ribonanzanet_sec_struct/config.yaml"),
            os.path.join(PROJECT_PATH, "checkpoints/ribonanzanet_sec_struct/ribonanzanet_ss.pt"),
            device,
        )
        # Transfer model to device in eval mode
        ribonanza_net_ss = ribonanza_net_ss.to(device)
        ribonanza_net_ss.eval()

    if "sc_score_rhofold" in metrics:
        from tools.rhofold.config import rhofold_config
        from tools.rhofold.rf import RhoFold

        # Initialise RhoFold for 3D self-consistency score
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = os.path.join(PROJECT_PATH, "checkpoints/rhofold/rhofold_20221010.pt")
        print(f"Loading RhoFold checkpoint: {rhofold_path}")
        rhofold.load_state_dict(
            torch.load(rhofold_path, map_location=torch.device("cpu"))["model"]
        )
        # Transfer model to device in eval mode
        rhofold = rhofold.to(device)
        rhofold.eval()
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    ####################################################
    # Evaluation loop over each data point sequentially
    ####################################################

    # per sample metric lists for storing evaluation results
    samples_list = []  # list of tensors of shape (n_samples, seq_len) per data point
    recovery_list = []  # list of mean recovery per data point
    perplexity_list = []  # list of mean perplexity per data point
    sc_score_ribonanzanet_list = []  # list of 1D self-consistency scores per data point
    sc_score_eternafold_list = []  # list of 2D self-consistency scores per data point
    sc_score_ribonanzanet_ss_list = (
        []
    )  # list of 2D self-consistency scores w/ pseudoknots per data point
    sc_score_rmsd_list = []  # list of 3D self-consistency RMSDs per data point
    rmsd_within_thresh_list = []  # list of % scRMSDs within threshold per data point
    sc_score_tm_list = []  # list of 3D self-consistency TM-scores per data point
    tm_within_thresh_list = []  # list of % scTMs within threshold per data point
    sc_score_gddt_list = []  # list of 3D self-consistency GDTs per data point
    gddt_within_thresh_list = []  # list of % scGDDTs within threshold per data point

    # DataFrame to store metrics and metadata per residue per sample for analysis and plotting
    df = pd.DataFrame(columns=["idx", "recovery", "sasa", "paired", "rmsds", "model_name"])

    model.eval()
    with torch.no_grad():
        t = tqdm(enumerate(dataset.data_list), total=len(dataset.data_list))
        for idx, raw_data in t:
            # featurise raw data
            data = dataset.featurizer(raw_data).to(device)

            # sample n_samples from model for single data point: n_samples x seq_len
            samples, logits = model.sample(data, n_samples, temperature, return_logits=True)
            samples_list.append(samples.cpu().numpy())

            # perplexity per sample: n_samples x 1
            n_nodes = logits.shape[1]
            perplexity = (
                torch.exp(
                    F.cross_entropy(
                        logits.view(n_samples * n_nodes, model.out_dim),
                        samples.view(n_samples * n_nodes).long(),
                        reduction="none",
                    )
                    .view(n_samples, n_nodes)
                    .mean(dim=1)
                )
                .cpu()
                .numpy()
            )
            perplexity_list.append(perplexity.mean())

            ###########
            # Metadata
            ###########

            # per residue average SASA: seq_len x 1
            mask_seq = data.mask_seq.cpu().numpy()
            sasa = np.mean(raw_data["sasa_list"], axis=0)[mask_seq]

            # per residue indicator for paired/unpaired: seq_len x 1
            paired = np.mean(
                [dotbracket_to_paired(sec_struct) for sec_struct in raw_data["sec_struct_list"]],
                axis=0,
            )[mask_seq]

            # per residue average RMSD: seq_len x 1
            if len(raw_data["coords_list"]) == 1:
                rmsds = np.zeros_like(sasa)
            else:
                rmsds = []
                for i in range(len(raw_data["coords_list"])):
                    for j in range(i + 1, len(raw_data["coords_list"])):
                        coords_i = get_c4p_coords(raw_data["coords_list"][i])
                        coords_j = get_c4p_coords(raw_data["coords_list"][j])
                        rmsds.append(
                            torch.sqrt(torch.sum((coords_i - coords_j) ** 2, dim=1)).cpu().numpy()
                        )
                rmsds = np.stack(rmsds).mean(axis=0)[mask_seq]

            ##########
            # Metrics
            ##########

            # sequence recovery per residue across all samples: n_samples x seq_len
            recovery = samples[:, mask_seq].eq(data.seq[mask_seq]).float().cpu().numpy()
            recovery_list.append(recovery.mean())
            t.set_description(f"{model_name.upper()} recovery: {np.mean(recovery_list):.4f}")

            # update per residue per sample dataframe
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "idx": [idx] * len(recovery.mean(axis=0)),
                            "recovery": recovery.mean(axis=0),
                            "sasa": sasa,
                            "paired": paired,
                            "rmsds": rmsds,
                            "model_name": [model_name] * len(recovery.mean(axis=0)),
                        }
                    ),
                ],
                ignore_index=True,
            )

            # global 2D self consistency score per sample: n_samples x 1
            if "sc_score_eternafold" in metrics:
                (
                    sc_score_eternafold,
                    pred_sec_structs_eternafold,
                ) = self_consistency_score_eternafold(
                    samples.cpu().numpy(),
                    raw_data["sec_struct_list"],
                    mask_seq,
                    return_sec_structs=True,
                )
                sc_score_eternafold_list.append(sc_score_eternafold.mean())

            # global 1D self consistency score per sample: n_samples x 1
            if "sc_score_ribonanzanet" in metrics:
                sc_score_ribonanzanet, pred_chem_mods = self_consistency_score_ribonanzanet(
                    samples.cpu().numpy(),
                    raw_data["sequence"],
                    mask_seq,
                    ribonanza_net,
                    return_chem_mods=True,
                )
                sc_score_ribonanzanet_list.append(sc_score_ribonanzanet.mean())

            # global self consistency score (2D, RibonanzaNet) per sample: n_samples x 1
            if "sc_score_ribonanzanet_ss" in metrics:
                (
                    sc_score_ribonanzanet_ss,
                    pred_sec_structs_ribonanzanet_ss,
                ) = self_consistency_score_ribonanzanet_sec_struct(
                    samples.cpu().numpy(),
                    raw_data["sec_struct_list"][0],
                    mask_seq,
                    ribonanza_net_ss,
                    return_sec_structs=True,
                )
                sc_score_ribonanzanet_ss_list.append(sc_score_ribonanzanet_ss.mean())

            # global 3D self consistency scores per sample: n_samples x 1, each
            if "sc_score_rhofold" in metrics:
                try:
                    output_dir = os.path.join(
                        wandb.run.dir, f"designs_{model_name}/{current_datetime}/sample{idx}/"
                    )
                except AttributeError:
                    output_dir = os.path.join(
                        PROJECT_PATH, f"designs_{model_name}/{current_datetime}/sample{idx}/"
                    )

                sc_score_rmsd, sc_score_tm, sc_score_gdt = self_consistency_score_rhofold(
                    samples.cpu().numpy(),
                    raw_data,
                    mask_seq,
                    rhofold,
                    output_dir,
                    save_designs=save_designs,
                )
                sc_score_rmsd_list.append(sc_score_rmsd.mean())
                sc_score_tm_list.append(sc_score_tm.mean())
                sc_score_gddt_list.append(sc_score_gdt.mean())

                rmsd_within_thresh_list.append((sc_score_rmsd <= RMSD_THRESHOLD).sum() / n_samples)
                tm_within_thresh_list.append((sc_score_tm >= TM_THRESHOLD).sum() / n_samples)
                gddt_within_thresh_list.append((sc_score_gdt >= GDT_THRESHOLD).sum() / n_samples)

                if save_designs:
                    # collate designed sequences in fasta format
                    sequences = [
                        SeqRecord(
                            Seq(raw_data["sequence"]),
                            id=f"input_sequence,",
                            description=f"pdb_id={raw_data['id_list'][0]} rfam={raw_data['rfam_list'][0]} eq_class={raw_data['eq_class_list'][0]} cluster={raw_data['cluster_structsim0.45']}",
                        )
                    ]
                    for idx, zipped in enumerate(
                        zip(
                            samples.cpu().numpy(),
                            perplexity,
                            recovery.mean(axis=1),
                            sc_score_eternafold,
                            # pred_sec_structs_eternafold,
                            sc_score_ribonanzanet,
                            # pred_chem_mods,
                            sc_score_ribonanzanet_ss,
                            # pred_sec_structs_ribonanzanet_ss,
                            sc_score_rmsd,
                            sc_score_tm,
                            sc_score_gdt,
                        )
                    ):
                        (
                            seq,
                            perp,
                            rec,
                            sc_eternafold,
                            # pred_ss_eternafold,
                            sc_ribo,
                            # pred_cm,
                            sc_ribo_ss,
                            # pred_ss_ribo,
                            sc_rmsd,
                            sc_tm,
                            sc_gdt,
                        ) = zipped
                        seq = "".join([NUM_TO_LETTER[num] for num in seq])
                        edit_dist = edit_distance(seq, raw_data["sequence"])
                        sequences.append(
                            SeqRecord(
                                Seq(seq),
                                id=f"sample={idx},",
                                description=f"temperature={temperature} perplexity={perp:.4f} recovery={rec:.4f} edit_dist={edit_dist} sc_score_eternafold={sc_eternafold:.4f} sc_score_ribonanzanet_ss={sc_ribo_ss:.4f} sc_score_ribonanzanet={sc_ribo:.4f} sc_score_rmsd={sc_rmsd:.4f} sc_score_tm={sc_tm:.4f} sc_score_gdt={sc_gdt:.4f}",
                            )
                        )
                    # write all designed sequences to output filepath
                    SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")

    out = {
        "df": df,
        "samples_list": samples_list,
        "recovery_list": recovery_list,
        "perplexity_list": perplexity_list,
    }
    if "sc_score_eternafold" in metrics:
        out["sc_score_eternafold"] = sc_score_eternafold_list
    if "sc_score_ribonanzanet" in metrics:
        out["sc_score_ribonanzanet"] = sc_score_ribonanzanet_list
    if "sc_score_ribonanzanet_ss" in metrics:
        out["sc_score_ribonanzanet_ss"] = sc_score_ribonanzanet_ss_list
    if "sc_score_rhofold" in metrics:
        out["sc_score_rmsd"] = sc_score_rmsd_list
        out["sc_score_tm"] = sc_score_tm_list
        out["sc_score_gddt"] = sc_score_gddt_list
        out["rmsd_within_thresh"] = rmsd_within_thresh_list
        out["tm_within_thresh"] = tm_within_thresh_list
        out["gddt_within_thresh"] = gddt_within_thresh_list
    return out


##################################
# Self-Consistency Scoring Functions
##################################

def self_consistency_score_eternafold(
    samples,
    true_sec_struct_list,
    mask_seq,
    n_samples_ss=1,
    num_to_letter=NUM_TO_LETTER,
    return_sec_structs=False,
):
    """Compute 2D secondary structure self-consistency using EternaFold.

    This function assesses whether designed RNA sequences are predicted to fold into
    the target secondary structure by using EternaFold as an independent forward-folding
    predictor. Self-consistency is measured via Matthews Correlation Coefficient (MCC)
    between predicted and target structures represented as adjacency matrices.

    High MCC scores (close to 1.0) indicate that the designed sequence is likely to
    adopt the intended secondary structure, providing in silico validation before
    experimental testing.

    Args:
        samples (np.ndarray or torch.Tensor): Designed sequences with shape [n_samples, seq_len]
            where each element is an integer nucleotide code (0=A, 1=G, 2=C, 3=U)
        true_sec_struct_list (list): List of target secondary structures in dot-bracket
            notation. Can contain multiple conformers: [sec_struct_1, sec_struct_2, ...]
        mask_seq (np.ndarray or torch.Tensor): Boolean mask of shape [seq_len] indicating
            valid positions to evaluate (False for missing coordinates)
        n_samples_ss (int): Number of secondary structure predictions per designed sequence
            using EternaFold's stochastic sampling. Defaults to 1.
        num_to_letter (dict): Mapping from integer codes to nucleotide letters
            (e.g., {0: 'A', 1: 'G', 2: 'C', 3: 'U'}). Defaults to NUM_TO_LETTER.
        return_sec_structs (bool): If True, return predicted structures along with scores.
            Defaults to False.

    Returns:
        np.ndarray or tuple: If return_sec_structs=False, returns an array of MCC scores
            with one value per designed sequence (shape: [n_samples]).
            If return_sec_structs=True, returns (mcc_per_sample, predicted_structures_list),
            where mcc_per_sample is a 1D numpy array and predicted_structures_list contains
            lists of dot-bracket structures per sample.

    Workflow:
        1. Convert target structures to adjacency matrix representation
        2. For each designed sequence:
            a. Convert integer sequence to nucleotide string
            b. Predict n_samples_ss structures using EternaFold
            c. Convert predictions to adjacency matrices
            d. Compute MCC against all target structures
            e. Average MCC across predictions
        3. Return mean MCC across all designed sequences
    """

    n_true_ss = len(true_sec_struct_list)
    sequence_length = mask_seq.sum()
    # map all entries from dotbracket to numerical representation
    true_sec_struct_list = np.array([dotbracket_to_adjacency(ss) for ss in true_sec_struct_list])
    # mask out missing sequence coordinates
    true_sec_struct_list = true_sec_struct_list[:, mask_seq][:, :, mask_seq]
    # reshape to (n_true_ss * n_samples_ss, seq_len, seq_len)
    true_sec_struct_list = (
        torch.tensor(true_sec_struct_list)
        .unsqueeze(1)
        .repeat(1, n_samples_ss, 1, 1)
        .reshape(-1, sequence_length, sequence_length)
    )

    mcc_scores = []
    pred_sec_structs = []
    for _sample in samples:
        # convert sample to string
        pred_seq = "".join([num_to_letter[num] for num in _sample])
        # predict secondary structure(s) for each sample
        pred_sec_struct_list = predict_sec_struct(pred_seq, n_samples=n_samples_ss)
        if return_sec_structs:
            pred_sec_structs.append(copy.copy(pred_sec_struct_list))
        # map all entries from dotbracket to numerical representation
        pred_sec_struct_list = np.array(
            [dotbracket_to_adjacency(ss) for ss in pred_sec_struct_list]
        )
        # mask out missing sequence coordinates
        pred_sec_struct_list = pred_sec_struct_list[:, mask_seq][:, :, mask_seq]
        # reshape to (n_samples_ss * n_true_ss, seq_len, seq_len)
        pred_sec_struct_list = (
            torch.tensor(pred_sec_struct_list)
            .unsqueeze(0)
            .repeat(n_true_ss, 1, 1, 1)
            .reshape(-1, sequence_length, sequence_length)
        )

        # compute mean MCC score between pairs of true and predicted secondary structures
        mcc_scores.append(
            binary_matthews_corrcoef(
                pred_sec_struct_list,
                true_sec_struct_list,
            )
            .float()
            .mean()
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
    """Compute chemical reactivity self-consistency using RibonanzaNet.

    This function assesses whether designed RNA sequences are predicted to have similar
    chemical reactivity profiles as the native sequence, using RibonanzaNet to predict
    SHAPE reactivity. Chemical reactivity is a sensitive reporter of RNA structure, 
    with unpaired nucleotides showing high reactivity and paired/structured regions 
    showing low reactivity.

    Self-consistency is measured via Mean Absolute Error (MAE) between the reactivity
    profiles of designed and native sequences. Lower MAE indicates that designs are
    predicted to adopt similar structures to the native sequence.

    Args:
        samples (np.ndarray or torch.Tensor): Designed sequences with shape [n_samples, seq_len]
            where each element is an integer nucleotide code
        true_sequence (str or np.ndarray): Native RNA sequence for reference reactivity
            prediction. Can be string or integer-encoded array.
        mask_seq (np.ndarray or torch.Tensor): Boolean mask of shape [seq_len] indicating
            valid positions (False for missing coordinates)
        ribonanza_net: RibonanzaNet model instance for reactivity prediction
        num_to_letter (dict): Mapping from integer codes to nucleotide letters.
            Defaults to NUM_TO_LETTER.
        return_chem_mods (bool): If True, return predicted reactivity profiles along
            with MAE scores. Defaults to False.

    Returns:
        np.ndarray or tuple: If return_chem_mods=False, returns an array of per-sample
            MAE values (shape: [n_samples]). If return_chem_mods=True, returns
            (mae_per_sample, designed_reactivities), where designed_reactivities has
            shape [n_samples, seq_len].

    Workflow:
        1. Predict reference reactivity profile for native sequence using RibonanzaNet
        2. For each designed sequence:
            a. Convert integer sequence to nucleotide string
            b. Predict reactivity profile using RibonanzaNet (2A3/SHAPE channel)
            c. Compute MAE against reference profile
        3. Return mean MAE across all designed sequences
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
    

def openknot_score_ribonanzanet(
    samples,
    true_sec_struct,
    mask_seq,
    ribonanza_net,
    num_to_letter=NUM_TO_LETTER,
    return_chem_mods=False,
):
    """Compute OpenKnot score for RNA designs using in silico SHAPE reactivity.

    The OpenKnot score is a metric developed for the Eterna OpenKnot Benchmark that
    assesses design quality by comparing predicted SHAPE reactivity patterns against
    the target secondary structure. It measures the consistency between chemical
    reactivity (which indicates structural flexibility) and the intended base pairing.

    Scores above 90 are considered high-confidence successful designs in the Eterna
    competition. This metric is particularly useful for pseudoknotted structures where
    traditional structure predictors may struggle.

    Args:
        samples (np.ndarray or torch.Tensor): Designed sequences with shape [n_samples, seq_len]
        true_sec_struct (str): Target secondary structure in dot-bracket notation,
            supporting pseudoknots (e.g., "((([[[)))]]]")  
        mask_seq (np.ndarray or torch.Tensor): Boolean mask indicating valid positions
        ribonanza_net: RibonanzaNet model for predicting SHAPE reactivity
        num_to_letter (dict): Nucleotide integer-to-letter mapping. Defaults to NUM_TO_LETTER.
        return_chem_mods (bool): If True, return predicted reactivity profiles.
            Defaults to False.

    Returns:
        np.ndarray or tuple: If return_chem_mods=False, returns an array of OpenKnot
            scores per sample (shape: [n_samples]). If return_chem_mods=True, returns
            (scores_per_sample, reactivity_profiles) where reactivity_profiles has shape
            [n_samples, seq_len].

    Workflow:
        1. Predict SHAPE reactivity for each designed sequence using RibonanzaNet
        2. For each design, compute OpenKnot score by comparing:
            - Low reactivity in paired/pseudoknotted regions (expected)
            - High reactivity in unpaired/loop regions (expected)
        3. Return mean score across all designs

    Note:
        Used specifically for the Eterna OpenKnot Benchmark evaluation. See
        src.openknot_score for detailed scoring algorithm.
    """
    # Predict SHAPE reactivity for designed sequences using RibonanzaNet
    _samples = np.array([[num_to_letter[num] for num in seq] for seq in samples])
    pred_chem_mods = ribonanza_net.predict(_samples[:, mask_seq]).cpu().numpy()[:, :, 0]

    # compute OpenKnot score
    _, _, _, openknot_scores = get_openknot_score(
        true_sec_struct,
        pred_chem_mods,
    )
        
    if return_chem_mods:
        return openknot_scores, pred_chem_mods
    else:
        return openknot_scores


def self_consistency_score_ribonanzanet_sec_struct(
    samples,
    true_sec_struct,
    mask_seq,
    ribonanza_net_ss,
    num_to_letter=NUM_TO_LETTER,
    return_sec_structs=False,
):
    """Compute 2D secondary structure self-consistency with pseudoknot support.

    This function extends traditional 2D structure self-consistency scoring to handle
    complex RNA topologies including pseudoknots, which are essential for many functional
    RNAs. RibonanzaNetSS, a specialized model trained on pseudoknotted structures, is
    used to predict secondary structures that can include non-nested base pairs.

    Pseudoknots occur when bases in a loop pair with bases outside that loop, creating
    interleaved base pairs that cannot be represented by standard nested parentheses.
    They are critical for ribosomal RNAs, viral RNAs, and riboswitches.

    Args:
        samples (np.ndarray or torch.Tensor): Designed sequences with shape [n_samples, seq_len]
        true_sec_struct (str): Target secondary structure in extended dot-bracket notation
            supporting pseudoknots (e.g., \"((([[[)))]]]\")
        mask_seq (np.ndarray or torch.Tensor): Boolean mask indicating valid positions
        ribonanza_net_ss: RibonanzaNetSS model for pseudoknotted structure prediction
        num_to_letter (dict): Nucleotide integer-to-letter mapping. Defaults to NUM_TO_LETTER.
        return_sec_structs (bool): If True, return predicted structures along with MCC scores.
            Defaults to False.

    Returns:
        float or tuple: If return_sec_structs=False, returns mean MCC score (float).
            If return_sec_structs=True, returns (mean_mcc, predicted_structures_list).

    Workflow:

        Input: For a given RNA molecule, we are given:
        - Designed sequences of shape (n_samples, seq_len)
        - True secondary structure in dot-bracket notation

        For each designed sequence:
        - Predict secondary structure using RibonanzaNetSS (supports pseudoknots)
        - Convert both true and predicted structures to adjacency matrix representations
        - Compute MCC score between the adjacency matrices

        Take the average MCC score across all n_samples designed sequences

    Note:
        Currently only supports single-state secondary structures.
    """

    # map from dotbracket to numerical representation
    true_sec_struct = np.array(dotbracket_to_adjacency(true_sec_struct, keep_pseudoknots=True))
    # mask out missing sequence coordinates
    true_sec_struct = true_sec_struct[mask_seq][:, mask_seq]
    # (n_samples, seq_len, seq_len)
    true_sec_struct = torch.tensor(true_sec_struct)

    _samples = np.array([[num_to_letter[num] for num in seq] for seq in samples])
    _, pred_sec_structs = ribonanza_net_ss.predict(_samples)  # (n_samples, seq_len, seq_len)

    mcc_scores = []
    for pred_sec_struct in pred_sec_structs:
        # map from dotbracket to numerical representation
        pred_sec_struct = torch.tensor(
            dotbracket_to_adjacency(pred_sec_struct, keep_pseudoknots=True)[mask_seq][:, mask_seq]
        )
        # compute mean MCC score between pairs of true and predicted secondary structures
        mcc_scores.append(
            binary_matthews_corrcoef(
                pred_sec_struct,
                true_sec_struct,
            )
            .float()
            .mean()
        )

    if return_sec_structs:
        return np.array(mcc_scores), pred_sec_structs
    else:
        return np.array(mcc_scores)


def self_consistency_score_rhofold(
    samples,
    true_raw_data,
    mask_seq,
    rhofold,
    output_dir,
    num_to_letter=NUM_TO_LETTER,
    save_designs=False,
    save_pdbs=False,
    use_relax=False,
):
    """Compute 3D tertiary structure self-consistency using RhoFold.

    Credit: adapted from Rishabh Anand

    This function assesses whether designed RNA sequences are predicted to fold into
    the target 3D structure by using RhoFold. Self-consistency is measured via 
    three complementary metrics:
        - RMSD: Root Mean Square Deviation of C4' atoms (lower is better)
        - TM-score: Template Modeling score for global similarity (higher is better)
        - GDT_TS: Global Distance Test Total Score (higher is better)

    Args:
        samples (np.ndarray or torch.Tensor): Designed sequences [n_samples, seq_len]
        true_raw_data (dict): Dictionary containing reference RNA data:
            - 'sequence': Native sequence
            - 'coords_list': List of coordinate tensors
            - Other metadata for structure comparison
        mask_seq (np.ndarray or torch.Tensor): Boolean mask for valid positions
        rhofold: RhoFold model instance for 3D structure prediction
        output_dir (str): Directory path for saving temporary files and predictions
        num_to_letter (dict): Nucleotide integer-to-letter mapping. Defaults to NUM_TO_LETTER.
        save_designs (bool): Whether to save designed sequences to FASTA files.
            Defaults to False.
        save_pdbs (bool): Whether to save predicted 3D structures as PDB files.
            Useful for visual inspection. Defaults to False.
        use_relax (bool): Whether to apply energy minimization to predicted structures.
            Can improve geometry but increases computation time. Defaults to False.

    Returns:
        tuple: (rmsd_scores, tm_scores, gddt_scores) where each is a list of scores
            per designed sequence.

    Workflow:
        1. Extract reference C4' coordinates from true structure
        2. For each designed sequence:
            a. Convert to nucleotide string and write FASTA
            b. Run RhoFold to predict 3D structure
            c. Extract predicted C4' coordinates
            d. Compute RMSD, TM-score, and GDT_TS against reference
            e. Optionally save PDB file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collate designed sequences in fasta format
    # first record: input sequence and model metadata
    input_seq = SeqRecord(
        Seq(true_raw_data["sequence"]), id=f"input_sequence,", description=f"input_sequence"
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
            description=f"sample={idx}",
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
        coords = get_c4p_coords(coords)[mask_seq, :]
        # zero-center coordinates
        coords = coords - coords.mean(dim=0)

        # Compute self-consistency between designed and groundtruth structures
        _sc_rmsds = []
        _sc_tms = []
        _sc_gddts = []
        for other_coords in true_raw_data["coords_list"]:
            _other = get_c4p_coords(other_coords)[mask_seq, :]
            # zero-center other coordinates
            _other = _other - _other.mean(dim=0)
            # globally align coordinates
            R_hat = rotation_matrix(_other, coords)[0]  # mobile set  # reference set
            _other = _other @ R_hat.T
            # compute metrics
            _sc_rmsds.append(get_rmsd(coords, _other, superposition=True, center=True))
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


##################################
# 3D Structure Comparison Metrics
##################################

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


##################################
# Sequence Comparison Utilities
##################################

def edit_distance(s: str, t: str) -> int:
    """A Space efficient Dynamic Programming based Python3 program to find minimum number
    operations to convert str1 to str2.

    Source: https://www.geeksforgeeks.org/edit-distance-dp-5/
    """
    n = len(s)
    m = len(t)

    prev = [j for j in range(m + 1)]
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1]
            else:
                mn = min(1 + prev[j], 1 + curr[j - 1])
                curr[j] = min(mn, 1 + prev[j - 1])
        prev = curr.copy()

    return prev[m]
