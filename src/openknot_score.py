import numpy as np
from arnie.utils import get_helices


def identify_crossing_bps(bps):
    """
    Identifies all bases involved in crossing base pairs.
    
    Args:
        bps (list of list of int): A list of base pair tuples, where each tuple contains two integers representing the positions of the paired bases (e.g., [[11, 17], [10, 18], [9, 19], ...]).
    
    Returns:
        list of int: A list of unique indexes of bases involved in crossing pairs. If no crossing pairs are found, an empty list is returned.
    """
    crossed_res = []
    # Check every base pair in the list
    for bp in bps:
        # The first value in the base pair should be less than the second value
        if (bp[0] < bp[1]):
            # Now we loop over the list again, grabbing base pairs to compare
            for next_bp in bps:
                # This checks whether the original bp and the compared bp cross
                # If they do, we add base positions to the crossed residue list
                if (bp[0] < next_bp[0] & next_bp[0] < bp[1] & bp[1] < next_bp[1]) | ( next_bp[0] < bp[0] & bp[0] < next_bp[1] & next_bp[1] < bp[1]):
                    crossed_res.append(bp[0])
                    crossed_res.append(bp[1])
                    crossed_res.append(next_bp[0])
                    crossed_res.append(next_bp[1])
                    
    # Returns a unique list, since crossed residues may contain duplicate values
    return list(set(crossed_res)) if len(crossed_res) else []


def remove_bps_in_blanked_regions(bps, num_res, BLANK_OUT5, BLANK_OUT3):
    filtered_bps = []
    for bp in bps:
        if (bp[0] < bp[1]):
            if (bp[0] <= BLANK_OUT5): continue
            if (bp[1] <= BLANK_OUT5): continue
            if (bp[0] > num_res - BLANK_OUT3): continue
            if (bp[1] > num_res - BLANK_OUT3): continue
            filtered_bps.append(bp)
    return filtered_bps


def get_openknot_score(
        target_sec_struct, 
        reactivity_data_arr, 
        threshold_SHAPE_fixed = 0.5,
        threshold_SHAPE_fixed_pair = 0.25,
        min_SHAPE_fixed = 0.0,
        BLANK_OUT5=0, 
        BLANK_OUT3=0
    ):
    """
    Calculates the OpenKnot score for a DBN structure and multiple accompanying datasets

    Credit: adapted from https://github.com/eternagame/OpenKnotScorePipeline (serialized version)
    
    crossed_pair_score = 
    100 * (number of residues in crossed pairs with data < 0.25) /
        [0.7*(length of region with data - 20)]

    crossed_pair_quality_score =
    100 * (number of residues in crossed pairs with data < 0.25) /
        ( number of residues modeled to be crossed pairs in structure )

    openknot_score = 0.5 * eterna_score + 0.5 * crossed_pair_quality_score
           
    Args:
        target_sec_struct: str, target secondary structure with pseudoknots
        reactivity_data_arr: np.array, array of predicted 2A3 chemical modifications
        threshold_SHAPE_fixed: float, threshold value for SHAPE data
        threshold_SHAPE_fixed_pair: float, threshold value for SHAPE data
        min_SHAPE_fixed: float, minimum value threshold of paired or unpaired in SHAPE data
        BLANK_OUT5: int, number of bases to blank out at the 5' end
        BLANK_OUT3: int, number of bases to blank out at the 3' end
    
    Returns:
        list of tuple: A list of tuples, where each tuple contains:
            eterna_score: float, Eterna score
            crossed_pair_score: float, crossed pair score
            crossed_pair_quality_score: float, crossed pair quality score
            openknot_score: float, openknot score
    """
    # Convert the input structure to a binary paired/unpaired list
    # We only have trustworthy data for the bases in the middle of the sequence
    # so we skip the blanked regions at the beginning and end
    prediction = np.array([1 if char == "." else 0 for char in target_sec_struct][BLANK_OUT5:(len(target_sec_struct) - BLANK_OUT3)])
    
    # Calculate correct hits for unpaired bases
    unpaired_hits = (prediction == 1) & (reactivity_data_arr > (0.25 * threshold_SHAPE_fixed + 0.75 * min_SHAPE_fixed))
    
    # Calculate correct hits for paired bases
    paired_hits = (prediction == 0) & (reactivity_data_arr < threshold_SHAPE_fixed)
    
    # Sum correct hits
    correct_hits = unpaired_hits | paired_hits
    
    # Calculate Eterna scores
    eterna_scores = (np.sum(correct_hits, axis=1) / reactivity_data_arr.shape[1]) * 100

    padded_data_arr = np.pad(
        reactivity_data_arr,
        ((0, 0), (BLANK_OUT5, BLANK_OUT3)),
        constant_values=np.nan,
    )
    
    # Some algorithms will produce a string of all x's when they fail on a sequence
    # This checks for a structure string that is all x characters and returns if true
    failed_structure = [char == "x" for char in target_sec_struct]
    if all(failed_structure):
        return [[0, 0] for _ in reactivity_data_arr]

    # Remove singlet base pairs
    stems = get_helices(target_sec_struct)
    stems = [stem for stem in stems if len(stem) > 1]
    # Convert the list of helices into a list of base pairs (flatten 3D list to 2D list)
    bp_list = [item for sublist in stems for item in sublist]
    
    # Get indexes for bases in crossed pairs
    crossed_res = identify_crossing_bps(bp_list)

    # Filter out base pairs that involve residues in the blanked out flanking regions
    bps_filtered = remove_bps_in_blanked_regions(bp_list, len(target_sec_struct), BLANK_OUT5, BLANK_OUT3)
    crossed_res_filtered = identify_crossing_bps(bps_filtered)
    
    crossed_res = np.array(crossed_res)
    crossed_res_filtered = np.array(crossed_res_filtered)
    
    max_count = np.sum((crossed_res >= BLANK_OUT5) & (crossed_res < len(target_sec_struct) - BLANK_OUT3))

    if len(crossed_res) == 0:
        num_crossed_pairs = np.zeros(reactivity_data_arr.shape[0])
    else:
        num_crossed_pairs = np.sum((padded_data_arr[:, crossed_res] < threshold_SHAPE_fixed_pair) & 
                                np.isin(crossed_res, crossed_res_filtered), axis=1) + \
                            0.5 * np.sum((padded_data_arr[:, crossed_res] < threshold_SHAPE_fixed_pair) & 
                                        ~np.isin(crossed_res, crossed_res_filtered), axis=1)
        
    data_region_length = padded_data_arr.shape[1] - BLANK_OUT5 - BLANK_OUT3
    max_crossed_pairs = 0.7 * np.maximum(data_region_length - 20, 20)
    crossed_pair_score = 100 * np.minimum(num_crossed_pairs / max_crossed_pairs, 1.0)
    
    crossed_pair_quality_score = 100 * (num_crossed_pairs / max_count) if max_count > 0 else 0

    openknot_score = 0.5 * eterna_scores + 0.5 * crossed_pair_quality_score
    
    return eterna_scores, crossed_pair_score, crossed_pair_quality_score, openknot_score
