import torch


def calculate_mask(target, cfg):
    """
    Converts a raw target mask into a list of merged ECG segments and confidence scores.

    Args:
        target (Tensor): The raw annotation target.
        cfg (object): Configuration object with necessary parameters.

    Returns:
        tuple: (segments, confidence)
            - segments: List of merged segments (label, start, end).
            - confidence: List of confidence values for each label.
    """
    confidence = ""
    if cfg.model_params.model_name in ("UNetECG", "SegNetECG"):
        segments = unet_to_segments(target)
        segments = unite_segments(segments, cfg)
        return segments, confidence

    mask, confidence = convert_to_segments(target, cfg)
    sequences = find_sequences(mask)
    merged_segments = unite_segments(sequences, cfg)
    return merged_segments, confidence


def unet_to_segments(target):
    """
    Convert UNet/SegNet 4-channel mask/logits to (label, start, end) segments.
    Uses the first 3 classes only: P(0), QRS(1), T(2).
    """
    result = []

    target = target[:3]
    if torch.is_floating_point(target):
        if target.min() < 0 or target.max() > 1:
            target = torch.sigmoid(target)
        target = (target > 0.5).to(torch.int32)
    else:
        target = target.to(torch.int32)

    diff = torch.diff(target, dim=1, prepend=torch.zeros_like(target[:, :1]))

    starts = torch.where(diff == 1)
    ends = torch.where(diff == -1)

    for row, start in zip(starts[0], starts[1]):
        row_ends = ends[1][(ends[0] == row) & (ends[1] > start)]
        end = row_ends.min().item() if len(row_ends) > 0 else target.shape[1]
        result.append((row.item(), start.item(), end))

    return sorted(result, key=lambda x: x[1])


def convert_to_segments(segment_annotation_target, cfg):
    """
    Converts a YOHO interval target to a 1D mask and computes confidence scores.
    """
    ecg_length = (
        cfg.ecg_record_params.observed_ecg_length
        * cfg.ecg_record_params.resampled_frequency
    )
    mask = torch.zeros(ecg_length)
    segment_annotation_target = segment_annotation_target[:9].T
    intervals = torch.linspace(0, ecg_length, segment_annotation_target.shape[0] + 1)
    intervals = list(zip(intervals[:-1], intervals[1:]))
    interval_len = intervals[0][1] - intervals[0][0]

    for interval_ind, (interval_start, interval_end) in enumerate(intervals):
        yoho_interval = segment_annotation_target[interval_ind]
        for peak_ind in range(3):
            if yoho_interval[peak_ind * 3] > 0.5:
                seg_start = int(
                    interval_start + yoho_interval[peak_ind * 3 + 1] * interval_len
                )
                seg_end = int(
                    interval_start + yoho_interval[peak_ind * 3 + 2] * interval_len
                )
                mask[seg_start:seg_end] = peak_ind + 1

    confidence = []
    for i in range(3):
        label_vals = segment_annotation_target[i * 3]
        confidence.append(torch.mean(torch.abs(label_vals - 0.5)).cpu().item())
        confidence.append(torch.std(torch.abs(label_vals - 0.5)).cpu().item())

    return mask, confidence


def find_sequences(mask):
    sequences = []
    start = 0
    for i in range(1, len(mask)):
        if mask[i] != mask[i - 1]:
            if mask[i - 1] != 0:
                sequences.append((int(mask[i - 1]) - 1, start, i))
            start = i
    if mask[-1] != 0:
        sequences.append((int(mask[-1]) - 1, start, len(mask)))
    return sequences


def unite_segments(segment_list, cfg):
    merged_segments = []
    current_label = None
    current_start = None
    current_end = None

    for label, start, end in segment_list:
        if current_label is None:
            current_label = label
            current_start = start
            current_end = end
        elif (
            label == current_label
            and start - current_end <= cfg.eval_params.gap_to_fill_dist
        ):
            current_end = end
        else:
            if current_end - current_start >= cfg.eval_params.interval_len_to_remove:
                merged_segments.append((current_label, current_start, current_end))
            current_label = label
            current_start = start
            current_end = end

    if (
        current_label is not None
        and current_end - current_start >= cfg.eval_params.interval_len_to_remove
    ):
        merged_segments.append((current_label, current_start, current_end))

    return merged_segments


def remove_overlaps(intervals, priority_map={0: 1, 1: 2, 2: 0}):
    removal_indices = []
    for i in range(len(intervals) - 1):
        if (
            intervals[i][2] > intervals[i + 1][1]
            and intervals[i][1] < intervals[i + 1][2]
        ):
            if priority_map[intervals[i][0]] < priority_map[intervals[i + 1][0]]:
                removal_indices.append(i)
            else:
                removal_indices.append(i + 1)
    return [
        interval
        for index, interval in enumerate(intervals)
        if index not in removal_indices
    ]
