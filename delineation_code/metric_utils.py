import statistics
import pandas as pd


def metrics_calculation(
    predictions, targets, window_tolerance, file_paths, df_save_path
):
    """
    Calculates detection metrics (Se, PPV, F1, mean error, std error) for each ECG delineation point.

    Args:
        predictions (list): List of predicted segments for each file.
        targets (list): List of ground truth segments for each file.
        window_tolerance (int): Allowed error in samples for a match.
        file_paths (list): List of file paths (for error reporting).
        df_save_path (str): Directory to save the CSV with false positives/negatives.

    Returns:
        dict: Dictionary with metrics for each delineation point.
    """
    points = ["P_onset", "P_offset", "QRS_onset", "QRS_offset", "T_onset", "T_offset"]
    result = {k: {"m": [], "sigma": [], "Se": [], "PPV": [], "F1": []} for k in points}

    label_to_search = {"P": 0, "Q": 1, "T": 2}
    df_with_false_predictions = pd.DataFrame(
        columns=["file_path", "label", "targets", "value"]
    )

    for one_label in points:
        error_list = []
        overall_tp = overall_fp = overall_fn = 0
        for index in range(len(predictions)):
            tp = fp = fn = 0
            pred = predictions[index]
            true_labels = targets[index]
            # Select only segments of the current label (P, Q, or T)
            one_label_pred = [
                label for label in pred if label[0] == label_to_search[one_label[0]]
            ]
            one_label_true = [
                label
                for label in true_labels
                if label[0] == label_to_search[one_label[0]]
            ]
            for label in one_label_pred:
                # For onset, compare start; for offset, compare end
                if "onset" in one_label:
                    label_find, sum_error = check_label(
                        label[1], one_label_true, window_tolerance, 1
                    )
                else:
                    label_find, sum_error = check_label(
                        label[2], one_label_true, window_tolerance, 2
                    )
                if label_find:
                    one_label_true.remove(label_find)
                    error_list.append(sum_error)
                    tp += 1
                else:
                    # False positive
                    fp_row = pd.DataFrame(
                        {
                            "file_path": [file_paths[index]],
                            "label": [label],
                            "targets": [true_labels],
                            "value": ["fp"],
                        }
                    )
                    df_with_false_predictions = pd.concat(
                        [df_with_false_predictions, fp_row], ignore_index=True
                    )
                    fp += 1
            # Remaining true labels are false negatives
            fn = len(one_label_true)
            for label in one_label_true:
                fp_row = pd.DataFrame(
                    {
                        "file_path": [file_paths[index]],
                        "label": [label],
                        "targets": [true_labels],
                        "value": ["fn"],
                    }
                )
                df_with_false_predictions = pd.concat(
                    [df_with_false_predictions, fp_row], ignore_index=True
                )
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

        print(
            f"Overall statistics for {one_label}: tp = {overall_tp}, fp = {overall_fp}, fn = {overall_fn}"
        )
        if overall_tp:
            ppv = overall_tp / (overall_tp + overall_fp)
            sensitivity = overall_tp / (overall_tp + overall_fn)
            f1 = (
                (2 * ppv * sensitivity) / (ppv + sensitivity)
                if (ppv + sensitivity) > 0
                else 0
            )
            if len(error_list) > 2:
                mean_err = statistics.mean(error_list)
                std_err = statistics.stdev(error_list)
                print(
                    f"Overall statistics for {one_label}: sens = {sensitivity:.3f}, ppv = {ppv:.3f}, "
                    f"f1-score = {f1:.3f}, m+-s= {mean_err:.1f}+/-{std_err:.1f}"
                )
            else:
                print(
                    f"Overall statistics for {one_label}: sens = {sensitivity:.3f}, ppv = {ppv:.3f}, "
                    f"f1-score = {f1:.3f}"
                )
    df_with_false_predictions.to_csv(df_save_path + "/fp_fn_labels.csv", index=False)
    return result


def check_label(label, one_label_true, window_tolerance, onset_ix):
    """
    Checks if a label (onset or offset) matches any in the true list within the tolerance.

    Args:
        label (int): The predicted onset or offset.
        one_label_true (list): List of true segments for this label.
        window_tolerance (int): Allowed error in samples.
        onset_ix (int): 1 for onset, 2 for offset.

    Returns:
        tuple: (matched_label, error) or (None, None) if not found.
    """
    if label:
        for l in one_label_true:
            if abs(l[onset_ix] - label) < window_tolerance:
                return l, l[onset_ix] - label
    return None, None


def stopping_criterion(val_loss, best_metrics, epochs_no_improve):
    """
    Early stopping criterion: increases counter if no improvement, resets if improved.

    Args:
        val_loss (float): Current validation loss.
        best_metrics (float): Best (lowest) validation loss so far.
        epochs_no_improve (int): Number of epochs without improvement.

    Returns:
        tuple: (epochs_no_improve, best_metrics)
    """
    if val_loss >= best_metrics:
        epochs_no_improve += 1
        return epochs_no_improve, best_metrics
    best_metrics = val_loss
    epochs_no_improve = 0
    return epochs_no_improve, best_metrics
