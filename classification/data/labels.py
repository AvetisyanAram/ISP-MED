from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd

PTB_XL_SCP_CODES = [
    "NDT",
    "NST_",
    "DIG",
    "LNGQT",
    "NORM",
    "IMI",
    "ASMI",
    "LVH",
    "LAFB",
    "ISC_",
    "IRBBB",
    "1AVB",
    "IVCD",
    "ISCAL",
    "CRBBB",
    "CLBBB",
    "ILMI",
    "LAO/LAE",
    "AMI",
    "ALMI",
    "ISCIN",
    "INJAS",
    "LMI",
    "ISCIL",
    "LPFB",
    "ISCAS",
    "INJAL",
    "ISCLA",
    "RVH",
    "ANEUR",
    "RAO/RAE",
    "EL",
    "WPW",
    "ILBBB",
    "IPLMI",
    "ISCAN",
    "IPMI",
    "SEHYP",
    "INJIN",
    "INJLA",
    "PMI",
    "3AVB",
    "INJIL",
    "2AVB",
    "ABQRS",
    "PVC",
    "STD_",
    "VCLVH",
    "QWAVE",
    "LOWT",
    "NT_",
    "PAC",
    "LPR",
    "INVT",
    "LVOLT",
    "HVOLT",
    "TAB_",
    "STE_",
    "PRC(S)",
    "SR",
    "AFIB",
    "STACH",
    "SARRH",
    "SBRAD",
    "PACE",
    "SVARR",
    "BIGU",
    "AFLT",
    "SVTAC",
    "PSVT",
    "TRIGU",
]


PTB_SUPER_PRESET = {
    "preset_name": "ptb_super",
    "pathology_names": [
        "NDT",
        "NST_",
        "DIG",
        "LNGQT",
        "NORM",
        "IMI",
        "ASMI",
        "LVH",
        "LAFB",
        "ISC_",
        "IRBBB",
        "1AVB",
        "IVCD",
        "ISCAL",
        "CRBBB",
        "CLBBB",
        "ILMI",
        "LAO/LAE",
        "AMI",
        "ALMI",
        "ISCIN",
        "INJAS",
        "LMI",
        "ISCIL",
        "LPFB",
        "ISCAS",
        "INJAL",
        "ISCLA",
        "RVH",
        "ANEUR",
        "RAO/RAE",
        "EL",
        "WPW",
        "ILBBB",
        "IPLMI",
        "ISCAN",
        "IPMI",
        "SEHYP",
        "INJIN",
        "INJLA",
        "PMI",
        "3AVB",
        "INJIL",
        "2AVB",
    ],
    "merge_map": {
        "STTC": [
            "NDT",
            "NST_",
            "DIG",
            "LNGQT",
            "ISC_",
            "ISCAL",
            "ISCIN",
            "ISCIL",
            "ISCAS",
            "ISCLA",
            "ANEUR",
            "EL",
            "ISCAN",
        ],
        "NORM": ["NORM"],
        "MI": [
            "IMI",
            "ASMI",
            "ILMI",
            "AMI",
            "ALMI",
            "INJAS",
            "LMI",
            "INJAL",
            "IPLMI",
            "IPMI",
            "INJIN",
            "INJLA",
            "PMI",
            "INJIL",
        ],
        "HYP": ["LVH", "LAO/LAE", "RVH", "RAO/RAE", "SEHYP"],
        "CD": [
            "LAFB",
            "IRBBB",
            "1AVB",
            "IVCD",
            "CRBBB",
            "CLBBB",
            "LPFB",
            "WPW",
            "ILBBB",
            "3AVB",
            "2AVB",
        ],
    },
}

PTB_SUB_PRESET = {
    "preset_name": "ptb_sub",
    "pathology_names": [
        "NDT",
        "NST_",
        "DIG",
        "LNGQT",
        "NORM",
        "IMI",
        "ASMI",
        "LVH",
        "LAFB",
        "ISC_",
        "IRBBB",
        "1AVB",
        "IVCD",
        "ISCAL",
        "CRBBB",
        "CLBBB",
        "ILMI",
        "LAO/LAE",
        "AMI",
        "ALMI",
        "ISCIN",
        "INJAS",
        "LMI",
        "ISCIL",
        "LPFB",
        "ISCAS",
        "INJAL",
        "ISCLA",
        "RVH",
        "ANEUR",
        "RAO/RAE",
        "EL",
        "WPW",
        "ILBBB",
        "IPLMI",
        "ISCAN",
        "IPMI",
        "SEHYP",
        "INJIN",
        "INJLA",
        "PMI",
        "3AVB",
        "INJIL",
        "2AVB",
    ],
    "merge_map": {
        "NORM": ["NORM"],
        "STTC": ["NDT", "DIG", "LNGQT", "ANEUR", "EL"],
        "NST_": ["NST_"],
        "IMI": ["IMI", "ILMI", "IPLMI", "IPMI", "INJIN", "INJIL"],
        "AMI": ["ASMI", "AMI", "ALMI", "INJAS", "INJAL", "INJLA"],
        "LVH": ["LVH"],
        "LAFB_LPFB": ["LAFB", "LPFB"],
        "ISC_": ["ISC_"],
        "IRBBB": ["IRBBB"],
        "_AVB": ["1AVB", "2AVB", "3AVB"],
        "IVCD": ["IVCD"],
        "ISCA": ["ISCAL", "ISCAS", "ISCAN", "ISCLA"],
        "CRBBB": ["CRBBB"],
        "CLBBB": ["CLBBB"],
        "LAO_LAE": ["LAO/LAE"],
        "ISCI": ["ISCIN", "ISCIL"],
        "LMI": ["LMI"],
        "RVH": ["RVH"],
        "RAO_RAE": ["RAO/RAE"],
        "WPW": ["WPW"],
        "ILBBB": ["ILBBB"],
        "SEHYP": ["SEHYP"],
        "PMI": ["PMI"],
    },
}

MAJOR_ISP_PATHOLOGIES = [
    "SR",
    "AXNOR",
    "LAD",
    "SBRAD",
    "AFIB",
    "AXHOR",
    "PVC",
    "IRBBB",
    "LAFB",
    "1AVB",
    "STACH",
    "AXVER",
    "CRBBB",
    "INVT",
    "QWAVE",
    "CLBBB",
    "SARRH",
    "LOWT",
    "ARHYT",
    "POORR",
    "SVPC",
    "AXIND",
    "RAD",
    "AFLT",
    "STD_",
    "EPVC",
    "CWRT",
    "BRADO",
    "VBIG",
    "PAC",
    "IVCD",
    "WPW",
    "SVPCS",
    "ASMI",
    "LMI",
]

USER_DEFINED_ISP_PTBLXL_INTERSECTION = [
    pathology for pathology in MAJOR_ISP_PATHOLOGIES if pathology in PTB_XL_SCP_CODES
]

PTB_ISP_PRESET = {
    "preset_name": "ptb_isp",
    # User-defined ISP pathology list filtered to labels that exist in PTB-XL SCP codes.
    # The original order is preserved.
    "pathology_names": USER_DEFINED_ISP_PTBLXL_INTERSECTION,
    "merge_map": {},
}

ISP_1_PRESET = {
    "preset_name": "isp_1",
    "pathology_names": [
        pathology
        for pathology in USER_DEFINED_ISP_PTBLXL_INTERSECTION
        if pathology not in ["IVCD", "WPW", "AFLT"]
    ],
    "merge_map": {},
}


PRESET_REGISTRY = {
    PTB_SUB_PRESET["preset_name"]: PTB_SUB_PRESET,
    PTB_SUPER_PRESET["preset_name"]: PTB_SUPER_PRESET,
    PTB_ISP_PRESET["preset_name"]: PTB_ISP_PRESET,
    ISP_1_PRESET["preset_name"]: ISP_1_PRESET,
}


def parse_label_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return [stripped]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, dict):
            return [str(key).strip() for key in parsed if str(key).strip()]
        return [str(parsed).strip()]
    if isinstance(value, dict):
        return [str(key).strip() for key in value if str(key).strip()]
    return [str(value).strip()]


def normalize_labels(labels: list[str]) -> list[str]:
    return [label.strip() for label in labels if label and label.strip()]


def merge_label_frame(
    frame: pd.DataFrame, merge_map: dict[str, list[str]]
) -> pd.DataFrame:
    merged = frame.copy()
    for new_label, source_labels in merge_map.items():
        existing_labels = [label for label in source_labels if label in merged.columns]
        if not existing_labels:
            merged[new_label] = 0
            continue
        merged[new_label] = merged[existing_labels].max(axis=1)
        drop_labels = [label for label in existing_labels if label != new_label]
        if drop_labels:
            merged = merged.drop(columns=drop_labels)
    return merged


def build_target_matrix(
    label_series: pd.Series,
    pathology_names: list[str],
    merge_map: dict[str, list[str]] | None = None,
) -> np.ndarray:
    parsed = label_series.apply(parse_label_list)
    extra_labels = sorted(
        {label for labels in parsed for label in labels if label not in pathology_names}
    )
    all_labels = list(pathology_names) + extra_labels
    label_to_index = {label: index for index, label in enumerate(all_labels)}
    target_matrix = np.zeros((len(parsed), len(all_labels)), dtype=np.float32)
    for row_index, labels in enumerate(parsed):
        for label in labels:
            column_index = label_to_index.get(label)
            if column_index is not None:
                target_matrix[row_index, column_index] = 1.0
    target_frame = pd.DataFrame(
        target_matrix, index=label_series.index, columns=all_labels, dtype=np.float32
    )
    if merge_map:
        target_frame = merge_label_frame(target_frame, merge_map)
    target_frame = target_frame.reindex(
        columns=ordered_labels(pathology_names, merge_map), fill_value=0
    )
    return target_frame.to_numpy(dtype=np.float32)


def ordered_labels(
    pathology_names: list[str], merge_map: dict[str, list[str]] | None = None
) -> list[str]:
    if merge_map:
        return list(merge_map.keys())
    return list(pathology_names)


def get_pathology_preset(name: str) -> dict:
    if name not in PRESET_REGISTRY:
        raise KeyError(
            f"Unknown pathology preset '{name}'. Available presets: {sorted(PRESET_REGISTRY)}"
        )
    preset = PRESET_REGISTRY[name]
    return {
        "preset_name": preset["preset_name"],
        "pathology_names": list(preset["pathology_names"]),
        "merge_map": dict(preset["merge_map"]),
        "target_labels": ordered_labels(preset["pathology_names"], preset["merge_map"]),
    }


def collect_label_counts(label_series: pd.Series) -> Counter:
    counts: Counter = Counter()
    for labels in label_series.apply(parse_label_list):
        counts.update(labels)
    return counts


def compute_overlap_labels(
    reference_labels: list[str], observed_label_series: pd.Series
) -> list[str]:
    observed = set(collect_label_counts(observed_label_series))
    return [label for label in reference_labels if label in observed]
