#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

unknown_sub_label = "UNKNOWN_SUBTYPE"
unknown_lineage_label = "UNKNOWN_LINEAGE"
script_dir = Path(__file__).resolve().parent
model_dir = script_dir / "models"

saved_files = {
    "flat_model_json": model_dir / "flat_model_no_type_feature_final.json",
    "flat_label_encoder_pkl": model_dir / "flat_label_encoder.pkl",
    "flat_metadata_json": model_dir / "flat_run_metadata.json",
    "flat_calibration_json": model_dir / "flat_oof_calibration.json",
    "flat_thresholds_json": model_dir / "flat_oof_thresholds.json",

    "lineage_model_json": model_dir / "stage1_lineage_model.json",
    "lineage_label_encoder_pkl": model_dir / "lineage_label_encoder.pkl",
    "lineage_metadata_json": model_dir / "lineage_run_metadata.json",
    "lineage_calibration_json": model_dir / "stage1_oof_calibration.json",
    "lineage_thresholds_json": model_dir / "stage1_oof_thresholds.json",
}

def check_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path

def load_json(path):
    with open(check_file(path), "r", encoding="utf-8") as f:
        return json.load(f)

def load_pickle(path):
    with open(check_file(path), "rb") as f:
        return pickle.load(f)

def load_booster(path, nthread=-1, predictor="cpu_predictor"):
    booster = xgb.Booster()
    booster.load_model(str(check_file(path)))
    booster.set_param({
        "nthread": int(nthread),
        "predictor": predictor,
    })
    return booster

def softmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=1, keepdims=True)
    np.exp(x, out=x)
    x /= np.sum(x, axis=1, keepdims=True)
    return x

def prepare_features(df, feature_columns, fillna_value=None):
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        msg = ", ".join(map(str, missing[:30]))
        if len(missing) > 30:
            msg += f" ... and {len(missing) - 30} more"
        raise ValueError(f"Missing required feature columns: {msg}")

    X = df.loc[:, feature_columns]
    X = X.replace([np.inf, -np.inf], np.nan)

    if fillna_value is not None and X.isnull().values.any():
        X = X.fillna(fillna_value)

    return X.to_numpy(dtype=np.float32, copy=False)

def predict_calibrated_proba_dmatrix_chunked(
    booster,
    X,
    temperature,
    predict_chunk_size=100000,
):
    X = np.asarray(X, dtype=np.float32, order="C")
    n = X.shape[0]

    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    out = None

    for start in range(0, n, predict_chunk_size):
        end = min(start + predict_chunk_size, n)

        dmat = xgb.DMatrix(X[start:end])
        margins = booster.predict(dmat, output_margin=True)
        margins = np.asarray(margins, dtype=np.float32)

        if margins.ndim == 1:
            margins = margins[:, None]

        probs = softmax_np(margins / temperature)

        if out is None:
            out = np.empty((n, probs.shape[1]), dtype=np.float32)

        out[start:end] = probs

    return out

def predict_with_reject_unknown(probs, thresholds_by_class):
    probs = np.asarray(probs, dtype=np.float32)

    pred = np.argmax(probs, axis=1).astype(np.int32)
    max_prob = probs[np.arange(probs.shape[0]), pred]

    thresholds = thresholds_by_class[pred]
    accepted = max_prob >= thresholds

    pred_out = pred.copy()
    pred_out[~accepted] = -1

    return pred_out, max_prob, accepted

def load_thresholds_by_encoder(threshold_json, encoder, key):
    data = load_json(threshold_json)
    thresholds_by_name = data[key]

    return np.array(
        [float(thresholds_by_name.get(name, 1.01)) for name in encoder.classes_],
        dtype=np.float32,
    )

def int_to_labels(pred_int, encoder):
    pred_int = np.asarray(pred_int, dtype=np.int32)
    return encoder.classes_[pred_int]

def pred_to_labels(pred_int, encoder, unknown_label):
    pred_int = np.asarray(pred_int)
    out = np.full(len(pred_int), unknown_label, dtype=object)

    accepted = pred_int >= 0
    if np.any(accepted):
        out[accepted] = encoder.classes_[pred_int[accepted]]

    return out

def load_model_bundle(prefix, nthread=-1, predictor="cpu_predictor"):
    if prefix == "flat":
        threshold_key = "thresholds_by_subtype"
        unknown_label = unknown_sub_label
        files = {
            "model_json": saved_files["flat_model_json"],
            "encoder_pkl": saved_files["flat_label_encoder_pkl"],
            "metadata_json": saved_files["flat_metadata_json"],
            "calibration_json": saved_files["flat_calibration_json"],
            "thresholds_json": saved_files["flat_thresholds_json"],
        }

    elif prefix == "lineage":
        threshold_key = "thresholds_by_lineage"
        unknown_label = unknown_lineage_label
        files = {
            "model_json": saved_files["lineage_model_json"],
            "encoder_pkl": saved_files["lineage_label_encoder_pkl"],
            "metadata_json": saved_files["lineage_metadata_json"],
            "calibration_json": saved_files["lineage_calibration_json"],
            "thresholds_json": saved_files["lineage_thresholds_json"],
        }

    else:
        raise ValueError("prefix must be 'flat' or 'lineage'")

    metadata = load_json(files["metadata_json"])
    encoder = load_pickle(files["encoder_pkl"])
    booster = load_booster(
        files["model_json"],
        nthread=nthread,
        predictor=predictor,
    )
    calibration = load_json(files["calibration_json"])

    return {
        "booster": booster,
        "encoder": encoder,
        "feature_columns": metadata["feature_columns"],
        "temperature": float(calibration["temperature"]),
        "thresholds": load_thresholds_by_encoder(
            files["thresholds_json"],
            encoder,
            threshold_key,
        ),
        "unknown_label": unknown_label,
    }

def run_loaded_model(df, bundle, predict_chunk_size=100000, fillna_value=None):
    X = prepare_features(
        df,
        bundle["feature_columns"],
        fillna_value=fillna_value,
    )

    probs = predict_calibrated_proba_dmatrix_chunked(
        booster=bundle["booster"],
        X=X,
        temperature=bundle["temperature"],
        predict_chunk_size=predict_chunk_size,
    )

    pred_plain = np.argmax(probs, axis=1).astype(np.int32)

    pred_reject, max_confidence, accepted = predict_with_reject_unknown(
        probs,
        bundle["thresholds"],
    )

    return {
        "pred_plain_int": pred_plain,
        "pred_reject_int": pred_reject,
        "pred_plain_label": int_to_labels(pred_plain, bundle["encoder"]),
        "pred_reject_label": pred_to_labels(
            pred_reject,
            bundle["encoder"],
            bundle["unknown_label"],
        ),
        "max_confidence": max_confidence,
        "accepted": accepted,
    }

def make_output_df(df_chunk, keep_input_columns, keep_first_n_columns):
    if keep_input_columns:
        return df_chunk.copy()

    output_df = pd.DataFrame(index=df_chunk.index)

    if keep_first_n_columns > 0:
        meta_cols = df_chunk.columns[:keep_first_n_columns]
        for col in meta_cols:
            output_df[col] = df_chunk[col].values

    return output_df

def process_chunk(
    df_chunk,
    model_type,
    flat_bundle=None,
    lineage_bundle=None,
    chunknown_id=None,
    keep_input_columns=False,
    keep_first_n_columns=2,
    predict_chunk_size=100000,
    fillna_value=None,
):
    output_df = make_output_df(
        df_chunk=df_chunk,
        keep_input_columns=keep_input_columns,
        keep_first_n_columns=keep_first_n_columns,
    )

    flat_result = None
    lineage_result = None

    if model_type in ["flat", "full"]:
        flat_result = run_loaded_model(
            df=df_chunk,
            bundle=flat_bundle,
            predict_chunk_size=predict_chunk_size,
            fillna_value=fillna_value,
        )

        output_df["flat_prediction_plain"] = flat_result["pred_plain_label"]
        output_df["flat_prediction_reject"] = flat_result["pred_reject_label"]
        output_df["flat_accepted"] = flat_result["accepted"].astype(np.int8)
        output_df["flat_confidence"] = flat_result["max_confidence"].astype(np.float32)

    if model_type in ["lineage", "full"]:
        lineage_result = run_loaded_model(
            df=df_chunk,
            bundle=lineage_bundle,
            predict_chunk_size=predict_chunk_size,
            fillna_value=fillna_value,
        )

        output_df["lineage_prediction_plain"] = lineage_result["pred_plain_label"]
        output_df["lineage_prediction_reject"] = lineage_result["pred_reject_label"]
        output_df["lineage_accepted"] = lineage_result["accepted"].astype(np.int8)
        output_df["lineage_confidence"] = lineage_result["max_confidence"].astype(np.float32)

    if model_type == "flat":
        output_df["final_prediction"] = flat_result["pred_reject_label"]
        output_df["final_prediction_level"] = np.where(
            flat_result["accepted"],
            "subtype",
            "unknown",
        )

    elif model_type == "lineage":
        output_df["final_prediction"] = lineage_result["pred_reject_label"]
        output_df["final_prediction_level"] = np.where(
            lineage_result["accepted"],
            "lineage",
            "unknown",
        )

    elif model_type == "full":
        flat_accepted = flat_result["accepted"]
        lineage_accepted = lineage_result["accepted"]

        final_prediction = np.full(len(df_chunk), unknown_sub_label, dtype=object)
        final_level = np.full(len(df_chunk), "unknown", dtype=object)

        final_prediction[flat_accepted] = flat_result["pred_reject_label"][flat_accepted]
        final_level[flat_accepted] = "subtype"

        fallback_mask = ~flat_accepted & lineage_accepted
        final_prediction[fallback_mask] = lineage_result["pred_reject_label"][fallback_mask]
        final_level[fallback_mask] = "lineage"

        output_df["final_prediction"] = final_prediction
        output_df["final_prediction_level"] = final_level

    if chunknown_id is not None:
        output_df["chunknown_id"] = np.int32(chunknown_id)

    return output_df

def load_required_bundles(model_type, nthread=-1, predictor="cpu_predictor"):
    flat_bundle = None
    lineage_bundle = None

    if model_type in ["flat", "full"]:
        flat_bundle = load_model_bundle(
            "flat",
            nthread=nthread,
            predictor=predictor,
        )

    if model_type in ["lineage", "full"]:
        lineage_bundle = load_model_bundle(
            "lineage",
            nthread=nthread,
            predictor=predictor,
        )

    return flat_bundle, lineage_bundle

def update_summary_counts(summary_counts, values):
    unique, counts = np.unique(values, return_counts=True)

    for key, value in zip(unique, counts):
        summary_counts[str(key)] = summary_counts.get(str(key), 0) + int(value)

def process_and_write_chunk(
    df_chunk,
    args,
    flat_bundle,
    lineage_bundle,
    output_path,
    chunknown_id,
    summary_counts,
):
    result_chunk = process_chunk(
        df_chunk=df_chunk,
        model_type=args.model_type,
        flat_bundle=flat_bundle,
        lineage_bundle=lineage_bundle,
        chunknown_id=chunknown_id,
        keep_input_columns=args.keep_input_columns,
        keep_first_n_columns=args.keep_first_n_columns,
        predict_chunk_size=args.predict_chunk_size,
        fillna_value=args.fillna_value,
    )

    result_chunk.to_csv(
        output_path,
        mode="a",
        index=False,
        header=(chunknown_id == 1),
    )

    update_summary_counts(
        summary_counts,
        result_chunk["final_prediction_level"].values,
    )

    return len(df_chunk)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        choices=["flat", "lineage", "full"],
        required=True,
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--input_format", choices=["csv", "excel"], required=True)
    parser.add_argument("--output", default="predictions.csv")

    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--predict_chunk_size", type=int, default=100000)
    parser.add_argument("--nthread", type=int, default=-1)

    parser.add_argument(
        "--predictor",
        choices=["cpu_predictor", "gpu_predictor"],
        default="cpu_predictor",
    )

    parser.add_argument(
        "--fillna_value",
        type=float,
        default=None,
        help="Replace nans with specific value.",
    )

    parser.add_argument(
        "--keep_input_columns",
        action="store_true",
        help="Keep all original input columns in output. Results in large file and slower output.",
    )

    parser.add_argument(
        "--keep_first_n_columns",
        type=int,
        default=2,
        help="Keep selected columns as a metadata.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    if output_path.exists():
        output_path.unlink()

    flat_bundle, lineage_bundle = load_required_bundles(
        model_type=args.model_type,
        nthread=args.nthread,
        predictor=args.predictor,
    )

    total_cells = 0
    chunknown_id = 0
    summary_counts = {}

    if args.input_format == "csv":
        reader = pd.read_csv(
            args.input,
            chunksize=args.chunk_size,
            low_memory=False,
        )

        for df_chunk in reader:
            chunknown_id += 1

            processed = process_and_write_chunk(
                df_chunk=df_chunk,
                args=args,
                flat_bundle=flat_bundle,
                lineage_bundle=lineage_bundle,
                output_path=output_path,
                chunknown_id=chunknown_id,
                summary_counts=summary_counts,
            )

            total_cells += processed
            print(f"Processed chunk {chunknown_id}: {processed} cells", flush=True)

    elif args.input_format == "excel":
        df = pd.read_excel(args.input)

        for start in range(0, len(df), args.chunk_size):
            chunknown_id += 1
            df_chunk = df.iloc[start:start + args.chunk_size]

            processed = process_and_write_chunk(
                df_chunk=df_chunk,
                args=args,
                flat_bundle=flat_bundle,
                lineage_bundle=lineage_bundle,
                output_path=output_path,
                chunknown_id=chunknown_id,
                summary_counts=summary_counts,
            )

            total_cells += processed
            print(f"Processed chunk {chunknown_id}: {processed} cells", flush=True)

    print(f"Predictions saved to: {output_path}")
    print(f"Total processed cells: {total_cells}")
    print(f"Total chunks: {chunknown_id}")
    print("Final prediction level counts:")

    for key, value in summary_counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()