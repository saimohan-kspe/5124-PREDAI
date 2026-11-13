"""CLI script for building and cleaning market dataset."""
import argparse
from pathlib import Path
import pandas as pd

from src.data.cleaning import clean_market_dataframe, save_quality_report
from src.data.feeds import ParquetFeed, SyntheticFeed


def save_dataframe(df: pd.DataFrame, base_path: Path) -> Path:
    """Saves DataFrame as Parquet if engine available, otherwise CSV."""
    try:
        parquet_path = base_path.with_suffix(".parquet")
        df.to_parquet(parquet_path)
        return parquet_path
    except Exception:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and clean dataset for ASTRAQUANT PS-12")
    parser.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "parquet", "csv"])
    parser.add_argument("--input", type=str, default=None, help="Input file path for CSV/Parquet")
    parser.add_argument("--out", type=str, default="artifacts/data", help="Output directory")
    parser.add_argument("--sessions", type=int, default=60, help="Number of synthetic sessions")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic feed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "synthetic":
        feed = SyntheticFeed(num_sessions=args.sessions, seed=args.seed)
        raw_df = feed.get_all_bars()
        proxies_df = feed.get_proxies()
        save_dataframe(proxies_df, out_dir / "proxies")
    else:
        if not args.input:
            raise ValueError("--input path required when using csv or parquet source")
        feed = ParquetFeed(args.input)
        raw_df = feed.get_all_bars()

    cleaned_df, report = clean_market_dataframe(raw_df)
    saved_path = save_dataframe(cleaned_df, out_dir / "nifty50_cleaned")

    json_rep, md_rep = save_quality_report(report, str(out_dir))
    print(f"Dataset successfully built at: {saved_path}")
    print(f"Data quality report saved to: {json_rep} and {md_rep}")
    print(f"Total rows: {len(cleaned_df)} | Total sessions: {cleaned_df['session_id'].nunique()}")


if __name__ == "__main__":
    main()
