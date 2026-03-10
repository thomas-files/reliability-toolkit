"""
aws_pipeline.py
---------------
AWS S3 + Athena pipeline for reliability data ingestion and assessment.

Architecture
------------
    S3 Bucket (raw CSVs)
        ↓
    AthenaClient  — SQL queries against field data
        ↓
    PipelineRunner — pulls queried data → runs ReliabilityReport → writes results to S3

Mock Mode
---------
    Set MOCK=True (default) to run entirely locally with the same interface.
    To use real AWS, set MOCK=False and configure credentials:
        export AWS_ACCESS_KEY_ID=...
        export AWS_SECRET_ACCESS_KEY=...
        export AWS_DEFAULT_REGION=us-east-1

Real AWS Setup (when ready)
---------------------------
    1. Create an S3 bucket: aws s3 mb s3://your-reliability-bucket
    2. Create an Athena database:
           CREATE DATABASE reliability_db;
    3. Create the failures table (see CREATE_TABLE_SQL below)
    4. Set MOCK=False and update BUCKET_NAME / DATABASE below

Dependencies
------------
    pip install boto3 pandas numpy
    (boto3 only needed for real AWS mode)
"""

import os
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────

MOCK = True                          # Set False for real AWS
BUCKET_NAME = "reliability-data"     # Your S3 bucket name
DATABASE = "reliability_db"          # Athena database name
RESULTS_PREFIX = "results/"          # S3 prefix for output CSVs
RAW_PREFIX = "raw/failures/"         # S3 prefix for input CSVs

# Athena DDL — run this once to register your S3 data as a queryable table
CREATE_TABLE_SQL = f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE}.failures (
    component_id  STRING,
    component_type STRING,
    tool_id       STRING,
    site          STRING,
    install_date  STRING,
    failure_date  STRING,
    time_to_failure DOUBLE,
    failed        BOOLEAN,
    failure_mode  STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://{BUCKET_NAME}/{RAW_PREFIX}'
TBLPROPERTIES ('skip.header.line.count'='1');
"""


# ── Mock S3 filesystem ───────────────────────────────────────────────

class MockS3:
    """
    Local filesystem mock of AWS S3.
    Stores files in a temp directory with the same interface as boto3 S3.
    Swap for real boto3 client by setting MOCK=False.
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or tempfile.mkdtemp(prefix="mock_s3_"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MockS3] Bucket root: {self.base_dir}")

    def upload(self, local_path: str, s3_key: str):
        dest = self.base_dir / s3_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(local_path, dest)
        print(f"[MockS3] Uploaded: {s3_key}")

    def download(self, s3_key: str, local_path: str):
        src = self.base_dir / s3_key
        if not src.exists():
            raise FileNotFoundError(f"Key not found: {s3_key}")
        import shutil
        shutil.copy2(src, local_path)

    def list_objects(self, prefix: str = "") -> list:
        results = []
        for f in self.base_dir.rglob("*.csv"):
            key = str(f.relative_to(self.base_dir))
            if key.startswith(prefix):
                results.append({"Key": key, "Size": f.stat().st_size})
        return results

    def read_csv(self, s3_key: str) -> pd.DataFrame:
        path = self.base_dir / s3_key
        return pd.read_csv(path)

    def write_csv(self, df: pd.DataFrame, s3_key: str):
        dest = self.base_dir / s3_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        print(f"[MockS3] Written: {s3_key}")


class RealS3:
    """
    Real AWS S3 client wrapper. Requires boto3 and valid credentials.
    Usage: set MOCK=False to use this instead of MockS3.
    """

    def __init__(self, bucket: str):
        try:
            import boto3
            self.s3 = boto3.client("s3")
            self.bucket = bucket
        except ImportError:
            raise ImportError("boto3 not installed. Run: pip install boto3")

    def upload(self, local_path: str, s3_key: str):
        self.s3.upload_file(local_path, self.bucket, s3_key)
        print(f"[S3] Uploaded s3://{self.bucket}/{s3_key}")

    def download(self, s3_key: str, local_path: str):
        self.s3.download_file(self.bucket, s3_key, local_path)

    def list_objects(self, prefix: str = "") -> list:
        paginator = self.s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            results.extend(page.get("Contents", []))
        return results

    def read_csv(self, s3_key: str) -> pd.DataFrame:
        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return pd.read_csv(obj["Body"])

    def write_csv(self, df: pd.DataFrame, s3_key: str):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            self.upload(f.name, s3_key)
        os.unlink(f.name)


# ── Mock Athena engine ───────────────────────────────────────────────

class MockAthena:
    """
    Local SQL engine that mimics AWS Athena using pandas + DuckDB-style queries.
    Queries run against in-memory DataFrames loaded from the mock S3 bucket.

    Supports the same SQL interface you'd use with real Athena:
        SELECT, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT
        DATE_DIFF, AVG, COUNT, MIN, MAX, PERCENTILE_APPROX
    """

    def __init__(self, s3: MockS3, raw_prefix: str = RAW_PREFIX):
        self.s3 = s3
        self.raw_prefix = raw_prefix
        self._table = None

    def _load_table(self) -> pd.DataFrame:
        """Load all CSVs from the raw prefix into a single DataFrame."""
        if self._table is not None:
            return self._table

        keys = self.s3.list_objects(prefix=self.raw_prefix)
        if not keys:
            raise ValueError(f"No CSVs found under prefix: {self.raw_prefix}")

        dfs = []
        for obj in keys:
            df = self.s3.read_csv(obj["Key"])
            dfs.append(df)

        self._table = pd.concat(dfs, ignore_index=True)
        print(f"[MockAthena] Loaded {len(self._table)} rows from {len(keys)} files")
        return self._table

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query against the failures table.
        Uses pandas for filtering/aggregation to mirror Athena behavior.
        """
        df = self._load_table()
        sql_clean = sql.strip().upper()
        print(f"[MockAthena] Running query...")

        # Route to appropriate handler based on query type
        if "GROUP BY component_type".upper() in sql_clean:
            return self._query_by_component_type(df, sql)
        elif "GROUP BY tool_id".upper() in sql_clean:
            return self._query_by_tool(df, sql)
        elif "GROUP BY site".upper() in sql_clean:
            return self._query_by_site(df, sql)
        elif "WHERE" in sql_clean and "component_type" in sql_clean.lower():
            return self._filter_component_type(df, sql)
        elif "ORDER BY time_to_failure".upper() in sql_clean:
            return self._early_failures(df, sql)
        else:
            # Default: return filtered rows
            return self._generic_filter(df, sql)

    def _query_by_component_type(self, df, sql) -> pd.DataFrame:
        failed = df[df["failed"] == True]
        result = failed.groupby("component_type").agg(
            n_failures=("time_to_failure", "count"),
            avg_ttf=("time_to_failure", "mean"),
            min_ttf=("time_to_failure", "min"),
            max_ttf=("time_to_failure", "max"),
            p10_ttf=("time_to_failure", lambda x: np.percentile(x, 10)),
            p50_ttf=("time_to_failure", lambda x: np.percentile(x, 50)),
        ).reset_index().round(1)
        return result.sort_values("avg_ttf")

    def _query_by_tool(self, df, sql) -> pd.DataFrame:
        failed = df[df["failed"] == True]
        result = failed.groupby("tool_id").agg(
            n_failures=("time_to_failure", "count"),
            avg_ttf=("time_to_failure", "mean"),
            p10_ttf=("time_to_failure", lambda x: np.percentile(x, 10)),
        ).reset_index().round(1)
        return result.sort_values("avg_ttf")

    def _query_by_site(self, df, sql) -> pd.DataFrame:
        result = df.groupby("site").agg(
            total_units=("component_id", "count"),
            n_failures=("failed", "sum"),
            failure_rate=("failed", "mean"),
            avg_ttf_failures=("time_to_failure", lambda x: x[df.loc[x.index, "failed"]].mean() if df.loc[x.index, "failed"].any() else np.nan),
        ).reset_index()
        result["failure_rate"] = (result["failure_rate"] * 100).round(1)
        return result.sort_values("failure_rate", ascending=False)

    def _filter_component_type(self, df, sql) -> pd.DataFrame:
        # Extract component type from WHERE clause
        import re
        match = re.search(r"component_type\s*=\s*'([^']+)'", sql, re.IGNORECASE)
        if match:
            ctype = match.group(1)
            return df[df["component_type"].str.lower() == ctype.lower()].copy()
        return df

    def _early_failures(self, df, sql) -> pd.DataFrame:
        import re
        match = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
        limit = int(match.group(1)) if match else 10
        failed = df[df["failed"] == True].copy()
        return failed.nsmallest(limit, "time_to_failure")[
            ["component_id", "component_type", "tool_id", "site", "time_to_failure"]
        ]

    def _generic_filter(self, df, sql) -> pd.DataFrame:
        return df.copy()


class RealAthena:
    """
    Real AWS Athena client. Requires boto3 and S3 results bucket.
    Usage: set MOCK=False.
    """

    def __init__(self, database: str, results_bucket: str, region: str = "us-east-1"):
        try:
            import boto3
            self.athena = boto3.client("athena", region_name=region)
            self.database = database
            self.results_location = f"s3://{results_bucket}/athena-results/"
        except ImportError:
            raise ImportError("boto3 not installed. Run: pip install boto3")

    def query(self, sql: str) -> pd.DataFrame:
        import time, io
        import boto3

        response = self.athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.results_location},
        )
        execution_id = response["QueryExecutionId"]

        # Poll until complete
        while True:
            status = self.athena.get_query_execution(
                QueryExecutionId=execution_id
            )["QueryExecution"]["Status"]["State"]
            if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(1)

        if status != "SUCCEEDED":
            raise RuntimeError(f"Athena query {status}: {execution_id}")

        result = self.athena.get_query_results(QueryExecutionId=execution_id)
        cols = [c["Label"] for c in result["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
        rows = [[r.get("VarCharValue", "") for r in row["Data"]]
                for row in result["ResultSet"]["Rows"][1:]]
        return pd.DataFrame(rows, columns=cols)


# ── Pipeline Runner ──────────────────────────────────────────────────

class PipelineRunner:
    """
    Orchestrates the full reliability assessment pipeline:

        1. Ingest raw CSVs → S3
        2. Query via Athena to filter/aggregate
        3. Run ReliabilityReport on each component group
        4. Write results summary back to S3

    Works identically in mock and real AWS mode.
    """

    def __init__(self, mock: bool = MOCK, base_dir: str = None):
        self.mock = mock
        if mock:
            self.s3 = MockS3(base_dir)
            self.athena = MockAthena(self.s3)
        else:
            self.s3 = RealS3(BUCKET_NAME)
            self.athena = RealAthena(DATABASE, BUCKET_NAME)

        print(f"[Pipeline] Mode: {'MOCK (local)' if mock else 'AWS'}")

    def ingest(self, data_dir: str):
        """
        Upload all CSVs from a local directory to S3 raw prefix.
        In production, this would be triggered by new file arrivals.
        """
        data_dir = Path(data_dir)
        csv_files = list(data_dir.glob("*.csv"))
        print(f"\n[Pipeline] Ingesting {len(csv_files)} files from {data_dir}")

        for f in csv_files:
            s3_key = f"{RAW_PREFIX}{f.name}"
            self.s3.upload(str(f), s3_key)

        print(f"[Pipeline] Ingestion complete.\n")

    def run_queries(self) -> dict:
        """
        Run the standard suite of Athena analytical queries.
        Returns dict of query_name → DataFrame.
        """
        print("[Pipeline] Running Athena queries...\n")

        queries = {
            "fleet_summary_by_component": f"""
                SELECT
                    component_type,
                    COUNT(*) AS n_failures,
                    AVG(time_to_failure) AS avg_ttf,
                    MIN(time_to_failure) AS min_ttf,
                    APPROX_PERCENTILE(time_to_failure, 0.10) AS p10_ttf,
                    APPROX_PERCENTILE(time_to_failure, 0.50) AS p50_ttf
                FROM {DATABASE}.failures
                WHERE failed = true
                GROUP BY component_type
                ORDER BY avg_ttf ASC
            """,

            "failure_rate_by_site": f"""
                SELECT
                    site,
                    COUNT(*) AS total_units,
                    SUM(CASE WHEN failed THEN 1 ELSE 0 END) AS n_failures,
                    ROUND(100.0 * SUM(CASE WHEN failed THEN 1 ELSE 0 END) / COUNT(*), 1)
                        AS failure_rate_pct
                FROM {DATABASE}.failures
                GROUP BY site
                ORDER BY failure_rate_pct DESC
            """,

            "tool_reliability_ranking": f"""
                SELECT
                    tool_id,
                    COUNT(*) AS n_failures,
                    AVG(time_to_failure) AS avg_ttf,
                    APPROX_PERCENTILE(time_to_failure, 0.10) AS p10_ttf
                FROM {DATABASE}.failures
                WHERE failed = true
                GROUP BY tool_id
                ORDER BY avg_ttf ASC
            """,

            "early_failures": f"""
                SELECT
                    component_id,
                    component_type,
                    tool_id,
                    site,
                    time_to_failure
                FROM {DATABASE}.failures
                WHERE failed = true
                ORDER BY time_to_failure ASC
                LIMIT 10
            """,
        }

        results = {}
        for name, sql in queries.items():
            print(f"  Query: {name}")
            results[name] = self.athena.query(sql)
            print(f"  → {len(results[name])} rows\n")

        return results

    def run_assessments(self) -> pd.DataFrame:
        """
        Pull data from S3, run ReliabilityReport on each component type,
        return summary DataFrame.
        """
        import sys
        sys.path.append(".")
        from report import ReliabilityReport

        print("[Pipeline] Running reliability assessments...\n")

        # Load all data
        keys = self.s3.list_objects(prefix=RAW_PREFIX)
        dfs = [self.s3.read_csv(obj["Key"]) for obj in keys]
        all_data = pd.concat(dfs, ignore_index=True)

        rows = []
        for ctype in all_data["component_type"].unique():
            subset = all_data[
                (all_data["component_type"] == ctype) &
                (all_data["failed"] == True)
            ]["time_to_failure"].dropna()

            if len(subset) < 5:
                print(f"  Skipping {ctype} — insufficient failures ({len(subset)})")
                continue

            try:
                report = ReliabilityReport(subset.values, component_name=ctype)
                report.fit()
                rows.append({
                    "component_type": ctype,
                    "n_failures": len(subset),
                    "best_dist": report.best_name,
                    "aic": round(report.comparison.iloc[0]["aic"], 2),
                    "mttf": round(report.metrics["mttf"], 1),
                    "b10_life": round(report.metrics["b10"], 1),
                    "b50_life": round(report.metrics["b50"], 1),
                    "shape_k": round(report._best.get("k", float("nan")), 3),
                    "failure_mode": (
                        "infant_mortality" if report._best.get("k", 1) < 1
                        else "random" if report._best.get("k", 1) < 1.5
                        else "wear_out"
                    ),
                })
                print(f"  ✓ {ctype}  MTTF={report.metrics['mttf']:.0f}  B10={report.metrics['b10']:.0f}")
            except Exception as e:
                print(f"  ✗ {ctype} failed: {e}")

        results_df = pd.DataFrame(rows).sort_values("b10_life").reset_index(drop=True)
        return results_df

    def write_results(self, df: pd.DataFrame, name: str = "fleet_summary"):
        """Write assessment results back to S3 results prefix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{RESULTS_PREFIX}{name}_{timestamp}.csv"
        self.s3.write_csv(df, s3_key)
        print(f"\n[Pipeline] Results written to S3: {s3_key}")

    def run_full_pipeline(self, data_dir: str):
        """
        End-to-end pipeline:
            ingest → query → assess → write results
        """
        print("\n" + "=" * 60)
        print("  RELIABILITY PIPELINE — FULL RUN")
        print("=" * 60)

        # Step 1: Ingest
        self.ingest(data_dir)

        # Step 2: Athena queries
        query_results = self.run_queries()

        # Step 3: ReliabilityReport on each component
        assessments = self.run_assessments()

        # Step 4: Write back to S3
        self.write_results(assessments)

        return query_results, assessments


# ── Data generator for demo ──────────────────────────────────────────

def generate_fleet_data(output_dir: str, seed: int = 42):
    """
    Generate a realistic multi-site, multi-tool fleet dataset.
    Writes one CSV per site to output_dir.

    Schema matches the Athena CREATE TABLE above:
        component_id, component_type, tool_id, site,
        install_date, failure_date, time_to_failure, failed, failure_mode
    """
    import sys
    sys.path.append(".")
    from simulate import censored_sample

    rng = np.random.default_rng(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Fleet configuration
    sites = ["Austin", "Tokyo", "Dresden"]
    tools_per_site = 3
    components = [
        ("bearing",        2.5, 1000, "wear_out"),
        ("laser_module",   0.7,  800, "infant_mortality"),
        ("optical_sensor", 1.1, 1500, "random"),
        ("motor_drive",    3.2,  600, "wear_out"),
        ("cooling_fan",    1.8, 2000, "wear_out"),
    ]

    base_date = datetime(2023, 1, 1)
    obs_time = 900.0  # 900 hour observation window
    units_per_component = 15

    for site in sites:
        rows = []
        for tool_num in range(1, tools_per_site + 1):
            tool_id = f"{site[:3].upper()}-T{tool_num:02d}"
            for comp_name, k, lam, mode in components:
                # Add site-level variation
                lam_site = lam * rng.uniform(0.85, 1.15)
                df = censored_sample(
                    units_per_component, k, lam_site, obs_time,
                    seed=int(rng.integers(0, 9999))
                )
                for i, row in df.iterrows():
                    install = base_date + timedelta(days=int(rng.integers(0, 180)))
                    fail_date = (install + timedelta(hours=row["time"])).strftime("%Y-%m-%d") \
                        if row["failed"] else None
                    rows.append({
                        "component_id": f"{tool_id}-{comp_name}-{i:03d}",
                        "component_type": comp_name,
                        "tool_id": tool_id,
                        "site": site,
                        "install_date": install.strftime("%Y-%m-%d"),
                        "failure_date": fail_date or "",
                        "time_to_failure": round(row["time"], 2),
                        "failed": row["failed"],
                        "failure_mode": mode if row["failed"] else "",
                    })

        site_df = pd.DataFrame(rows)
        out_path = f"{output_dir}/{site.lower()}_failures.csv"
        site_df.to_csv(out_path, index=False)
        n_failed = site_df["failed"].sum()
        print(f"  Generated {site}: {len(site_df)} units, {n_failed} failures")

    print(f"\nFleet data written to: {output_dir}\n")


# ── Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = f"{tmpdir}/fleet_data"
        s3_dir = f"{tmpdir}/mock_s3"

        # Generate synthetic fleet data
        print("Generating synthetic fleet data...\n")
        generate_fleet_data(data_dir)

        # Run full pipeline
        pipeline = PipelineRunner(mock=True, base_dir=s3_dir)
        query_results, assessments = pipeline.run_full_pipeline(data_dir)

        # Print query results
        print("\n" + "=" * 60)
        print("  ATHENA QUERY RESULTS")
        print("=" * 60)

        print("\n--- Fleet Summary by Component Type ---")
        print(query_results["fleet_summary_by_component"].to_string(index=False))

        print("\n--- Failure Rate by Site ---")
        print(query_results["failure_rate_by_site"].to_string(index=False))

        print("\n--- Tool Reliability Ranking ---")
        print(query_results["tool_reliability_ranking"].to_string(index=False))

        print("\n--- Top 10 Earliest Failures ---")
        print(query_results["early_failures"].to_string(index=False))

        # Print assessment results
        print("\n" + "=" * 60)
        print("  RELIABILITY ASSESSMENTS — RANKED BY B10 LIFE")
        print("=" * 60)
        print(assessments.to_string(index=False))

        print("\n[Pipeline] Complete. In production, swap MOCK=False")
        print("and configure AWS credentials to run against real S3/Athena.")
