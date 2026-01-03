use anyhow::Context;
use arrow_array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::concat::concat_batches;
use burn_dataset::WarehouseManifest;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "warehouse_export",
    about = "Export warehouse manifest summaries to Parquet for analytics"
)]
struct Args {
    #[command(flatten)]
    output: cli_support::common::WarehouseOutputArgs,
    /// Path to manifest.json produced by warehouse_etl.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Output parquet path.
    #[arg(long, default_value = "warehouse_summary.parquet")]
    out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let manifest_path = args
        .manifest
        .unwrap_or_else(|| args.output.output_root.join("manifest.json"));
    let manifest = WarehouseManifest::load(&manifest_path)
        .with_context(|| format!("loading manifest {}", manifest_path.display()))?;

    let schema = Schema::new(vec![
        Field::new("run", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("total", DataType::UInt64, false),
        Field::new("non_empty", DataType::UInt64, false),
        Field::new("empty", DataType::UInt64, false),
        Field::new("missing_image", DataType::UInt64, false),
        Field::new("missing_file", DataType::UInt64, false),
        Field::new("invalid", DataType::UInt64, false),
        Field::new("version", DataType::Utf8, false),
        Field::new("code_version", DataType::Utf8, false),
    ]);

    let mut batches = Vec::new();

    // Per-run rows
    if !manifest.summary.runs.is_empty() {
        let run_ids: Vec<_> = manifest
            .summary
            .runs
            .iter()
            .map(|r| r.run_dir.display().to_string())
            .collect();
        let run_kind = vec!["run"; run_ids.len()];
        let total: Vec<u64> = manifest
            .summary
            .runs
            .iter()
            .map(|r| r.total as u64)
            .collect();
        let non_empty: Vec<u64> = manifest
            .summary
            .runs
            .iter()
            .map(|r| r.non_empty as u64)
            .collect();
        let empty: Vec<u64> = manifest
            .summary
            .runs
            .iter()
            .map(|r| r.empty as u64)
            .collect();
        let missing_image: Vec<u64> = manifest
            .summary
            .runs
            .iter()
            .map(|r| r.missing_image as u64)
            .collect();
        let missing_file: Vec<u64> = manifest
            .summary
            .runs
            .iter()
            .map(|r| r.missing_file as u64)
            .collect();
        let invalid: Vec<u64> = manifest
            .summary
            .runs
            .iter()
            .map(|r| r.invalid as u64)
            .collect();
        let version: Vec<_> = vec![manifest.version.clone(); run_ids.len()];
        let code_version: Vec<_> = vec![manifest.code_version.clone(); run_ids.len()];

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(StringArray::from(run_ids)) as ArrayRef,
                Arc::new(StringArray::from(run_kind)) as ArrayRef,
                Arc::new(UInt64Array::from(total)) as ArrayRef,
                Arc::new(UInt64Array::from(non_empty)) as ArrayRef,
                Arc::new(UInt64Array::from(empty)) as ArrayRef,
                Arc::new(UInt64Array::from(missing_image)) as ArrayRef,
                Arc::new(UInt64Array::from(missing_file)) as ArrayRef,
                Arc::new(UInt64Array::from(invalid)) as ArrayRef,
                Arc::new(StringArray::from(version)) as ArrayRef,
                Arc::new(StringArray::from(code_version)) as ArrayRef,
            ],
        )?;
        batches.push(batch);
    }

    // Aggregate row
    let totals = &manifest.summary.totals;
    let agg_batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(vec!["__total__"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["total"])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![totals.total as u64])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![totals.non_empty as u64])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![totals.empty as u64])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![totals.missing_image as u64])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![totals.missing_file as u64])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![totals.invalid as u64])) as ArrayRef,
            Arc::new(StringArray::from(vec![manifest.version.clone()])) as ArrayRef,
            Arc::new(StringArray::from(vec![manifest.code_version.clone()])) as ArrayRef,
        ],
    )?;
    batches.push(agg_batch);

    let merged = if batches.len() == 1 {
        batches.pop().unwrap()
    } else {
        concat_batches(&std::sync::Arc::new(schema.clone()), &batches)?
    };

    let file = File::create(&args.out)
        .with_context(|| format!("creating parquet {}", args.out.display()))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, merged.schema(), Some(props))?;
    writer.write(&merged)?;
    writer.close()?;

    println!(
        "Wrote Parquet summary to {} (rows={})",
        args.out.display(),
        merged.num_rows()
    );
    Ok(())
}
