# Software Fault Detection Dataset and Reproduction Package

This repository contains the data and scripts used to reconstruct monthly
cumulative software fault-count datasets and reproduce the kernel-regression
prediction experiments.

The package is organized so that reviewers can answer three questions quickly:

1. What data are included?
2. How were the OSS monthly datasets constructed?
3. How can the single-kernel and multi-kernel experiments be run?

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── IST_OSS_datasets_records_OSS1_OSS8.xlsx
│   ├── processed_monthly/
│   │   ├── monthly_cumulative_data_reconstructed.csv
│   │   ├── OSS1_redis_monthly.csv
│   │   ├── ...
│   │   └── OSS8_rsshub_monthly.csv
│   ├── OSS/
│   │   ├── OSS_1.xlsx
│   │   ├── ...
│   │   └── OSS_8.xlsx
│   └── CSS/
│       ├── CSS_1.xlsx
│       ├── ...
│       └── CSS_6.xlsx
└── scripts/
    ├── construct_oss_datasets.py
    ├── single_kernel_reproduce.py
    └── multi_kernel_reproduce.py
```

## Environment

Python 3.10 or later is recommended.

Install the required packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

The required Python packages are:

```text
numpy
pandas
openpyxl
numba
pyswarm
```

## Data Files

### Main OSS Workbook

The main OSS workbook is:

```text
data/IST_OSS_datasets_records_OSS1_OSS8.xlsx
```

It contains four sheets:

| Sheet | Purpose |
|---|---|
| `README` | Brief workbook-level description |
| `Dataset_Metadata` | Dataset IDs, projects, repositories, time spans, and observation counts |
| `OSS_Bug_Fix_Issue_Records` | Cleaned bug/fault-related GitHub issue records |
| `Monthly_Cumulative_Data` | Monthly fault counts and cumulative fault counts |

The construction script also accepts the legacy raw-record sheet name
`Raw_Issue_Records` for compatibility, but the current workbook uses
`OSS_Bug_Fix_Issue_Records`.

### OSS Datasets

The OSS datasets were constructed from public GitHub issue records. Issue
creation time is treated as the fault-detection time. Records are aggregated
into calendar-month intervals, and months with no newly reported fault-related
issues are retained as zero-increment months.

| ID | Project | Repository | Raw records | Time span | Monthly observations | Final cumulative faults |
|---|---|---:|---:|---|---:|---:|
| OSS1 | Redis | `redis/redis` | 150 | 2011-09-16 to 2022-09-23 | 133 | 150 |
| OSS2 | Wox | `Wox-launcher/Wox` | 371 | 2014-01-06 to 2023-03-24 | 111 | 371 |
| OSS3 | Backbone | `jashkenas/backbone` | 190 | 2010-10-14 to 2023-01-19 | 148 | 190 |
| OSS4 | Homebrew | `Homebrew/brew` | 768 | 2016-04-03 to 2023-04-07 | 85 | 768 |
| OSS5 | PyTorch | `pytorch/pytorch` | 1266 | 2016-09-18 to 2023-03-27 | 79 | 1266 |
| OSS6 | Chart.js | `chartjs/Chart.js` | 2069 | 2013-03-18 to 2023-03-29 | 121 | 2069 |
| OSS7 | DevDocs | `freeCodeCamp/devdocs` | 234 | 2013-11-01 to 2023-03-06 | 113 | 234 |
| OSS8 | RSSHub | `DIYgod/RSSHub` | 1089 | 2018-04-27 to 2023-04-07 | 61 | 1089 |

### Derived Monthly CSV Files

The directory:

```text
data/processed_monthly/
```

contains reconstructed monthly datasets. Each CSV has the columns:

```text
Dataset ID, Project, Repository, Month, Monthly Fault Count, Cumulative Fault Count
```

The combined file is:

```text
data/processed_monthly/monthly_cumulative_data_reconstructed.csv
```

The per-dataset files are named:

```text
OSS1_redis_monthly.csv
OSS2_wox_monthly.csv
OSS3_backbone_monthly.csv
OSS4_brew_monthly.csv
OSS5_pytorch_monthly.csv
OSS6_chartjs_monthly.csv
OSS7_devdocs_monthly.csv
OSS8_rsshub_monthly.csv
```

### Experiment Input Arrays

The prediction scripts read simple Excel files from a data directory. Each file
is treated as one dataset.

`data/OSS/` contains the single-column cumulative fault-count arrays generated
from the OSS monthly data:

| File | Observations | Final cumulative faults |
|---|---:|---:|
| `data/OSS/OSS_1.xlsx` | 133 | 150 |
| `data/OSS/OSS_2.xlsx` | 111 | 371 |
| `data/OSS/OSS_3.xlsx` | 148 | 190 |
| `data/OSS/OSS_4.xlsx` | 85 | 768 |
| `data/OSS/OSS_5.xlsx` | 79 | 1266 |
| `data/OSS/OSS_6.xlsx` | 121 | 2069 |
| `data/OSS/OSS_7.xlsx` | 113 | 234 |
| `data/OSS/OSS_8.xlsx` | 61 | 1089 |

`data/CSS/` contains additional single-column cumulative fault-count arrays:

| File | Observations | Final cumulative faults |
|---|---:|---:|
| `data/CSS/CSS_1.xlsx` | 17 | 54 |
| `data/CSS/CSS_2.xlsx` | 14 | 38 |
| `data/CSS/CSS_3.xlsx` | 19 | 120 |
| `data/CSS/CSS_4.xlsx` | 14 | 9 |
| `data/CSS/CSS_5.xlsx` | 20 | 66 |
| `data/CSS/CSS_6.xlsx` | 30 | 52 |

## Reconstructing the OSS Monthly Datasets

Run the construction script from the repository root:

```bash
python scripts/construct_oss_datasets.py
```

This reads:

```text
data/IST_OSS_datasets_records_OSS1_OSS8.xlsx
```

and writes:

```text
data/processed_monthly/*.csv
data/OSS/OSS_*.xlsx
```

The script compares the reconstructed monthly data with the
`Monthly_Cumulative_Data` sheet in the workbook. A successful run prints:

```text
[OK] Reconstructed monthly data matches the Monthly_Cumulative_Data sheet.
```

Explicit equivalent command:

```bash
python scripts/construct_oss_datasets.py \
  --input data/IST_OSS_datasets_records_OSS1_OSS8.xlsx \
  --output-dir data/processed_monthly \
  --oss-output-dir data/OSS
```

## Running the Prediction Experiments

The experiment scripts support two input formats:

1. One numeric column: cumulative failures `y`; the script creates an implicit
   index `1..n`.
2. Two numeric columns: time/order `x` and cumulative failures `y`; `x` is used
   only for sorting. The model input is normalized internally by the number of
   observations.

Header rows are not required. Non-numeric rows are ignored.

The default start ratios are:

```text
0.2, 0.5, 0.8
```

The main error metric is PMAE.

### Single-Kernel Experiment

Run the OSS single-kernel experiment:

```bash
python scripts/single_kernel_reproduce.py \
  --data-dir data/OSS \
  --output single_kernel_OSS_results.xlsx
```

Run the CSS single-kernel experiment:

```bash
python scripts/single_kernel_reproduce.py \
  --data-dir data/CSS \
  --output single_kernel_CSS_results.xlsx
```

Useful explicit settings:

```bash
python scripts/single_kernel_reproduce.py \
  --data-dir data/OSS \
  --output single_kernel_OSS_results.xlsx \
  --processes 8 \
  --start-ratios 0.2,0.5,0.8 \
  --grid-size 100 \
  --lscv-grid-size 100
```

The output workbook contains four sheets:

```text
NW_Fixed
NW_Dynamic
LL_Fixed
LL_Dynamic
```

Each sheet contains:

```text
Dataset, Target_Start_Ratio, Actual_Start_Ratio, Predicted_Y_Sequence, PMAE
```

### Multi-Kernel Experiment

Run the OSS multi-kernel experiment:

```bash
python scripts/multi_kernel_reproduce.py \
  --data-dir data/OSS \
  --output multi_kernel_OSS_results.xlsx
```

Run the CSS multi-kernel experiment:

```bash
python scripts/multi_kernel_reproduce.py \
  --data-dir data/CSS \
  --output multi_kernel_CSS_results.xlsx
```

Useful explicit settings:

```bash
python scripts/multi_kernel_reproduce.py \
  --data-dir data/OSS \
  --output multi_kernel_OSS_results.xlsx \
  --processes 4 \
  --start-ratios 0.2,0.5,0.8 \
  --num-restarts 5 \
  --swarmsize 50 \
  --maxiter 50 \
  --lscv-grid-size 100
```

Use `--fixed-only` to run only fixed-mode predictions:

```bash
python scripts/multi_kernel_reproduce.py \
  --data-dir data/OSS \
  --output multi_kernel_OSS_fixed_only.xlsx \
  --fixed-only
```

The multi-kernel script uses PSO optimization through `pyswarm`.
`--use-weight-refine` enables the optional weight-refinement stage.

## Quick Smoke Tests

These commands use reduced grids and one start ratio. They are intended only to
verify that the scripts run successfully, not to reproduce the final experiment
settings.

```bash
python scripts/single_kernel_reproduce.py \
  --data-dir data/OSS \
  --output smoke_single_kernel.xlsx \
  --processes 1 \
  --start-ratios 0.8 \
  --grid-size 5 \
  --lscv-grid-size 5
```

```bash
python scripts/multi_kernel_reproduce.py \
  --data-dir data/OSS \
  --output smoke_multi_kernel.xlsx \
  --processes 1 \
  --start-ratios 0.8 \
  --fixed-only \
  --num-restarts 1 \
  --swarmsize 5 \
  --maxiter 2 \
  --lscv-grid-size 5
```

## Reproducibility Notes

- The OSS construction step removes duplicate issue numbers within each
  dataset.
- The monthly aggregation keeps a continuous calendar-month range from the
  first fault-related issue to the last fault-related issue.
- Months without new fault-related issues are retained with zero monthly
  increments.
- The cumulative fault-count sequence is the cumulative sum of the monthly
  fault counts.
- `Actual_Start_Ratio` can differ slightly from `Target_Start_Ratio` because
  the starting index is computed from the integer observation count.
- Multi-kernel PSO optimization can be time-consuming. Runtime depends on the
  number of datasets, start ratios, worker processes, restarts, swarm size, and
  iteration count.

## Minimal Reviewer Workflow

From a clean checkout:

```bash
python -m pip install -r requirements.txt
python scripts/construct_oss_datasets.py
python scripts/single_kernel_reproduce.py --data-dir data/OSS --output single_kernel_OSS_results.xlsx
python scripts/multi_kernel_reproduce.py --data-dir data/OSS --output multi_kernel_OSS_results.xlsx
```

For the additional CSS datasets, replace `data/OSS` with `data/CSS`.
