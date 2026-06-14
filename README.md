# Data and Reproducibility Package

This repository provides the datasets and supporting files used in the paper:

**Long-Term Software Reliability Prediction via Nonparametric Kernel Regression Methods**

The purpose of this repository is to make the open-source software (OSS) dataset construction process transparent and reproducible. The OSS datasets were constructed from public GitHub issue records and transformed into grouped cumulative fault-count sequences for software reliability prediction.

---

## 1. Repository Contents

```text
.
├── README.md
├── data/
│   └── IST_OSS_datasets_clean_release.xlsx
└── scripts/
  ├── construct_oss_datasets.py
  └── run_experiments.py

 
```

The main dataset file is:

```text
data/IST_OSS_datasets_clean_release.xlsx
```

This Excel file contains the cleaned OSS issue records and the corresponding monthly cumulative fault-count data used in the experiments.

---

## 2. OSS Dataset Overview

The OSS datasets were collected from six public GitHub repositories. Each raw issue record contains the issue number, issue title, issue creation time, and issue labels/categories. The issue creation time is used as the fault-detection time.

| Dataset ID | Project  | GitHub Repository    | Raw Records | Time Span                | Monthly Observations |
| ---------- | -------- | -------------------- | ----------: | ------------------------ | -------------------: |
| OSS1       | Redis    | `redis/redis`        |         150 | 2011-09-16 to 2022-09-23 |                  133 |
| OSS2       | Wox      | `Wox-launcher/Wox`   |         371 | 2014-01-06 to 2023-03-24 |                  111 |
| OSS3       | Backbone | `jashkenas/backbone` |         190 | 2010-10-14 to 2023-01-19 |                  148 |
| OSS4       | Homebrew | `Homebrew/brew`      |         768 | 2016-04-03 to 2023-04-07 |                   85 |
| OSS5       | PyTorch  | `pytorch/pytorch`    |        1266 | 2016-09-18 to 2023-03-27 |                   79 |
| OSS6       | Chart.js | `chartjs/Chart.js`   |        2069 | 2013-03-18 to 2023-03-29 |                  121 |

---

## 3. Excel File Description

The file `IST_OSS_datasets_clean_release.xlsx` contains four sheets.

| Sheet Name                | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `README`                  | Brief description of the data file                                         |
| `Dataset_Metadata`        | Summary information for each OSS dataset                                   |
| `Raw_Issue_Records`       | Cleaned raw GitHub issue records for the six OSS datasets                  |
| `Monthly_Cumulative_Data` | Monthly fault counts and cumulative fault counts used as experiment inputs |

The sheet `Raw_Issue_Records` contains the following columns:

| Column          | Description                                |
| --------------- | ------------------------------------------ |
| `dataset_id`    | Dataset identifier, from OSS1 to OSS6      |
| `project`       | Project name                               |
| `repository`    | GitHub repository                          |
| `issue_number`  | GitHub issue number                        |
| `title`         | Issue title                                |
| `created_at`    | Issue creation time                        |
| `labels`        | Issue labels or categories                 |
| `created_month` | Calendar month extracted from `created_at` |
| `issue_url`     | URL of the corresponding GitHub issue      |

The sheet `Monthly_Cumulative_Data` contains the following columns:

| Column                   | Description                                                                    |
| ------------------------ | ------------------------------------------------------------------------------ |
| `dataset_id`             | Dataset identifier                                                             |
| `project`                | Project name                                                                   |
| `month`                  | Calendar-month interval                                                        |
| `monthly_fault_count`    | Number of newly reported fault-related issues in that month                    |
| `cumulative_fault_count` | Cumulative number of reported fault-related issues up to the end of that month |

---

## 4. Fault-Related Issue Selection

Fault-related records were selected using project-specific labels or categories indicating bug reports, crashes, errors, or defect-related events.

| Dataset  | Selection Rule                                                                                                                         |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Redis    | Defect-related labels such as `class:bug`, `critical bug`, `non critical bug`, and `crash report`                                      |
| Wox      | Labels containing `bug`, such as `bug`, `bug, plugin`, `bug, ui`, and `bug, help wanted`                                               |
| Backbone | Labels containing `bug`, such as `bug`, `bug, fixed`, and `bug, change`                                                                |
| Homebrew | Mainly labels containing `bug` or other defect-related categories in the original collected issue set                                  |
| PyTorch  | Defect-related labels such as `bug`, `module: crash`, `module: error checking`, `module: dependency bug`, and `module: assert failure` |
| Chart.js | Issues labeled with `type: bug`                                                                                                        |

The constructed OSS datasets represent reported fault-count processes. They should be interpreted as fault-detection records rather than complete fault-repair histories, because issue creation times are used instead of issue closing times or fix-commit times.

---

## 5. Dataset Construction Procedure

The raw GitHub issue records were transformed into grouped cumulative fault-count sequences using the following procedure:

1. Collect raw issue records from the corresponding GitHub repositories.
2. Select fault-related issues according to the label rules described above.
3. Remove duplicate issue numbers.
4. Sort the remaining records chronologically by issue creation time.
5. Aggregate the issue records into calendar-month intervals.
6. Retain months with no newly reported fault-related issues as zero-increment months.
7. Compute the cumulative fault-count sequence by cumulatively summing the monthly counts.

Let (n_j) denote the number of newly reported fault-related issues in the (j)-th month. The cumulative fault count is defined as

[
y_j = \sum_{k=1}^{j} n_k,
]

where (y_j) represents the cumulative number of reported faults up to the end of month (j).

---

## 6. Reconstructing the Monthly Datasets

To reconstruct the monthly cumulative fault-count sequences from the cleaned raw issue records, run:

```bash
python scripts/construct_oss_datasets.py
```

The script reads the raw issue records from:

```text
data/IST_OSS_datasets_clean_release.xlsx
```

and regenerates the monthly cumulative data corresponding to the sheet:

```text
Monthly_Cumulative_Data
```

---

## 7. Running the Experiments

To reproduce the prediction experiments, run:

```bash
python scripts/run_experiments.py
```

The experiments use the grouped cumulative fault-count sequences in `Monthly_Cumulative_Data`.

The prediction settings correspond to three testing phases:

| Phase        | Initial Training Ratio |
| ------------ | ---------------------: |
| Early phase  |                    20% |
| Middle phase |                    50% |
| Late phase   |                    80% |

The main evaluation metric is PMAE.

---

## 8. Notes

The GitHub repositories may continue to evolve after the data collection period. Therefore, the released Excel file should be used to reproduce the exact datasets used in the paper.

This repository provides the cleaned raw issue records, the monthly cumulative fault-count sequences, and the scripts needed to reproduce the dataset construction and experiments.
