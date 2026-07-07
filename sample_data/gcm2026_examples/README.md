# GCM 2026 Sample Data

This directory contains five de-identified example cases from the GCM 2026
dataset. Each case includes ADC, T2_FS, and venous-phase MRI volumes with the
corresponding gastric lesion mask files.

The accompanying `sample_classification_metrics.xlsx` file stores
de-identified task labels and staging fields for these examples. Personal
identifiers such as names, hospital numbers, and contact information are
intentionally excluded.

```text
sample_data/gcm2026_examples/
  sample_classification_metrics.xlsx
  cases/
    case_id/
      ADC/
      T2_FS/
      V/
```
