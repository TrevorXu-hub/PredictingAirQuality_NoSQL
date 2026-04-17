# DS 4320 Project 2: Predicting Air Quality Using PM2.5 Measurements

**Name:** Trevor Xu
**NetID:** pxg6af
**DOI:** [10.5281/zenodo.19634674](https://doi.org/10.5281/zenodo.19634674) *(update after Zenodo upload)*
**License:** [MIT](LICENSE)

**Quick Links:**
- [Press Release](press_release/press_release.md)
- [Pipeline Notebook](pipeline/pipeline.ipynb)
- [Pipeline (Markdown)](pipeline/pipeline.md)

---

## Executive Summary

This repository contains all materials for DS 4320 Project 2, which investigates the prediction of daily PM2.5-based Air Quality Index (AQI) levels across major U.S. cities using a document-oriented database and machine learning pipeline. Raw monitoring data was sourced from the EPA Air Quality System (AQS), cleaned, and stored as structured documents in a MongoDB Atlas collection. A Jupyter Notebook pipeline queries the database, engineers temporal and spatial features, trains a Random Forest classifier to predict AQI category (Good, Moderate, Unhealthy, etc.), and produces publication-quality visualizations of model performance and spatial air quality patterns. The final model achieves 79.4% test accuracy, with the previous day's PM2.5 reading, 7-day rolling standard deviation, and 7-day rolling mean identified as the top three predictors.

---

## Problem Definition

### General Problem
Predicting air quality.

### Specific Problem
Can we predict the next-day AQI category (Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy / Very Unhealthy / Hazardous) for a given U.S. monitoring site using only that site's historical daily PM2.5 measurements and temporal features?

### Motivation
Air pollution is one of the leading environmental causes of premature death globally, with fine particulate matter (PM2.5) posing particular risks to respiratory and cardiovascular health. Despite the availability of real-time monitoring data from thousands of stations across the United States, most individuals have limited ability to anticipate deteriorating air quality before it occurs. Accurate next-day AQI prediction would allow vulnerable populations — including the elderly, children, and those with pre-existing conditions — to plan outdoor activities, adjust medication schedules, and take preventive action. It would also support public health agencies in issuing timely advisories and allocating resources proactively rather than reactively.

### Rationale for Refinement
The general problem of "predicting air quality" is deliberately broad and could encompass dozens of pollutants, geographic scales, and time horizons. This project narrows the focus to PM2.5 specifically because it is the pollutant most consistently linked to adverse health outcomes and the primary driver of the U.S. AQI on most days. Predicting AQI *category* rather than a raw concentration value makes the output directly actionable for end users who rely on the color-coded AQI scale. Limiting the feature set to historical measurements and temporal variables keeps the data pipeline reproducible and fully grounded in the EPA AQS secondary dataset.

### Press Release
[Most Americans Don't Know Their Air Will Be Unhealthy Tomorrow. A New Machine Learning Tool Can Change That.](press_release/press_release.md)

---

## Domain Exposition

### Domain Overview
This project lives at the intersection of environmental science, public health, and data science. The U.S. Environmental Protection Agency operates the Air Quality System (AQS), a national network of thousands of outdoor monitoring stations that continuously measure concentrations of criteria pollutants including particulate matter, ozone, carbon monoxide, sulfur dioxide, and nitrogen dioxide. The EPA translates raw pollutant concentrations into the Air Quality Index — a unified 0–500 scale with six color-coded categories — to communicate health risk to the public. PM2.5 refers to airborne particles with an aerodynamic diameter of 2.5 micrometers or less, small enough to penetrate deep into the lungs and enter the bloodstream. Sources include vehicle exhaust, industrial emissions, wildfire smoke, and secondary formation from chemical reactions in the atmosphere. Monitoring methods are standardized: parameter code 88502 designates the "Acceptable PM2.5 AQI & Speciation Mass" method, an EPA-approved alternative to the Federal Reference Method (88101) that uses a range of continuous and filter-based instruments.

### Terminology

| Term | Definition |
|------|------------|
| PM2.5 | Fine particulate matter ≤ 2.5 µm in diameter |
| AQI | Air Quality Index: 0–500 scale translating pollutant concentration to health risk |
| AQS | EPA Air Quality System — national monitoring database |
| Parameter 88502 | EPA code for Acceptable PM2.5 AQI & Speciation Mass monitoring method |
| FRM | Federal Reference Method — gold-standard gravimetric PM2.5 measurement |
| µg/m³ | Micrograms per cubic meter — concentration unit for PM2.5 |
| CBSA | Core-Based Statistical Area — metro area grouping used by EPA |
| Site ID | Composite identifier: state code – county code – site number |
| Daily Mean | Arithmetic average of sub-daily PM2.5 readings for a calendar day |
| Observation Percent | Percentage of possible daily observations actually recorded |
| Good | AQI 0–50: air quality satisfactory, little or no risk |
| Moderate | AQI 51–100: acceptable; some pollutants may concern sensitive individuals |
| USG | AQI 101–150: Unhealthy for Sensitive Groups |
| Unhealthy | AQI 151–200: everyone may experience health effects |
| Very Unhealthy | AQI 201–300: health alert, serious effects for all |
| Hazardous | AQI 301–500: emergency conditions |

### Background Reading

All supporting articles are stored in the [`background_reading/`](background_reading/) folder.

| Title | Description | File |
|-------|-------------|------|
| EPA AQI Technical Assistance Document | Official EPA documentation of AQI calculation methodology and category breakpoints | [background_reading/epa_aqi_technical_doc.pdf](background_reading/epa_aqi_technical_doc.pdf) |
| Health Effects of PM2.5 (EPA) | Summary of peer-reviewed evidence linking PM2.5 to cardiovascular and respiratory disease | [background_reading/pm25_health_effects.pdf](background_reading/pm25_health_effects.pdf) |
| AQS Data Dictionary | Column-level documentation for all EPA AQS bulk download files | [background_reading/aqs_data_dictionary.pdf](background_reading/aqs_data_dictionary.pdf) |
| WHO Global Air Quality Guidelines 2021 | International benchmark PM2.5 exposure guidelines and their health rationale | [background_reading/who_air_quality_guidelines.pdf](background_reading/who_air_quality_guidelines.pdf) |
| Machine Learning for Air Quality Prediction (Review) | Survey of ML approaches applied to AQI forecasting across recent literature | [background_reading/ml_air_quality_review.pdf](background_reading/ml_air_quality_review.pdf) |

---

## Data Creation

### Provenance
The dataset is derived from the EPA Air Quality System (AQS) bulk download portal (`https://aqs.epa.gov/aqsweb/airdata/download_files.html`). The specific source file is `daily_88502_2023.csv`, the 2023 annual daily summary for parameter code 88502 (Acceptable PM2.5 AQI & Speciation Mass). This file is generated by the EPA from readings submitted by state, local, and tribal air quality agencies across the United States and is updated annually after data certification. The raw file contains 176,246 rows spanning 47 states and 224 cities before cleaning.

Data was ingested using the custom Python script `ingestion.py`, which reads the local CSV, removes instrument-error outliers, and batch-upserts documents into a MongoDB Atlas cluster. The script logs every step to `logs/ingestion.log` for audit traceability and includes error handling for connection failures, CSV parsing errors, and partial batch write failures.

### Code

| File | Description |
|------|-------------|
| [`ingestion.py`](ingestion/ingestion.py) | Reads local EPA CSV, cleans outliers, and upserts documents into MongoDB Atlas with full logging |
| [`pipeline.ipynb`](pipeline.pipeline.ipynb) | End-to-end analysis pipeline: MongoDB query → feature engineering → Random Forest → visualizations |
| [`pipeline.md`](pipeline/pipeline.md) | Markdown export of the pipeline notebook for easy viewing on GitHub |
| [`requirements.txt`](requirements.txt) | Python package dependencies |

### Critical Decisions and Uncertainty

Several judgement calls were made during data creation that affect the final dataset and introduce or mitigate uncertainty:

**Parameter choice (88502 vs 88101):** The 88502 parameter encompasses multiple EPA-approved PM2.5 measurement methods beyond the Federal Reference Method (88101). This introduces a small source of measurement uncertainty because different instruments can produce readings that differ by a few µg/m³ under certain atmospheric conditions. This tradeoff was accepted because 88502 provides broader geographic coverage — particularly in states like Washington, Oregon, and California — which is more valuable for a multi-state predictive model than strict method uniformity.

**Averaging window:** The 24-hour averaging window used for daily means smooths out short-term pollution spikes but may underrepresent episodic pollution events such as wildfire smoke incursions or industrial accidents that last only a few hours.

**State selection:** Ten states were selected for ingestion (CA, TX, NY, IL, AZ, CO, WA, FL, OR, ID) to balance geographic diversity with collection size. Including all 47 states in the raw file would exceed MongoDB Atlas free-tier storage limits and slow model training without materially improving the predictive framework.

**Outlier removal thresholds:** Negative PM2.5 values and AQI values above 500 were treated as data-entry or instrument errors and removed before ingestion. Alternative approaches such as winsorization were considered but rejected because these errors are clearly non-physical and would bias the model if retained.

### Bias Identification
- **Geographic bias:** Monitoring sites are not uniformly distributed. Urban and suburban areas are systematically over-represented relative to rural communities, which may have fewer regulatory monitors despite significant agricultural or wildfire-related pollution exposure.
- **Instrument bias:** The 88502 parameter aggregates multiple measurement methods; differences in instrument sensitivity and calibration frequency across agencies introduce inter-site variability that is not captured in the data.
- **Seasonal bias:** The 2023 dataset spans one calendar year. Wildfire events (predominantly summer/fall in Western states) and winter inversion events are captured, but a single-year snapshot may not represent long-term climatological patterns.
- **Missing data bias:** Sites with low `observation_percent` values have incomplete daily records. Rows with missing PM2.5 mean values were dropped, which may disproportionately remove data from under-resourced monitoring programs.

### Bias Mitigation
- Negative PM2.5 values (360 rows, ~0.2%) were removed as instrument artefacts prior to ingestion.
- Rows with AQI > 500 (3 rows) were removed as they exceed the defined EPA scale and represent data entry errors.
- The `observation_percent` field is retained in each document so downstream analysis can filter to well-sampled days (e.g., ≥ 75% completeness).
- Geographic diversity was maintained by including 10 states spanning different climate regions, population densities, and pollution source profiles.
- The Random Forest classifier uses `class_weight='balanced'` to counter the heavy class imbalance (Good days far outnumber Hazardous days), ensuring the model does not trivially predict "Good" for every sample.

---

## Metadata

### Implicit Schema Guidelines

Every document in the `pm25_daily` collection follows the structure below. All top-level keys are present in every document; fields that may be null are marked accordingly. The schema is enforced by `ingestion.py`, which rejects any row missing required measurements before upsert.

```
{
  date:        ISODate       -- UTC midnight of the observation date
  year:        Integer       -- calendar year (2023)
  month:       Integer       -- 1–12
  day_of_week: String        -- "Monday" … "Sunday"

  location: {
    city:      String        -- city name from EPA record
    county:    String        -- county name
    state:     String        -- full state name
    cbsa_name: String | null -- metro area name (null for rural sites)
    latitude:  Float | null  -- decimal degrees, WGS84
    longitude: Float | null  -- decimal degrees, WGS84
  }

  air_quality: {
    pm25_mean:    Float        -- µg/m³, daily arithmetic mean, ≥ 0
    pm25_max:     Float | null -- µg/m³, highest single-hour reading
    aqi:          Integer | null -- 0–500
    aqi_category: String       -- Good / Moderate / USG / Unhealthy /
                               --   Very Unhealthy / Hazardous / Unknown
  }

  provenance: {
    source:    String          -- "EPA AQS"
    parameter: String          -- "PM2.5 Speciation Mass (88502)"
  }
}
```

**Schema rules enforced during ingestion:**
- A compound unique index on `(date, location.site_id)` prevents duplicate daily readings for the same monitoring site.
- PM2.5 mean values must be non-negative; AQI values must not exceed 500 (EPA scale maximum).
- Documents missing `pm25_mean` or `date` are rejected — these are the minimum fields required for downstream analysis.
- Timestamps are stored in UTC to avoid timezone ambiguity across states.

### Data Summary

| Property | Value |
|----------|-------|
| Database | `air_quality_db` |
| Collection | `pm25_daily` |
| Source year | 2023 |
| Source parameter | EPA AQS 88502 (Acceptable PM2.5 AQI & Speciation Mass) |
| States included | CA, TX, NY, IL, AZ, CO, WA, FL, OR, ID |
| Raw rows in source CSV | 176,246 |
| Rows removed (cleaning) | 363 (negative PM2.5 + AQI > 500) |
| Documents after cleaning | ~95,000 *(update with actual count from MongoDB)* |
| Date range | 2023-01-01 – 2023-12-31 |
| Unique cities | 224 |
| Approximate monitoring sites | 500+ |

### Data Dictionary

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `date` | ISODate | UTC observation date | `2023-07-15T00:00:00Z` |
| `year` | Integer | Calendar year | `2023` |
| `month` | Integer | Month number 1–12 | `7` |
| `day_of_week` | String | Day name | `"Saturday"` |
| `location.city` | String | City name per EPA record | `"Los Angeles"` |
| `location.county` | String | County name | `"Los Angeles"` |
| `location.state` | String | Full state name | `"California"` |
| `location.cbsa_name` | String \| null | Metro area name | `"Los Angeles-Long Beach-Anaheim, CA"` |
| `location.latitude` | Float \| null | Latitude, decimal degrees | `34.0694` |
| `location.longitude` | Float \| null | Longitude, decimal degrees | `-118.2276` |
| `air_quality.pm25_mean` | Float | Daily arithmetic mean PM2.5 concentration (µg/m³) | `12.4` |
| `air_quality.pm25_max` | Float \| null | Highest hourly PM2.5 reading of the day (µg/m³) | `18.1` |
| `air_quality.aqi` | Integer \| null | EPA Air Quality Index value | `52` |
| `air_quality.aqi_category` | String | AQI health category label | `"Moderate"` |
| `provenance.source` | String | Data origin | `"EPA AQS"` |
| `provenance.parameter` | String | EPA parameter description | `"PM2.5 Speciation Mass (88502)"` |

### Quantification of Uncertainty for Numerical Features

EPA AQS data is subject to several well-documented sources of measurement uncertainty. The table below quantifies these for each numerical field, drawing on EPA quality assurance documentation and peer-reviewed studies of 88502-class instruments.

| Field | Uncertainty Type | Quantified Range | Source / Rationale |
|-------|------------------|------------------|--------------------|
| `location.latitude` | Coordinate precision | ± 0.0001° (≈ ± 11 meters at 40° N) | EPA AQS reports coordinates to 4 decimal places; this reflects GPS resolution of monitoring site surveys. |
| `location.longitude` | Coordinate precision | ± 0.0001° (≈ ± 8–10 meters at 40° N) | Same as latitude; longitude distance varies with latitude. |
| `air_quality.pm25_mean` | Instrument bias + method variability | ± 1.0 to ± 2.5 µg/m³ at concentrations < 35 µg/m³; ± 10% at higher concentrations | EPA QA Handbook Vol II §12; 88502 encompasses multiple FEM and continuous methods whose inter-method agreement has been measured in collocation studies. |
| `air_quality.pm25_mean` | Temporal sampling error | ± 5–15% when `observation_percent` < 75% | Daily means computed from fewer than 18 of 24 hours have higher variance. Our documents retain `observation_percent` so downstream code can filter. |
| `air_quality.pm25_max` | Instrument response time | ± 2–3 µg/m³ | Hourly maxima are more sensitive to short-timescale instrument noise than daily means. |
| `air_quality.aqi` | Derived uncertainty | ± 5 AQI units at concentrations near category breakpoints | AQI is a piecewise-linear function of pm25_mean, so any measurement error is amplified when the true concentration is near a breakpoint (e.g., 35.4 µg/m³ separates Moderate from Unhealthy for Sensitive Groups). |
| `air_quality.aqi_category` | Category misclassification | 5–10% of days near breakpoints | Downstream model predictions inherit this uncertainty — days with PM2.5 near breakpoints are inherently harder to classify, regardless of model quality. |

**Overall dataset reliability:**
- Approximately **99.8%** of raw rows passed cleaning (363 of 176,246 removed).
- Missing-data rates for retained documents: `pm25_max` absent in ~4% of records, `aqi` absent in ~1%, `cbsa_name` absent in ~6% (rural sites).
- The 88502 parameter is classified by EPA as "acceptable" for compliance-equivalent analysis but is not the Federal Reference Method (88101); results should not be used for direct regulatory enforcement.
