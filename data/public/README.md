# Public benchmarks

| File | Benchmark | Provenance | Licence / access |
|---|---|---|---|
| `HDI_2023.xlsx` | Human Development Index, data for 2023 (193 countries, 4 dimension indicators) | UNDP, *Human Development Report 2025*, Statistical Annex, Table 1 | public |
| `WHR24.xlsx` | World Happiness Report 2024, explained components (144 complete rows, 6 components) | Helliwell et al. (eds.), *World Happiness Report 2024* | public |
| `BLI_2017.csv` | OECD Better Life Index 2017 (35 OECD countries, 11 dimension scores) | R package **Compind** (Vidoli & Fusco), dataset `BLI_2017`; GitHub mirror `cran/Compind`, commit ba7736512a8a | GPL-3 (package) / OECD data |
| `ASEM_iData.csv`, `ASEM_iMeta.csv` | ASEM Sustainable Connectivity (51 countries, 49 indicators, 8 pillars) | R package **COINr** (Becker et al., JRC), datasets `ASEM_iData`, `ASEM_iMeta`; GitHub mirror `cran/COINr`, commit 9fcd763ec5ee | MIT (package) / JRC ASEM portal |
| `HPI_2017_2019.csv` | Happy Planet Index 2017–2019 (optional, not used in the manuscript) | R package **Compind**, dataset `data_HPI` | GPL-3 |
| `../electric_v_selection.xlsx` | Compact electric vehicles (19 × 15) | Więckowski et al. (2024), *Artificial Intelligence Review* 57:270 | published table |

The CSV files were exported unchanged from the `.rda` files with `pyreadr`. The ASEM pillar construction
(denomination, min-max normalisation with direction, mean within pillars, official hierarchy weights) is
in `code/gpii/data.py::load_asem`.
