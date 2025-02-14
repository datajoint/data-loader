# Changelog

Observes [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standard and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) convention.

## [0.7.0] - 2024-08-09

+ Add - `memoized_result` decorator to cache function results
+ Update - `prairie_view_loader.py` to create big tiff files from `.ome.tif` files
+ Update - `run_caiman.py` to run latest version of CaImAn
+ Update - `caiman_loader.py` to process output of latest version of CaImAn
+ Fix - general fixes and improvements

## [0.6.1] - 2023-08-02

+ Update DANDI upload functionality to improve usability

## [0.6.0] - 2023-07-26

+ Update - `prairieviewreader.py` -> `prairie_view_loader.py`
+ Update - `get_pv_metadata()` -> `get_prairieview_metadata()`
+ Update - Internal variable names within `prairie_view_loader.py`

## [0.5.4] - 2023-05-25

+ Fix - DANDI URL for uploads where `staging=False`.
+ Fix - Docs to remove Google Analytics Key and add Markdown extensions

## [0.5.3] - 2023-05-11

+ Fix - `.ipynb` dark mode output for all notebooks.
+ Fix - Remove `GOOGLE_ANALYTICS_KEY` from `u24_element_release_call.yml`.

## [0.5.2] - 2023-04-28

+ Fix - `.ipynb` output in tutorials is not visible in dark mode.

## [0.5.1] - 2023-03-15

+ Fix - ingestion routine for multiple Z devices in `prairieviewreader.py`.

## [0.5.0] - 2023-01-09

+ Remove - `recursive_search` function
+ Add - pre-commit checks to the repo to observe flake8, black, isort
+ Add - `value_to_bool` and `QuietStdOut` utilities

## [0.4.2] - 2022-12-16

+ Update - PrairieView loader checks for multi-plane vs single-plane scans.

## [0.4.1] - 2022-12-15

+ Update - PrairieView loader now reads recording start time from metadata file

## [0.4.0] - 2022-12-14

+ Add - mkdocs documentation
+ Add - improved docstrings for mkdocs
+ Add - EXTRACT trigger and loader tools

## [0.3.0] - 2022-10-7

+ Add - Function `prairieviewreader` to parse metadata from Bruker PrarieView acquisition
    system
+ Update - Changelog with tag links

## [0.2.1] - 2022-07-13

+ Add - Adopt `black` formatting
+ Add - Code of Conduct

## [0.2.0] - 2022-07-06

+ First release of `element-interface`.
+ Bugfix - Fix for `tifffile` import.
+ Add - Function `run_caiman` to trigger CNMF algorithm.
+ Add - Function `ingest_csv_to_table` to insert data from CSV files into tables.
+ Add - Function `recursive_search` to search through nested dictionary for a key.
+ Add - Function `upload_to_dandi` to upload Neurodata Without Borders file to the DANDI
    platform.
+ Update - Remove `extras_require` feature to allow this package to be published to PyPI.

## [0.1.0a1] - 2022-01-12

+ Change - Rename the package `element-data-loader` to `element-interface`.

## [0.1.0a0] - 2021-06-21

+ Add - Readers for: `ScanImage`, `Suite2p`, `CaImAn`.


[0.7.0]: https://github.com/datajoint/element-interface/releases/tag/0.7.0
[0.6.0]: https://github.com/datajoint/element-interface/releases/tag/0.6.0
[0.5.4]: https://github.com/datajoint/element-interface/releases/tag/0.5.4
[0.5.3]: https://github.com/datajoint/element-interface/releases/tag/0.5.3
[0.5.2]: https://github.com/datajoint/element-interface/releases/tag/0.5.2
[0.5.1]: https://github.com/datajoint/element-interface/releases/tag/0.5.1
[0.5.0]: https://github.com/datajoint/element-interface/releases/tag/0.5.0
[0.4.2]: https://github.com/datajoint/element-interface/releases/tag/0.4.2
[0.4.1]: https://github.com/datajoint/element-interface/releases/tag/0.4.1
[0.4.0]: https://github.com/datajoint/element-interface/releases/tag/0.4.0
[0.3.0]: https://github.com/datajoint/element-interface/releases/tag/0.3.0
[0.2.1]: https://github.com/datajoint/element-interface/releases/tag/0.2.1
[0.2.0]: https://github.com/datajoint/element-interface/releases/tag/0.2.0
[0.1.0a1]: https://github.com/datajoint/element-interface/releases/tag/0.1.0a1
[0.1.0a0]: https://github.com/datajoint/element-interface/releases/tag/0.1.0a0
