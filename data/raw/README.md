# Raw data

Run `python scripts/fetch_princeton_raw.py` to create
`princeton_dark_patterns_raw.json` from the pinned, public
`yamanalab/ec-darkpattern` TSV. The fetcher preserves original text, source page
IDs, and categories while mapping only the three requested dark-pattern
categories and genuine `Not Dark Pattern` examples.

Do not add synthetic rows or alter the text to simulate markup. Keep raw data
separate from generated cleaned splits.
