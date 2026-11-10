# Bibliography tools

`references.bib` contains peer-reviewed works only (arXiv-only preprints are not cited).

- Entries with a DOI were generated from the publishers' records in Crossref:
  `python tools/make_bib.py tools/dois.tsv crossref.bib`, then lightly normalised
  (venue names of CVPR/ICCV before 2018/2019, the ICPR 2024 proceedings name and volume, umlauts).
- Entries without a DOI (NeurIPS, ICLR, ICML, TMLR, AISTATS, BMVC; Koch & Ullman 1985 via PubMed)
  were written by hand and checked against Semantic Scholar records that carry DBLP proceedings keys.
