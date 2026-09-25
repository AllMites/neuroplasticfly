# Christie et al. 2026 supplement (local copy)

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12869359/ (PMC12869359, Current Biology).
Fetched 2026-09-23 into this directory so that `christie_reconciliation_2026-09-23.md`
keeps its sources after the session scratchpad is gone.

**The files themselves are not redistributed here.** They are a third party's article
text and supplementary data; redistribution terms are theirs, not this repository's, and
nothing in this repository's MIT licence extends to them. Fetch them from the PMC link
above into this directory if you want to re-check the reconciliation.

| File | Content |
|---|---|
| `pmc.txt` | full text of the PMC article, plain text |
| `NIHMS2134081-supplement-2.xlsx` | Data S2 - Fox (CB0525) connectivity, incl. S2A Fox -> FDA synapse counts |
| `NIHMS2134081-supplement-3.xlsx` | Data S3 - simulation results, responsive-PAM counts per Wsyn / GRN rate |
| `NIHMS2134081-supplement-4.xlsx` | Data S4 - FlyWire root IDs of the modelled cells |
| `NIHMS2134081-supplement-5.xlsx` | Data S5 - silencing conditions (Fox, CB0233, FDA-I, FDA-II, FDA-all) |
| `NIHMS2134081-supplement-6.pdf` | Data S6 - supplemental figures. 6.5 MB, **gitignored** (see `.gitignore` in this dir); re-fetch from PMC if needed |

## Gotchas when reading these

- **Data S2A column C has Excel date-mangled cells.** Cells that should read
  `10/1`, `9/1`, `6/1` (counts of the form "n/m") were auto-converted to dates
  and display as `2000-10-01`, `2000-09-01`, `2000-06-01`. Read the numerator.
- Root IDs in Data S4 all resolve in FlyWire **v783**; the paper does not state
  a version. Fox -> CB0233 = 217 synapses matches v783 locally, so v783 is what
  this reconciliation assumes.
- Christie's "responsive PAM" criterion is a **count of PAM-DANs with nonzero
  mean rate** over 30 x 1000 ms trials - not a Hz threshold. It is much more
  lenient than flychess's usual `>= 1 Hz` silence cut.
