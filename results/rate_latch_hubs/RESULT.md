# RESULT: latch vs Li et al. hub overlap (paper-1 phase 3, descriptive)

Rule (written with the numbers in one commit; timing not checkable): latched (persistence 100-200 ms post-offset, not the suite 500-1000 ms window) = seed-mean rate > 1 Hz over the last 100 ms (bins 5-9) of the 200 ms off period; an odour latches if > 100 neurons do. Test set = hub neurons (cell_type in ['KCab', 'L2', 'LPLC2', 'LC9', 'PFNv']) UNION their direct pre/post partners (synapse floor 5, brain_gpu.npz). Per odour: enrichment = P(in set | latched) / P(in set | brain), hypergeometric upper-tail p. ENRICHED if, in BOTH LIF regimes, median ratio >= 1.5 and p < 0.01 in >= 80% of latching odours; NOT-ENRICHED if median ratio <= 1.2 in both; otherwise INCONCLUSIVE.

Hub types (PRD list; Li et al. text not accessible, HTTP 403): KCab, L2, LPLC2, LC9, PFNv; counts {'KCab': 1643, 'L2': 1699, 'LPLC2': 210, 'LC9': 179, 'PFNv': 21}. Hub neurons 3752, partners 24123 (floor 5), set = 20.0% of brain.
Rate C0 not analysed: only latch counts are stored, no per-neuron arrays (would need a forward rate run).

| regime | odours latching | median n latched | enrichment (hub+partner) | enrichment excl. KC | KC share of latched | odours p<0.01 | hub neurons latched (median % of hubs) | base frac in set | median partners latched | MBON06 latched (odours) |
|---|---|---|---|---|---|---|---|---|---|---|
| stock | 39/53 | 5324 | 0.82 | 0.40 | 0.32 | 0% | 16% | 0.200 | 281 | 0/39 |
| eln8 | 27/53 | 661 | 0.19 | 0.20 | 0.00 | 0% | 0% | 0.200 | 25 | 0/27 |

**VERDICT (LIF reference only, both regimes; rate C0 not tested): NOT-ENRICHED for the hub+partner union.** Decomposition (post hoc, not in rule; independent evaluator 2026-10-01): stock hub-only enrichment 4.16x (p<0.01 in 38/39 odours), entirely KCab, which is 1.10x within KCs, i.e. a KC effect, not a hub effect; partners depleted (0.31 stock, 0.22 eln8). eln8: 0 hubs latched. Hub list from PRD, not verified against Li et al. (HTTP 403); KCab-p (128) excluded, and including it does not change the verdict (0.83/0.19).
