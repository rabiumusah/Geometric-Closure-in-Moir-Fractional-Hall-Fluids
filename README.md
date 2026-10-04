# Geometric-Closure-in-Moir-Fractional-Hall-Fluids
An exact intraband selection rule exposes the limits of geometric closure in moiré fractional Hall fluids M. Rabiu

It contains every script, parameter file (specs2.json, partners.json, match_out.json, registry.json),
the unit tests, the build scripts and reproduce.sh, but not the stored raw results (code/res2, res2x, res3,
res4 and the *.pkl caches, about 26 MB) or the manuscript sources.

  ./reproduce.sh test   works as is (unit tests of the non-Abelian Chern routine).
  ./reproduce.sh full   recomputes all raw results (15-20 core-hours); to also rebuild the PDFs, copy the
                        manuscript sources (build/*.tex, build/fig, build/numbers.json) from the full archive.
  ./reproduce.sh light  needs the stored results: use the full archive fqah-closure-v1.1-prx.zip instead.
  python3 code/llval.py reproduces Table S2 / Fig. S1 from scratch in about a minute (writes into build/).
