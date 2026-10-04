#!/usr/bin/env bash
# Reproduction routes for "An exact intraband selection rule exposes the limits of geometric closure in moire fractional Hall fluids".
#   ./reproduce.sh light   : regenerate every table, figure and number of the manuscript from the stored raw results
#                            (minutes on a laptop; requires python3 with numpy, scipy, numba, joblib, matplotlib; pdflatex)
#   ./reproduce.sh full    : rerun all exact-diagonalization calculations first (about 15-20 core-hours), then 'light'
#   ./reproduce.sh test    : run the unit tests of the sewn non-Abelian Chern routine (a few minutes)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# relocate hard-coded paths of the original workstation to this archive
sed -i "s#/home/claude/sim#$ROOT/code#g; s#/home/claude/build2#$ROOT/build#g; s#/home/claude/build/si_body.tex#$ROOT/build/si_body.tex#g" "$ROOT"/code/*.py "$ROOT"/build/*.py
cd "$ROOT/code"
if [ "${1:-light}" = "test" ]; then python3 tests/test_chern.py; exit 0; fi
if [ "${1:-light}" = "full" ]; then
  python3 prod2.py specs2.json 2
  python3 extra2.py; python3 run9.py A; python3 run9.py P3; python3 dichC.py; python3 sflow.py
  python3 dyn.py; python3 repscan.py 6 1; python3 repscan2.py 0; python3 repscan2.py 1
  python3 newcalc.py quick; python3 newcalc.py n8; python3 wse2.py 2.0 2.5 3.0; python3 mixest.py
  python3 etascan.py; python3 phiunc.py; python3 cspec.py; python3 chern_na.py conv; python3 chern_na.py all
  python3 stats2.py; python3 repsens.py
fi
for s in analysis2.py tables2.py numbers2.py figs2.py figrep.py 'figdyn.py 7' numbers3.py numbers4.py post_r4.py post_r5.py llval.py; do
  python3 $s
done
cd "$ROOT/build" && python3 assemble2.py && python3 assemble_prx.py
for i in 1 2 3; do pdflatex -interaction=nonstopmode sm_prx.tex >/dev/null; pdflatex -interaction=nonstopmode main_prx.tex >/dev/null; done
python3 freeze_xr.py   # self-contained upload versions main_prx_upload.tex / sm_prx_upload.tex
echo "main_prx.pdf and sm_prx.pdf regenerated in $ROOT/build (requires REVTeX 4.2)"
