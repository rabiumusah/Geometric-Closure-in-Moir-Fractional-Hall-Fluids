# Geometric-Closure-in-Moir-Fractional-Hall-Fluids
An exact intraband selection rule exposes the limits of geometric closure in moiré fractional Hall fluids M. Rabiu, S. S. Abukari
and M. Amekewu 
This archive contains all code, parameter files and raw result arrays behind every figure, table and quoted number of the article and
its Supplemental Material, together with the LaTeX sources. 
Quick start 
./reproduce.sh./reproduce.sh./reproduce.sh
testlightfull# unit tests of the sewn non-Abelian Chern-number routine (a few minutes)
# regenerate all tables, figures, numbers and both PDFs from stored results (minutes)
# rerun every calculation first (15-20 core-hours on two cores), then 'light'
Requirements: Python 3.10+, numpy, scipy, numba, joblib, matplotlib; pdflatex with REVTeX 4.2. SHA256SUMS.txt lists checksums
of the 322 files under code/ (verify with sha256sum -c SHA256SUMS.txt). 
