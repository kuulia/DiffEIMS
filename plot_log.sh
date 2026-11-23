#!/usr/bin/env bash
# ASCII plots of val NLL / edge-type metrics from a DiffMS training log.
# Usage: ./plot_log.sh outfiles/ddp_XXXX.out [HEIGHT] [WIDTH]
#        watch -n 60 ./plot_log.sh outfiles/ddp_XXXX.out
# Only needs bash + any POSIX awk. Plots only validations from completed
# training epochs (no sanity check, no test phase). More epochs than
# WIDTH columns -> each column is a bin: "*" = mean, ":" = min..max of the
# bin (so single-epoch spikes stay visible). nan/inf points are skipped.
# Prints run name, dataset, encoder/decoder and output dir from the config
# first. If the log has a test table, top-1/10/100 accuracy + tanimoto and validity
# are printed after the plots.
f=${1:?usage: $0 logfile [height] [width]}
[ -r "$f" ] || { echo "cannot read $f" >&2; exit 1; }
H=${2:-10}
W=${3:-$(( $(tput cols 2>/dev/null || echo 100) - 26 ))}

awk -v H="$H" -v W="$W" '
function add(k, v) {
  if (v !~ /^-?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$/) { bad[k]++; return }
  n[k]++; d[k, n[k]] = v + 0
}
function grab(s, key,   i, t) {      # value following "key" in s, or "" if absent
  if (!(i = index(s, key))) return ""
  t = substr(s, i + length(key)); sub(/ .*/, "", t); return t
}
function pyval(s, key,   i, t) {  # value after key in a printed python dict
  if (!(i = index(s, key))) return "n/a"
  t = substr(s, i + length(key))
  if (t ~ /^\047/) { t = substr(t, 2); sub(/\047.*/, "", t) } else sub(/[,}].*/, "", t)
  return t
}
# run info: first printed config dict (starts with general/name) and output dir
!cfg && index($0, "{\047general\047: {") {
  cfg = 1
  run = pyval($0, "\047general\047: {\047name\047: "); ds = pyval($0, "\047dataset\047: {\047name\047: ")
  enc = pyval($0, "\047encoder\047: ");                dec = pyval($0, "\047decoder\047: ")
}
# fallback when the dict line was cut by output of another rank (resumed runs
# log the checkpoint cfg here, so it is only used if the dict lacks it)
yds == "" && prev ~ /Dataset config:$/ && /^name: / { yds = substr($0, 7) }
{ prev = $0 }
!outdir && /Output directory: / { outdir = $0; sub(/.*Output directory: /, "", outdir) }
function parse(s,   v) {
  add("val NLL", grab(s, "Val NLL "))
  if ((v = grab(s, "Edge type KL: "))   != "") add("val Edge KL", v)
  if ((v = grab(s, "Edge type CE: "))   != "") add("val Edge CE", v)
  if ((v = grab(s, "Edge type logp: ")) != "") add("val Edge logp", v)
}
function epoch(s) { sub(/.*Epoch /, "", s); sub(/:.*/, "", s); return s }
# Only monitor training: a val line counts only once the train summary of the
# same epoch follows it (this drops the sanity check, which has none), and
# everything after Trainer.fit stops (the test phase) is ignored.
/`Trainer.fit` stopped/ { stopped = 1; next }
# Lightning test table rows: "│  test/acc_at_1  │  0.2513...  │"
$2 ~ /^test\/[A-Za-z0-9_]+$/ { t[substr($2, 6)] = $4; next }
stopped { next }
/Epoch [0-9]+: Val NLL/ { pend = $0; pep = epoch($0); next }
/Epoch [0-9]+: X_CE/ {
  if (pend != "" && pep == epoch($0)) parse(pend)
  pend = ""
  add("train E_CE", grab($0, "E_CE: "))
}
function lv(v, lo, hi) { return (hi == lo) ? 0 : int((v - lo) / (hi - lo) * (H - 1) + 0.5) }
function plot(k,   N, cols, c, i, lo, hi, a, b, s, y, ymin, ymax, r, row, best, bi, sgn, lvl, l0, l1, ch) {
  N = n[k]; if (!N) return
  cols = (N < W) ? N : W
  for (c = 1; c <= cols; c++) {   # bin N points into cols columns: mean + min/max
    a = int((c - 1) * N / cols) + 1; b = int(c * N / cols); s = 0
    ymin[c] = ymax[c] = d[k, a]
    for (i = a; i <= b; i++) {
      s += d[k, i]
      if (d[k, i] < ymin[c]) ymin[c] = d[k, i]
      if (d[k, i] > ymax[c]) ymax[c] = d[k, i]
    }
    y[c] = s / (b - a + 1)
    if (c == 1 || ymin[c] < lo) lo = ymin[c]
    if (c == 1 || ymax[c] > hi) hi = ymax[c]
  }
  sgn = (k ~ /logp/) ? -1 : 1                     # logp: higher is better
  best = d[k, 1]; bi = 1
  for (i = 2; i <= N; i++) if (sgn * d[k, i] < sgn * best) { best = d[k, i]; bi = i }
  printf "\n%s  (n=%d, last=%g, best=%g @%d)%s\n", k, N, d[k, N], best, bi, \
         bad[k] ? sprintf("  [%d nan/inf skipped]", bad[k]) : ""
  for (c = 1; c <= cols; c++) { lvl[c] = lv(y[c], lo, hi); l0[c] = lv(ymin[c], lo, hi); l1[c] = lv(ymax[c], lo, hi) }
  for (r = H - 1; r >= 0; r--) {   # "*" bin mean, ":" spread of epochs in the bin
    row = ""
    for (c = 1; c <= cols; c++) { ch = (lvl[c] == r) ? "*" : (r >= l0[c] && r <= l1[c]) ? ":" : " "; row = row ch }
    printf "%10s |%s\n", (r == H - 1) ? sprintf("%.4g", hi) : (r == 0) ? sprintf("%.4g", lo) : "", row
  }
  row = ""; for (c = 1; c <= cols; c++) row = row "-"
  printf "%10s +%s  (epochs 1..%d)\n", "", row, N
}
END {
  if (ds == "n/a" && yds != "") ds = yds "  (from Dataset config block; dict line truncated)"
  if (cfg || outdir) {
    printf "%-8s %s\n%-8s %s\n%-8s %s\n%-8s %s\n%-8s %s\n", "run:", cfg ? run : "n/a", "dataset:", cfg ? ds : "n/a", \
           "encoder:", cfg ? enc : "n/a", "decoder:", cfg ? dec : "n/a", "output:", outdir ? outdir : "n/a"
  }
  split("val NLL|val Edge KL|val Edge CE|val Edge logp|train E_CE", ks, "|")
  for (j = 1; j <= 5; j++) { plot(ks[j]); any += n[ks[j]] }
  if (!any) print "no completed training epochs in " FILENAME
  if (("validity" in t) || ("acc_at_1" in t)) {
    printf "\ntest metrics:\n%10s %9s %9s %9s\n", "", "top-1", "top-10", "top-100"
    printf "%10s %9s %9s %9s\n", "accuracy", fmt("acc_at_1", 1), fmt("acc_at_10", 1), fmt("acc_at_100", 1)
    printf "%10s %9s %9s %9s\n", "tanimoto", fmt("tanimoto_at_1"), fmt("tanimoto_at_10"), fmt("tanimoto_at_100")
    printf "%10s %9s\n", "validity", fmt("validity", 1)
  }
}
function fmt(k, pct) {             # accuracy/validity: "25.6 %", tanimoto: "0.44"
  if (!(k in t)) return "n/a"
  if (t[k] !~ /^-?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$/) return t[k]   # nan etc.
  return pct ? sprintf("%.1f %%", 100 * t[k]) : sprintf("%.2f", t[k])
}' "$f"
