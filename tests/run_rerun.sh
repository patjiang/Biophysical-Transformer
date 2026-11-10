#!/usr/bin/env bash
# Restricted run: arm delta on the full test set and with corpus-overlap
# proteins removed, for every split and head, then plot.
#
#   bash run_rerun.sh            # everything
#   bash run_rerun.sh --dry      # check paths and arms, run nothing
set -u

SPLITS="human human_cell mixed_split"
CALL=0.80
NBOOT=4000
OUT=rerun
CANDS=overlap_audit/candidates.csv
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

fail=0
echo "== paths =="
for S in $SPLITS; do
  for D in res_novox_${S}_e79 res_voxel_${S}_e75 mt_novox_${S}_e79 mt_voxel_${S}_e75; do
    if [ -d "$D" ]; then printf '  ok      %s\n' "$D"
    else printf '  MISSING %s\n' "$D"; fail=1; fi
  done
done
[ -f "$CANDS" ] || { echo "  MISSING $CANDS -- run overlap_audit.py first"; fail=1; }
[ $fail -eq 1 ] && { echo "fix the above first"; exit 1; }

echo "== arms =="
HDR=$(head -1 res_voxel_human_e75/test_predictions.csv)
ARMS=""
for A in attn cnn ridge_bos ridge_mean ridge_stat; do
  case ",$HDR," in
    *",$A,"*) ARMS="$ARMS $A"; printf '  found   %s\n' "$A" ;;
    *)        printf '  absent  %s\n' "$A" ;;
  esac
done
[ -z "$ARMS" ] && { echo "no known arm columns in $HDR"; exit 1; }
echo "  running:$ARMS"

N=0
for S in $SPLITS; do for A in $ARMS; do N=$((N+2)); done; done
echo "== $N stack_check runs at --n-boot $NBOOT =="
[ $DRY -eq 1 ] && { echo "(dry run, stopping)"; exit 0; }

mkdir -p "$OUT"
for S in $SPLITS; do
  R_NOV=res_novox_${S}_e79;  M_NOV=mt_novox_${S}_e79
  R_VOX=res_voxel_${S}_e75;  M_VOX=mt_voxel_${S}_e75
  for A in $ARMS; do
    echo; echo "---- $S / $A / full ----"
    python stack_check.py --arm "$A" \
        --run novox "$R_NOV" "$M_NOV" \
        --run voxel "$R_VOX" "$M_VOX" \
        --n-boot $NBOOT --split-name "$S" --label full \
        --json "$OUT/${S}_${A}_full.json" || echo "  FAILED (continuing)"

    echo; echo "---- $S / $A / restricted ----"
    python stack_check.py --arm "$A" \
        --run novox "$R_NOV" "$M_NOV" \
        --run voxel "$R_VOX" "$M_VOX" \
        --exclude "$CANDS" --exclude-split "$S" --exclude-call $CALL \
        --n-boot $NBOOT --split-name "$S" --label restricted \
        --json "$OUT/${S}_${A}_restricted.json" || echo "  FAILED (continuing)"
  done
done

echo; echo "== plotting =="
python plot_rerun.py --json "$OUT"/*.json --out "$OUT"
