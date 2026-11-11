"""Hand-checkable tests for the metric and post-processing in src/filament.py.

Run:  <ml-lab python> tests/test_filament.py     (also works under pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import filament as F  # noqa: E402


def _pq(pred, gt):
    return F.panoptic_quality(*F.match_instances(pred, gt))


def _sq(shape=(60, 60)):
    return np.zeros(shape, np.int32)


# ------------------------------------------------------------------- the metric
def test_perfect_match():
    gt = _sq(); gt[10:20, 10:20] = 1
    r = _pq(gt.copy(), gt)
    assert r["pq"] == 1.0 and (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)


def test_missed_instance_scores_zero():
    gt = _sq(); gt[10:20, 10:20] = 1
    r = _pq(_sq(), gt)
    assert r["pq"] == 0.0 and r["fn"] == 1


def test_partial_overlap_above_threshold():
    # 10x10 gt, prediction shifted 2px: inter 80, union 120, IoU 2/3 > 0.5
    gt = _sq(); gt[10:20, 10:20] = 1
    pred = _sq(); pred[10:20, 12:22] = 1
    assert abs(_pq(pred, gt)["pq"] - 80 / 120) < 1e-12


def test_overlap_below_threshold_is_fp_plus_fn():
    # shifted 6px: inter 40, union 160, IoU 0.25 -> no match
    gt = _sq(); gt[10:20, 10:20] = 1
    pred = _sq(); pred[10:20, 16:26] = 1
    r = _pq(pred, gt)
    assert r["pq"] == 0.0 and (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)


def test_fragmentation_scores_zero_despite_high_pixel_iou():
    # one 100px filament predicted as two 40px halves: pixel IoU 0.8, but
    # neither half reaches IoU 0.5 alone (0.4), so it is 2 FP + 1 FN
    gt = _sq(); gt[10:20, 10:20] = 1
    pred = _sq(); pred[10:20, 10:14] = 1; pred[10:20, 16:20] = 2
    r = _pq(pred, gt)
    assert r["pq"] == 0.0 and (r["tp"], r["fp"], r["fn"]) == (0, 2, 1)


def test_one_of_two_found():
    gt = _sq(); gt[0:10, 0:10] = 1; gt[30:40, 30:40] = 2
    pred = _sq(); pred[0:10, 0:10] = 1
    r = _pq(pred, gt)
    assert abs(r["pq"] - 1 / 1.5) < 1e-12 and r["sq"] == 1.0


# ---------------------------------------------------------- hysteresis / filters
def _two_blobs():
    """A confident blob with a soft halo, and a blob that never gets confident."""
    p = np.zeros((80, 80), np.float32)
    p[10:30, 10:30] = 0.4          # halo: 400 px above low=0.3
    p[15:25, 15:25] = 0.999        # confident core inside it
    p[50:70, 50:70] = 0.6          # 400 px, never above 0.9
    return p


def test_hysteresis_drops_unconfident_blob_but_keeps_full_extent_of_confident_one():
    inst, n = F.segment_instances(_two_blobs(), low=0.3, high=0.9, min_area=20)
    assert n == 1
    assert (inst > 0).sum() == 400            # the whole halo survives, not just the core
    assert inst[20, 20] > 0 and inst[60, 60] == 0


def test_without_high_both_blobs_survive():
    _, n = F.segment_instances(_two_blobs(), low=0.3, high=None, min_area=20)
    assert n == 2


def test_min_area_removes_specks():
    p = np.zeros((60, 60), np.float32)
    p[5:25, 5:25] = 0.99            # 400 px
    p[40:43, 40:43] = 0.99          # 9 px
    inst, n = F.segment_instances(p, low=0.5, min_area=20)
    assert n == 1 and inst[41, 41] == 0


def test_closing_rejoins_fragments():
    p = np.zeros((60, 60), np.float32)
    p[10:30, 10:20] = 0.99
    p[10:30, 23:33] = 0.99          # 3px gap
    _, n_open = F.segment_instances(p, low=0.5, min_area=20, close=0)
    _, n_closed = F.segment_instances(p, low=0.5, min_area=20, close=3)
    assert (n_open, n_closed) == (2, 1)


def test_component_table_values():
    lab = np.zeros((20, 20), np.int32)
    lab[2:12, 2:5] = 1               # 10 x 3 = 30 px, elongated
    prob = np.where(lab > 0, 0.8, 0.0).astype(np.float32)
    prob[2, 2] = 1.0
    t = F.component_table(lab, 1, prob)
    assert t["area"][1] == 30 and t["max_p"][1] == 1.0
    assert abs(t["mean_p"][1] - (29 * 0.8 + 1.0) / 30) < 1e-6
    assert t["elong"][1] > 2.5       # long thin rectangle


def test_empty_prediction():
    inst, n = F.segment_instances(np.zeros((40, 40), np.float32), low=0.5)
    assert n == 0 and inst.max() == 0


def test_matched_prediction_ids_agrees_with_match_instances():
    gt = _sq(); gt[0:10, 0:10] = 1; gt[30:40, 30:40] = 2
    pred = _sq(); pred[0:10, 0:10] = 1; pred[30:40, 36:46] = 2; pred[50:55, 50:55] = 3
    # 1 matches gt1 exactly; 2 overlaps gt2 only 4x10=40 of union 160 (IoU .25); 3 is spurious
    assert F.matched_prediction_ids(pred, gt) == {1}
    ious, tp, fp, fn = F.match_instances(pred, gt)
    assert (tp, fp, fn) == (1, 2, 1)


def test_matched_prediction_ids_empty():
    assert F.matched_prediction_ids(_sq(), _sq()) == set()


# ------------------------------------------------------- format conversion helpers
def _blob_instance_map():
    inst = np.zeros((200, 300), np.uint16)
    inst[20:60, 30:90] = 1                       # a rectangle
    yy, xx = np.ogrid[:200, :300]
    inst[((yy - 130) ** 2 + (xx - 200) ** 2) <= 35 ** 2] = 2   # a disc
    return inst


def test_rle_round_trip_is_exact():
    from pycocotools import mask as maskUtils
    inst = _blob_instance_map()
    for i in (1, 2):
        rle = F.instance_rle(inst, i)
        assert isinstance(rle["counts"], str) and rle["size"] == [200, 300]
        back = maskUtils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
        assert np.array_equal(back, (inst == i).astype(np.uint8))


def test_polygon_round_trip_recovers_the_mask():
    import cv2
    inst = _blob_instance_map()
    for i in (1, 2):
        polys = F.instance_polygons(inst, i)
        assert len(polys) == 1 and len(polys[0]) >= 6
        canvas = np.zeros(inst.shape, np.uint8)
        pts = np.array(polys[0], dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(canvas, [pts], 1)
        truth = (inst == i)
        iou = (canvas.astype(bool) & truth).sum() / (canvas.astype(bool) | truth).sum()
        assert iou > 0.99, iou


def test_polygons_of_empty_instance_is_empty():
    assert F.instance_polygons(np.zeros((50, 50), np.uint16), 1) == []


# ------------------------------------------------------------- submission CSV
def _fake_results(tmp):
    """A results directory shaped like predict_test_instances() output, 2 images."""
    import json
    names = ["20150125172714Mh.jpeg", "20170501024112Bh.jpeg"]
    a = np.zeros((2048, 2048), np.uint16)
    a[100:140, 200:900] = 1
    a[1500:1530, 300:340] = 2
    a[10:20, 10:20] = 3
    b = np.zeros((2048, 2048), np.uint16)             # an image with no detections
    for nm, inst in zip(names, (a, b)):
        stem = nm.split(".")[0]
        np.savez_compressed(os.path.join(tmp, stem + ".npz"), inst=inst)
        meta = {"file_name": nm, "height": 2048, "width": 2048,
                "instances": [{"id": int(i), "area": int((inst == i).sum()),
                               "mean_p": 0.9, "max_p": 0.99, "bbox_xywh": [0, 0, 1, 1]}
                              for i in np.unique(inst[inst > 0])]}
        with open(os.path.join(tmp, stem + ".json"), "w") as f:
            json.dump(meta, f)
    return names


def test_submission_csv_round_trip():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        names = _fake_results(tmp)
        out = os.path.join(tmp, "sub.csv")
        info = F.write_submission_csv(tmp, out, names)
        assert info["rows"] == 3 and info["images_without_rows"] == ["20170501024112Bh.jpeg"]
        text = open(out, encoding="utf-8").read().split("\n")
        assert text[0] == "filament_id,segmentation_rle"
        assert [l.split(",")[0] for l in text[1:4]] == [
            "20150125172714Mh_1", "20150125172714Mh_2", "20150125172714Mh_3"]
        assert '"' not in "".join(text)
        v = F.verify_submission_csv(out, tmp, names)
        assert v["problems"] == [] and v["mismatched_masks"] == 0
        assert v["images_without_rows"] == ["20170501024112Bh"]


def test_verifier_catches_a_corrupted_mask():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        names = _fake_results(tmp)
        out = os.path.join(tmp, "sub.csv")
        F.write_submission_csv(tmp, out, names)
        rows = F.read_submission_csv(out)
        # swap the masks of instances 1 and 2: still valid RLE, but the wrong pixels
        swapped = [(rows[0][0], rows[1][1]), (rows[1][0], rows[0][1]), rows[2]]
        with open(out, "w") as f:
            f.write("filament_id,segmentation_rle\n")
            for fid, c in swapped:
                f.write(f"{fid},{c}\n")
        v = F.verify_submission_csv(out, tmp, names)
        assert v["mismatched_masks"] == 2 and v["problems"]


def test_verifier_catches_quotes_and_duplicates():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        names = _fake_results(tmp)
        out = os.path.join(tmp, "sub.csv")
        F.write_submission_csv(tmp, out, names)
        rows = F.read_submission_csv(out)
        with open(out, "w") as f:
            f.write("filament_id,segmentation_rle\n")
            f.write(f'{rows[0][0]},"{rows[0][1]}"\n')          # quoted, as the page forbids
            f.write(f"{rows[0][0]},{rows[0][1]}\n")             # duplicate id
        v = F.verify_submission_csv(out, tmp, names)
        assert any("quote" in p for p in v["problems"])
        assert any("duplicate" in p for p in v["problems"])


def test_dice_values():
    gt = {1: np.zeros((20, 20), np.int16), 2: np.zeros((20, 20), np.int16)}
    gt[1][0:10, 0:10] = 1                                   # 100 px
    probs = {1: np.zeros((20, 20), np.float32), 2: np.zeros((20, 20), np.float32)}
    probs[1][0:10, 0:5] = 0.99                              # predicts half of it: 50 px
    # image 1: 2*50/(50+100) = 2/3.  image 2: both empty -> 1.0
    mean, per = F.dice_from_probs([1, 2], probs, None, gt, low=0.5, min_area=1)
    assert abs(per[0] - 2 / 3) < 1e-9 and per[1] == 1.0 and abs(mean - (2 / 3 + 1) / 2) < 1e-9


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
