"""Compare l'API locale et l'API cloud : cohérence des prédictions et temps de réponse.
Usage : python tests/integration_check.py https://...-7860.app.github.dev
"""
import sys
import time

import requests

LOCAL = "http://127.0.0.1:8000"
COMMENTS = ["I love this video, amazing!", "Worst tutorial ever, waste of time",
            "Posted at 10am", "Not bad but could be better", "This is so helpful thank you"]


def call(base):
    t0 = time.perf_counter()
    r = requests.post(f"{base}/predict_batch", json={"comments": COMMENTS}, timeout=60)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t0) * 1000


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage : python tests/integration_check.py <URL_CLOUD>")
    cloud = sys.argv[1].rstrip("/")
    for base in (LOCAL, cloud):
        h = requests.get(f"{base}/health", timeout=60)
        print(f"{base}/health -> {h.status_code} {h.json()}")

    local_res, t_local = call(LOCAL)
    cloud_res, t_cloud = call(cloud)
    print(f"\nTemps aller-retour : local {t_local:.0f} ms | cloud {t_cloud:.0f} ms")
    print(f"Temps de calcul API : local {local_res['processing_time_ms']} ms | "
          f"cloud {cloud_res['processing_time_ms']} ms\n")

    same = True
    for a, b in zip(local_res["predictions"], cloud_res["predictions"]):
        ok = a["label"] == b["label"] and abs(a["confidence"] - b["confidence"]) < 1e-3
        same &= ok
        print(f"{'OK  ' if ok else 'DIFF'} | local {a['sentiment']:<8} {a['confidence']:.3f} | "
              f"cloud {b['sentiment']:<8} {b['confidence']:.3f} | {a['text'][:35]}")
    print("\nRésultats identiques entre local et cloud." if same else "\nDifférences détectées.")


if __name__ == "__main__":
    main()