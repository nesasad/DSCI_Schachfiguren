"""
Clustering von Bildausschnitten (z. B. Schachfeldern) mit DINOv2-Einbettungen.
 
Ablauf
1. Jedes Bild wird von facebook/dinov2-base in einen Vektor umgewandelt
   (Einbettung): CLS-Token und Mittelwert der Patch-Tokens, zusammen 1536 Zahlen,
   danach auf Laenge 1 normiert. Aehnliche Bilder haben aehnliche Vektoren.
   DINOv2 ist selbstueberwacht trainiert und braucht keine Beschriftungen.
2. k-means gruppiert die Vektoren in k Cluster.
3. Je Cluster entsteht ein Kontaktbogen (Bilder nahe am Clusterzentrum zuerst),
   dazu eine CSV mit Datei, Cluster und Abstand zum Zentrum. Optional werden die
   Bilder in Ordner pro Cluster kopiert, damit man sie von Hand benennen kann.
 
Hinweise
- Bei Schachfeldern bildet DINOv2 oft zusaetzlich nach Feldfarbe (hell/dunkel)
  Cluster. Darum k etwas groesser waehlen als die Zahl der Klassen: 13 Klassen
  (6 Figurenarten x 2 Farben + leer), mit Feldfarbe bis etwa 26 Cluster.
- Der Standard-Prozessor von DINOv2 skaliert auf 256 Pixel und schneidet 224
  Pixel aus der Mitte. Bei quadratischen Feldern fallen nur die Raender weg.
 
Aufruf
    python cluster_dinov2.py "<ordner mit bildern>" --k 13
    python cluster_dinov2.py "<ordner>" --k-bereich 10 30      # k per Silhouette waehlen
    python cluster_dinov2.py "<ordner>" --k 20 --sortieren     # Bilder in Cluster-Ordner kopieren
 
Pakete: torch, transformers, scikit-learn, pandas, pillow, tqdm
"""
import argparse
import os
import shutil
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel
 
MODELL = "facebook/dinov2-base"
ENDUNGEN = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
 
 
def finde_bilder(ordner):
    """Alle Bilddateien im Ordner und in Unterordnern, sortiert."""
    return sorted(p for p in Path(ordner).rglob("*") if p.suffix.lower() in ENDUNGEN)
 
 
@torch.no_grad()
def einbettungen(pfade, batch=64, modell=MODELL, device=None):
    """Berechnet normierte DINOv2-Vektoren, Form (Anzahl Bilder, 1536)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Auf dem gemeinsam genutzten Rechner mindestens 2 Kerne freilassen
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    prozessor = AutoImageProcessor.from_pretrained(modell)
    netz = AutoModel.from_pretrained(modell).to(device).eval()
 
    teile = []
    for i in tqdm(range(0, len(pfade), batch), desc="Einbettungen"):
        bilder = []
        for p in pfade[i:i + batch]:
            with Image.open(p) as im:
                bilder.append(im.convert("RGB"))
        eingabe = prozessor(images=bilder, return_tensors="pt").to(device)
        h = netz(**eingabe).last_hidden_state          # (batch, 1 + Patches, 768)
        vektor = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
        teile.append(vektor.float().cpu().numpy())
    x = np.concatenate(teile)
    return x / np.linalg.norm(x, axis=1, keepdims=True)
 
 
def lade_oder_berechne(pfade, cache, batch):
    """Einbettungen aus dem Cache laden, falls Dateiliste unveraendert."""
    namen = np.array([str(p) for p in pfade])
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if len(d["namen"]) == len(namen) and (d["namen"] == namen).all():
            print("Einbettungen aus Cache:", cache)
            return d["x"]
    x = einbettungen(pfade, batch=batch)
    np.savez(cache, x=x, namen=namen)
    return x
 
 
def clustere(x, k, seed=0):
    """k-means. Rueckgabe: Labels und Abstand jedes Bildes zu seinem Zentrum."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    abstand = np.linalg.norm(x - km.cluster_centers_[km.labels_], axis=1)
    return km.labels_, abstand
 
 
def waehle_k(x, von, bis, seed=0):
    """Silhouetten-Koeffizient fuer k = von..bis (hoeher = klarer getrennt)."""
    werte = {}
    for k in range(von, bis + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(x)
        werte[k] = silhouette_score(x, labels)
        print(f"k = {k:2d}  Silhouette = {werte[k]:.3f}")
    bestes = max(werte, key=werte.get)
    print("Bestes k nach Silhouette:", bestes, "(nur ein Anhaltspunkt, Kontaktboegen ansehen)")
    return bestes
 
 
def kontaktboegen(pfade, labels, abstand, ausgabe, pro_cluster=36, spalten=12, kachel=80):
    """Pro Cluster ein PNG mit den zentrumsnaechsten Bildern."""
    ausgabe.mkdir(parents=True, exist_ok=True)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(abstand[idx])][:pro_cluster]
        zeilen = int(np.ceil(len(idx) / spalten))
        blatt = Image.new("RGB", (spalten * kachel, zeilen * kachel), "white")
        for n, i in enumerate(idx):
            with Image.open(pfade[i]) as im:
                t = im.convert("RGB").resize((kachel, kachel))
            blatt.paste(t, ((n % spalten) * kachel, (n // spalten) * kachel))
        blatt.save(ausgabe / f"cluster_{c:02d}.png")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ordner", help="Ordner mit den Bildern (auch Unterordner)")
    ap.add_argument("--k", type=int, default=13, help="Anzahl Cluster (Standard 13)")
    ap.add_argument("--k-bereich", type=int, nargs=2, metavar=("VON", "BIS"),
                    help="k per Silhouette aus diesem Bereich waehlen")
    ap.add_argument("--ausgabe", help="Ausgabeordner (Standard: neben dem Bilderordner)")
    ap.add_argument("--sortieren", action="store_true", help="Bilder in Cluster-Ordner kopieren")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
 
    quelle = Path(a.ordner)
    ausgabe = Path(a.ausgabe) if a.ausgabe else quelle.parent / "cluster_dinov2"
    ausgabe.mkdir(parents=True, exist_ok=True)
 
    pfade = finde_bilder(quelle)
    if not pfade:
        raise SystemExit("Keine Bilder in " + str(quelle))
    print(len(pfade), "Bilder")
 
    x = lade_oder_berechne(pfade, ausgabe / "einbettungen.npz", a.batch)
 
    k = waehle_k(x, *a.k_bereich) if a.k_bereich else a.k
    labels, abstand = clustere(x, k)
 
    tabelle = pd.DataFrame({"datei": [str(p.relative_to(quelle)) for p in pfade],
                            "cluster": labels, "abstand": abstand.round(4)})
    tabelle.to_csv(ausgabe / "cluster.csv", index=False)
    print(tabelle["cluster"].value_counts().sort_index().to_string())
 
    kontaktboegen(pfade, labels, abstand, ausgabe)
 
    if a.sortieren:
        for p, c in zip(pfade, labels):
            ziel = ausgabe / f"cluster_{c:02d}"
            ziel.mkdir(exist_ok=True)
            shutil.copy2(p, ziel / p.name)
 
    print("Ergebnis in", ausgabe)
 
 
if __name__ == "__main__":
    main()
 
 