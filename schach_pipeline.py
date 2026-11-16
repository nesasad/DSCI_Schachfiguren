"""
Schach-Bilderkennung: Brett zuschneiden -> Felder -> Figuren einzeln -> Baumdiagramm

Aufruf:  python schach_pipeline.py brett.jpeg  [--out ausgabe]

Schritte
  1. Brett finden    : äusseren Rahmen als Viereck erkennen, perspektivisch entzerren,
                       dann das 8x8-Raster über Kanten-Projektionen exakt finden
                       und alles ausserhalb (Holzrahmen + Koordinatenrand) wegschneiden.
  2. Felder          : das Brett in 64 Felder teilen, Belegung über Kantendichte bestimmen.
  3. Figuren einzeln : jede Figur mit GrabCut freistellen und als PNG (transparenter
                       Hintergrund) speichern; Farbe (weiss/schwarz) über Farbanalyse.
  4. Figurentyp      : k-NN-Klassifikator auf HOG-Merkmalen + Silhouetten-Höhe.
                       Trainingsdaten = Bilder in der Grundstellung (automatisch gelabelt).
  5. Baumdiagramm    : Brett -> Farbe -> Figurentyp -> Feld (mit Bildern), via Graphviz.
"""
import argparse, json, os, subprocess
import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

NAMEN = {"K": "König", "D": "Dame", "T": "Turm", "L": "Läufer", "S": "Springer", "B": "Bauer"}
GRUNDSTELLUNG = {  # Rang 1/8 Figuren von Linie a..h
    1: "TSLDKLST", 8: "TSLDKLST"}
WARP = 900  # Kantenlänge des entzerrten Rahmenbilds


# ---------------------------------------------------------------- 1. Brett
def order_pts(p):
    p = np.array(p, dtype=np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)


def finde_rahmen(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.dilate(cv2.Canny(g, 50, 150), np.ones((3, 3)))
    cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(cs, key=cv2.contourArea, reverse=True):
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(ap) == 4 and cv2.contourArea(ap) > 0.15 * img.shape[0] * img.shape[1]:
            best = ap.reshape(4, 2)
            break
    if best is None:  # Fallback: ganzes Bild
        h, w = img.shape[:2]
        best = [[0, 0], [w, 0], [w, h], [0, h]]
    return order_pts(best)


def raster_linien(profil, n=9):
    """Sucht n gleichabständige Linien mit maximaler Kantenstärke."""
    pk, _ = find_peaks(profil, distance=len(profil) // 20, height=profil.max() * 0.2)
    diffs = np.diff(pk)
    m = np.median(diffs[(diffs > len(profil) / 14) & (diffs < len(profil) / 7)])
    best, bestscore = None, -1
    for d in np.linspace(0.92 * m, 1.08 * m, 33):
        for s in range(0, int(len(profil) - 8 * d)):
            pos = s + d * np.arange(n)
            idx = np.clip(pos.astype(int), 0, len(profil) - 1)
            score = sum(profil[max(i - 2, 0):i + 3].max() for i in idx)
            if score > bestscore:
                best, bestscore = pos, score
    return best


def schneide_brett(img, out):
    rahmen = finde_rahmen(img)
    M1 = cv2.getPerspectiveTransform(rahmen, np.float32([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]]))
    w = cv2.warpPerspective(img, M1, (WARP, WARP))
    g = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).astype(float)
    gx = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3))
    # mittlere Reihen sind meist leer -> sauberes Kantensignal
    c0, c1 = int(WARP * .35), int(WARP * .65)
    xs = raster_linien(gx[c0:c1, :].sum(0))
    ys = raster_linien(gy[:, c0:c1].sum(1))
    # Rasterecken zurück ins Originalbild -> zweite, exakte Entzerrung
    ecken_w = np.float32([[xs[0], ys[0]], [xs[-1], ys[0]], [xs[-1], ys[-1]], [xs[0], ys[-1]]])
    ecken = cv2.perspectiveTransform(ecken_w[None], np.linalg.inv(M1))[0]
    S = 100  # Pixel pro Feld
    M2 = cv2.getPerspectiveTransform(ecken, np.float32([[0, 0], [8 * S, 0], [8 * S, 8 * S], [0, 8 * S]]))
    brett = cv2.warpPerspective(img, M2, (8 * S, 8 * S))
    # Rand-Variante (für Figuren, die über ihr Feld hinausragen)
    P = 90
    M3 = cv2.getPerspectiveTransform(ecken, np.float32([[P, P], [8 * S + P, P], [8 * S + P, 8 * S + P], [P, 8 * S + P]]))
    brett_pad = cv2.warpPerspective(img, M3, (8 * S + 2 * P, 8 * S + 2 * P), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(f"{out}/1_brett_zugeschnitten.png", brett)
    dbg = img.copy()
    cv2.polylines(dbg, [rahmen.astype(int)], True, (0, 0, 255), 4)
    cv2.polylines(dbg, [ecken.astype(int)], True, (0, 255, 0), 4)
    cv2.imwrite(f"{out}/0_erkennung.png", dbg)
    return brett, brett_pad, S, P


# ---------------------------------------------------------------- Orientierung
def orientierung(brett, S):
    """Liefert Funktion (zeile, spalte) -> Feldname. Weiss oben oder unten wird erkannt.
    a1 ist immer ein dunkles Feld -> bestimmt, ob die Linien gespiegelt sind."""
    hell = lambda r, c: brett[r*S+10:(r+1)*S-10, c*S+10:(c+1)*S-10].mean()
    oben_weiss = cream_anteil(brett[:2*S]) > cream_anteil(brett[-2*S:])
    # Feld unten links (Zeile 7, Spalte 0) aus Sicht des Bildes
    def name(r, c):
        rang = r + 1 if oben_weiss else 8 - r
        # Regel: a1 dunkel. Prüfen anhand leerer Mittelreihen (Zeile 3/4)
        return rang
    # Parität: Feld (r,c) dunkel <=> (Linie+Rang) gerade (a=1). Finde Parität aus Feld (3,0)
    dunkel_30 = hell(3, 0) < hell(3, 1)
    rang30 = name(3, 0)
    # wenn Linie a bei Spalte 0: linie=1 -> dunkel <=> (1+rang)%2==0
    a_links = dunkel_30 == ((1 + rang30) % 2 == 0)
    def feld(r, c):
        linie = c if a_links else 7 - c
        return "abcdefgh"[linie] + str(name(r, c))
    return feld, oben_weiss


def cream_anteil(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(((lab[..., 0] > 180) & (lab[..., 2] > 138)).mean())


# ---------------------------------------------------------------- 2./3. Felder + Figuren
def analysiere_felder(brett, brett_pad, S, P, out):
    os.makedirs(f"{out}/felder", exist_ok=True)
    os.makedirs(f"{out}/figuren", exist_ok=True)
    feld, oben_weiss = orientierung(brett, S)
    gray = cv2.cvtColor(brett, cv2.COLOR_BGR2GRAY)
    m = 18
    belegt = [[cv2.Canny(cv2.GaussianBlur(gray[r*S+m:(r+1)*S-m, c*S+m:(c+1)*S-m], (3, 3), 0), 40, 120).mean() / 255 > 0.01
               for c in range(8)] for r in range(8)]
    modell = leeres_brett_modell(brett_pad, belegt, S, P)
    ergebnisse = []
    for r in range(8):
        for c in range(8):
            name = feld(r, c)
            f = brett[r*S:(r+1)*S, c*S:(c+1)*S]
            cv2.imwrite(f"{out}/felder/{name}.png", f)
            m = 18
            kern = gray[r*S+m:(r+1)*S-m, c*S+m:(c+1)*S-m]
            kanten = cv2.Canny(cv2.GaussianBlur(kern, (3, 3), 0), 40, 120).mean() / 255
            belegt = kanten > 0.01  # leere Felder liegen bei ~0.0
            eintrag = {"feld": name, "zeile": r, "spalte": c, "kanten": round(kanten, 4), "belegt": bool(belegt)}
            if belegt:
                eintrag.update(figur_ausschneiden(brett_pad, modell, r, c, S, P, name, out))
            ergebnisse.append(eintrag)
    return ergebnisse, oben_weiss


def leeres_brett_modell(brett_pad, belegt, S, P):
    """Synthetisches Bild des LEEREN Bretts (Lab): helle/dunkle Feldfarbe aus leeren
    Feldern geschätzt, Rand = dunkle Farbe. Abweichung davon = Figur."""
    lab = cv2.cvtColor(cv2.GaussianBlur(brett_pad, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    farben = {0: [], 1: []}
    for r in range(8):
        for c in range(8):
            if not belegt[r][c]:
                y, x = r*S + P + 20, c*S + P + 20
                farben[(r + c) % 2].append(np.median(lab[y:y+S-40, x:x+S-40].reshape(-1, 3), 0))
    f0, f1 = (np.median(farben[k], 0) for k in (0, 1))
    # Rand (Koordinatenleiste, Linie, Holz): Profil je Seite getrennt messen (Beleuchtung!)
    #  links/rechts in den figurfreien Reihen 3-6, oben/unten in den Eckbereichen
    E = 8*S + P
    mid = slice(3*S + P, 5*S + P)
    ecke = np.r_[0:P - 5, E + 5:E + P]
    prof = {
        "l": [np.median(lab[mid, P - d], 0) for d in range(1, P + 1)],
        "r": [np.median(lab[mid, E + d - 1], 0) for d in range(1, P + 1)],
        "o": [np.median(lab[P - d, ecke], 0) for d in range(1, P + 1)],
        "u": [np.median(lab[E + d - 1, ecke], 0) for d in range(1, P + 1)],
    }
    bg = np.empty_like(lab)
    H = lab.shape[0]
    yy, xx = np.mgrid[:H, :H]
    abst = {"o": P - yy, "u": yy - (E - 1), "l": P - xx, "r": xx - (E - 1)}
    seite = np.argmax(np.stack(list(abst.values())), 0)
    for i, k in enumerate(abst):
        m = (seite == i) & (abst[k] > 0)
        bg[m] = np.array(prof[k])[np.clip(abst[k][m], 1, P) - 1]
    for r in range(8):
        for c in range(8):
            soll = f0 if (r + c) % 2 == 0 else f1
            # lokale Feldfarbe (Beleuchtung variiert): Randring des Feldes, nur Pixel nahe Sollfarbe
            blk = lab[r*S+P:(r+1)*S+P, c*S+P:(c+1)*S+P]
            ring = np.concatenate([blk[:8].reshape(-1, 3), blk[-8:].reshape(-1, 3),
                                   blk[:, :8].reshape(-1, 3), blk[:, -8:].reshape(-1, 3)])
            anders = f1 if (r + c) % 2 == 0 else f0
            nah = ring[(np.linalg.norm(ring - soll, axis=1) < np.linalg.norm(ring - anders, axis=1))
                       & (np.linalg.norm(ring - soll, axis=1) < 70)]
            bg[r*S+P:(r+1)*S+P, c*S+P:(c+1)*S+P] = np.median(nah, 0) if len(nah) > 50 else soll
    return lab, bg


def figur_ausschneiden(brett_pad, modell, r, c, S, P, name, out):
    lab, bg = modell
    # Ausschnitt: Feld + Rand (Figuren ragen perspektivisch über das Feld hinaus)
    y0, x0 = r*S + P - 70, c*S + P - 35
    sl = (slice(y0, y0 + S + 140), slice(x0, x0 + S + 70))
    roi = brett_pad[sl].copy()
    diff = np.linalg.norm(lab[sl] - bg[sl], axis=2)
    h, w = diff.shape
    cy, cx = 70 + S // 2, 35 + S // 2
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot((yy - cy) / 1.3, xx - cx) / S
    # lokale Textur: leere Felder sind glatt, Figuren (auch schwarz auf schwarz) haben Struktur
    L = lab[sl][..., 0]
    textur = np.sqrt(np.maximum(cv2.blur(L * L, (9, 9)) - cv2.blur(L, (9, 9)) ** 2, 0))
    # Textur nur auf dunklen Feldern nutzen und nicht direkt an Feldgrenzen (Rasterlinien)
    gy, gx = np.mgrid[y0:y0 + h, x0:x0 + w]
    an_linie = (np.minimum((gy - P) % S, S - (gy - P) % S) < 6) | (np.minimum((gx - P) % S, S - (gx - P) % S) < 6)
    dunkel_bg = bg[sl][..., 0] < 90
    tex_fg = (textur > 4) & dunkel_bg & ~an_linie
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[((diff > 9) | tex_fg) & (dist < 0.75)] = cv2.GC_PR_FGD
    mask[((diff < 4) & ((textur < 1.5) | ~dunkel_bg) & (dist > 0.3)) | (dist > 0.85)] = cv2.GC_BGD
    mask[dist < 0.18] = cv2.GC_FGD          # Feldmitte ist immer Figur
    bgm, fgm_ = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(roi, mask, None, bgm, fgm_, 6, cv2.GC_INIT_WITH_MASK)
    fgm = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    fgm = cv2.morphologyEx(fgm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fgm = cv2.morphologyEx(fgm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labs, st, _ = cv2.connectedComponentsWithStats(fgm)
    k = labs[cy, cx] if labs[cy, cx] > 0 else 1 + st[1:, cv2.CC_STAT_AREA].argmax()
    fgm = np.where(labs == k, 255, 0).astype(np.uint8)  # Komponente in der Feldmitte
    cs, _ = cv2.findContours(fgm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fgm, cs, -1, 255, cv2.FILLED)          # Löcher füllen
    ys, xs = np.nonzero(fgm)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = cv2.GaussianBlur(fgm, (3, 3), 0)
    crop = rgba[y1:y2, x1:x2]
    pfad = f"{out}/figuren/{name}.png"
    cv2.imwrite(pfad, crop)
    # Farbe: Anteil cremefarbener Pixel innerhalb der Maske
    pix = roi[fgm > 0]
    lab_px = cv2.cvtColor(pix[None], cv2.COLOR_BGR2LAB)[0]
    creme = float(((lab_px[:, 0] > 170) & (lab_px[:, 2] > 136)).mean())
    farbe = "weiss" if creme > 0.08 else "schwarz"
    return {"bild": pfad, "farbe": farbe, "creme": round(creme, 3),
            "_crop": crop, "_mask": fgm[y1:y2, x1:x2]}


# ---------------------------------------------------------------- 4. Figurentyp
def merkmale(crop, mask):
    g = cv2.cvtColor(crop[..., :3], cv2.COLOR_BGR2GRAY)
    g = np.where(mask > 0, g, 0).astype(np.uint8)
    g = cv2.resize(g, (64, 64))
    h = hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    hgt, wid = mask.shape
    form = [hgt / 100, wid / 100, (mask > 0).mean(), hgt / max(wid, 1)]
    return np.concatenate([h, np.array(form) * 5])


def grundstellung_labels(ergebnisse):
    lab = {}
    for e in ergebnisse:
        linie, rang = "abcdefgh".index(e["feld"][0]), int(e["feld"][1])
        if rang in (2, 7):
            lab[e["feld"]] = "B"
        elif rang in (1, 8):
            lab[e["feld"]] = GRUNDSTELLUNG[rang][linie]
    return lab


def klassifiziere(ergebnisse, modell=None):
    figs = [e for e in ergebnisse if e["belegt"]]
    X = np.array([merkmale(e["_crop"], e["_mask"]) for e in figs])
    if modell is None:  # Training auf Grundstellung
        lab = grundstellung_labels(figs)
        y = np.array([lab[e["feld"]] for e in figs])
        # Leave-one-out zur ehrlichen Einschätzung
        treffer = 0
        for i in range(len(figs)):
            mk = make_pipeline(StandardScaler(), KNeighborsClassifier(1))
            idx = np.arange(len(figs)) != i
            mk.fit(X[idx], y[idx])
            treffer += mk.predict(X[i:i+1])[0] == y[i]
        loo = treffer / len(figs)
        modell = make_pipeline(StandardScaler(), KNeighborsClassifier(1)).fit(X, y)
        print(f"Leave-one-out-Genauigkeit Figurentyp: {loo:.0%} ({treffer}/{len(figs)})")
    pred = modell.predict(X)
    for e, p in zip(figs, pred):
        e["typ"] = p
    return modell


# ---------------------------------------------------------------- 5. Baumdiagramm
def baumdiagramm(ergebnisse, out):
    figs = [e for e in ergebnisse if e["belegt"]]
    L = ['digraph G {', 'rankdir=LR; nodesep=0.12; ranksep=0.9;',
         'node [fontname="Helvetica", fontsize=13, shape=box, style="rounded,filled", fillcolor="#f4f1ea", color="#8a8170"];',
         'edge [color="#8a8170"];',
         f'brett [label="Schachbrett\\n{len(figs)} Figuren, {64-len(figs)} leere Felder", fillcolor="#3b3a36", fontcolor="white", fontsize=15];']
    for farbe in ("weiss", "schwarz"):
        fz = [e for e in figs if e["farbe"] == farbe]
        if not fz:
            continue
        fill, font = ("#ffffff", "#222222") if farbe == "weiss" else ("#222222", "white")
        L.append(f'{farbe} [label="{farbe.capitalize()} ({len(fz)})", fillcolor="{fill}", fontcolor="{font}"];')
        L.append(f'brett -> {farbe};')
        for typ in "KDTLSB":
            tz = sorted([e for e in fz if e.get("typ") == typ], key=lambda e: e["feld"])
            if not tz:
                continue
            tid = f"{farbe}_{typ}"
            L.append(f'{tid} [label="{NAMEN[typ]} ({len(tz)})"];')
            L.append(f'{farbe} -> {tid};')
            for e in tz:
                fid = f'{farbe}_{e["feld"]}'
                img = os.path.abspath(e["bild"])
                L.append(f'{fid} [shape=none, style="", label=<<TABLE BORDER="0" CELLSPACING="0">'
                         f'<TR><TD FIXEDSIZE="TRUE" WIDTH="44" HEIGHT="52"><IMG SRC="{img}" SCALE="TRUE"/></TD>'
                         f'<TD>{e["feld"]}</TD></TR></TABLE>>];')
                L.append(f'{tid} -> {fid};')
    L.append('}')
    dot = f"{out}/5_baumdiagramm.dot"
    open(dot, "w").write("\n".join(L))
    for fmt in ("png", "svg"):
        subprocess.run(["dot", f"-T{fmt}", "-Gdpi=110", dot, "-o", f"{out}/5_baumdiagramm.{fmt}"], check=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bild")
    ap.add_argument("--out", default="ausgabe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img = cv2.imread(a.bild)
    brett, brett_pad, S, P = schneide_brett(img, a.out)
    erg, oben_weiss = analysiere_felder(brett, brett_pad, S, P, a.out)
    klassifiziere(erg)
    baumdiagramm(erg, a.out)
    # Übersicht: Felderraster mit Erkennung
    ov = brett.copy()
    for e in erg:
        x, y = e["spalte"] * S, e["zeile"] * S
        cv2.rectangle(ov, (x, y), (x + S, y + S), (0, 200, 0) if e["belegt"] else (120, 120, 120), 2)
        txt = e["feld"] + (" " + e.get("typ", "") if e["belegt"] else "")
        cv2.putText(ov, txt, (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imwrite(f"{a.out}/2_felder_raster.png", ov)
    json.dump([{k: v for k, v in e.items() if not k.startswith("_")} for e in erg],
              open(f"{a.out}/ergebnis.json", "w"), ensure_ascii=False, indent=1)
    print(f"Weiss steht {'oben' if oben_weiss else 'unten'}; {sum(e['belegt'] for e in erg)} Figuren gefunden.")


if __name__ == "__main__":
    main()
