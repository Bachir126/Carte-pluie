"""
Carte de pluie du Sénégal MISE À JOUR AUTOMATIQUEMENT (CHIRPS + Cartopy)
Chaque exécution prend la date du jour : lancer 1 fois par semaine.

Installation (Colab / Jupyter) :
    pip install cartopy rasterio geopandas requests matplotlib

Modifier uniquement la section PARAMETRES.
"""
import gzip
import shutil
import datetime as dt
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- PARAMETRES -----------------------------
MODE = "saison"      # "saison" = cumul depuis le 1er juin | "semaine" = 7 derniers jours
AUTEUR = "Ton Nom"
LATENCE_JOURS = 2    # CHIRPS publie les données avec 1-2 jours de retard

DATE_FIN = dt.date.today() - dt.timedelta(days=LATENCE_JOURS)
ANNEE = DATE_FIN.year
DATE_DEBUT = dt.date(ANNEE, 6, 1) if MODE == "saison" else DATE_FIN - dt.timedelta(days=6)
if DATE_FIN < dt.date(ANNEE, 6, 1):
    raise SystemExit("Hors saison des pluies : rien à faire avant le 1er juin.")
PERIODE_SAISON = f"01/06/{ANNEE} au 31/10/{ANNEE}"
VMAX = 120                                    # max de l'échelle (mm)
SORTIE_DIR = Path("cartes")
DOSSIER = Path("chirps_data")
BBOX = (-17.8, 12.1, -11.3, 16.8)             # lon_min, lat_min, lon_max, lat_max
# -----------------------------------------------------------------------

URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/"
       "{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.tif.gz")


def telecharger(jour: dt.date) -> Path:
    """Télécharge et décompresse un fichier CHIRPS journalier (avec cache)."""
    DOSSIER.mkdir(exist_ok=True)
    tif = DOSSIER / f"chirps_{jour:%Y%m%d}.tif"
    if tif.exists():
        return tif
    r = requests.get(URL.format(y=jour.year, m=jour.month, d=jour.day), timeout=120)
    r.raise_for_status()
    gz = tif.with_suffix(".tif.gz")
    gz.write_bytes(r.content)
    with gzip.open(gz, "rb") as fin, open(tif, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    gz.unlink()
    return tif


def cumul(debut: dt.date, fin: dt.date):
    """Somme des pluies journalières sur la période. Retourne (array, transform)."""
    total, transform = None, None
    jour = debut
    while jour <= fin:
        try:
            tif = telecharger(jour)
        except requests.HTTPError:
            print(f"Jour non disponible, ignoré : {jour}")
            jour += dt.timedelta(days=1)
            continue
        with rasterio.open(tif) as src:
            win = from_bounds(*BBOX, src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win).astype("float32")
            arr[arr < 0] = np.nan          # nodata (-9999)
            transform = src.window_transform(win)
        total = np.nan_to_num(arr) if total is None else total + np.nan_to_num(arr)
        jour += dt.timedelta(days=1)
    if total is None:
        raise SystemExit("Aucune donnée CHIRPS disponible pour cette période.")
    return total, transform


# ------------------------- Données géographiques -----------------------
shp = shpreader.natural_earth(resolution="10m", category="cultural",
                              name="admin_1_states_provinces")
regions = gpd.read_file(shp)
regions = regions[regions["admin"] == "Senegal"].to_crs("EPSG:4326")

# ------------------------------- Calcul --------------------------------
pluie, tr = cumul(DATE_DEBUT, DATE_FIN)

# Masquer tout ce qui est hors du Sénégal
masque = geometry_mask(regions.geometry, out_shape=pluie.shape,
                       transform=tr, invert=True)
pluie = np.ma.masked_where(~masque, pluie)

h, w = pluie.shape
extent = [tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f]  # gauche, droite, bas, haut

# -------------------------------- Carte --------------------------------
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([BBOX[0], BBOX[2], BBOX[1], BBOX[3]], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.OCEAN, facecolor="#7fa6dc", zorder=0)
ax.add_feature(cfeature.LAND, facecolor="#f0e8d0", zorder=0)

im = ax.imshow(pluie, extent=extent, origin="upper", transform=ccrs.PlateCarree(),
               cmap="Blues", vmin=0, vmax=VMAX, interpolation="nearest", zorder=1)

# Limites des régions + contour du pays
ax.add_geometries(regions.geometry, ccrs.PlateCarree(), facecolor="none",
                  edgecolor="white", linewidth=0.8, zorder=2)
ax.add_geometries([regions.union_all() if hasattr(regions, "union_all")
                   else regions.unary_union],
                  ccrs.PlateCarree(), facecolor="none",
                  edgecolor="black", linewidth=1.8, zorder=3)

# Noms des régions
for _, row in regions.iterrows():
    p = row.geometry.representative_point()
    ax.text(p.x, p.y, row["name"], fontsize=8, fontweight="bold",
            ha="center", va="center", transform=ccrs.PlateCarree(), zorder=4)

# Grille avec coordonnées
gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.6)
gl.top_labels = False
gl.right_labels = False

# Barre de couleur avec flèche en haut
cbar = fig.colorbar(im, ax=ax, extend="max", shrink=0.9, pad=0.04)
cbar.set_label("Pluie (mm / période)")

# Titre + encadré
ax.set_title(f"Cumul de pluie du {DATE_DEBUT:%d/%m/%Y} au {DATE_FIN:%d/%m/%Y}")
ax.text(0.01, 0.01, f"Réalisation : {AUTEUR}\nPériode du {PERIODE_SAISON}",
        transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), zorder=5)

SORTIE_DIR.mkdir(exist_ok=True)
for nom in (f"pluie_{DATE_FIN:%Y%m%d}.png", "pluie_derniere.png"):
    plt.savefig(SORTIE_DIR / nom, dpi=200, bbox_inches="tight")
plt.close()
print(f"Carte enregistrée : {SORTIE_DIR}/pluie_{DATE_FIN:%Y%m%d}.png")
