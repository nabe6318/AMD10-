# app.py
# 標高補正付き気象マップ（10mメッシュ + 1kmメッシュを別表示：気温のみ）
# O. Watanabe, Shinshu Univ. / AMD_Tools4 を利用

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import AMD_Tools4 as amd
import xml.etree.ElementTree as ET
from io import StringIO
import copy
from datetime import date as _date
import math

# ============================================================
# 画面設定
# ============================================================
st.set_page_config(page_title="標高補正付き気象マップ（10m + 1km別表示）", layout="wide")

st.markdown(
    "<h2 style='text-align: center; font-size:22px;'>標高補正付き気象マップ（10mメッシュ + 1kmメッシュ）</h2>",
    unsafe_allow_html=True
)

# ============================================================
# 気象要素の選択肢（気温のみ）
# ============================================================
ELEMENT_OPTIONS = {
    "日平均気温 (TMP_mea)": "TMP_mea",
    "日最高気温 (TMP_max)": "TMP_max",
    "日最低気温 (TMP_min)": "TMP_min",
}

# ============================================================
# 入力 UI
# ============================================================
xml_file = st.file_uploader("📂 10m標高メッシュXMLファイル（※10mのみ対応、±3 m許容）", type="xml")
element_label = st.selectbox("気象要素を選択", list(ELEMENT_OPTIONS.keys()))
element = ELEMENT_OPTIONS[element_label]
date_sel = st.date_input("対象日を選択", value=_date.today())

# ============================================================
# XML解析関数
# ============================================================
def parse_gml_tuplelist_xml_10m(xml_bytes: bytes, tol_m: float = 3.0):
    xml_str = xml_bytes.decode("utf-8")
    lines = xml_str.splitlines()

    try:
        idx = lines.index('<gml:tupleList>')
    except ValueError:
        idxs = [i for i, l in enumerate(lines) if "<gml:tupleList" in l]
        if not idxs:
            raise ValueError("gml:tupleList タグが見つかりません。")
        idx = idxs[0]
    try:
        idx_end = lines.index('</gml:tupleList>')
    except ValueError:
        idxs = [i for i, l in enumerate(lines) if "</gml:tupleList>" in l]
        if not idxs:
            raise ValueError("</gml:tupleList> タグが見つかりません。")
        idx_end = idxs[0]

    headers = lines[:idx]
    datalist = lines[idx + 1 : idx_end]

    try:
        body = np.array([float(l.split(',')[1].rstrip(') \r\n')) for l in datalist], dtype=float)
    except Exception:
        raise ValueError("標高データの読み取りに失敗しました。")

    def header(tag):
        hit = next((l for l in headers if f"<gml:{tag}>" in l or f"{tag}" in l), None)
        if hit is None:
            raise ValueError(f"ヘッダ {tag} が見つかりません。")
        txt = hit.split(">")[1].split("<")[0].strip()
        return txt.split(" ")

    lats, lons = map(float, header("lowerCorner"))
    late, lone = map(float, header("upperCorner"))
    high_vals = list(map(int, header("high")))

    candidates = []
    for rev in [True, False]:
        hv = high_vals[::-1] if rev else high_vals[:]
        if len(hv) < 2:
            continue
        ny, nx = hv[0] + 1, hv[1] + 1
        if ny * nx != len(body):
            continue
        dlat = (late - lats) / max(ny - 1, 1)
        dlon = (lone - lons) / max(nx - 1, 1)
        mean_lat = (lats + late) / 2.0
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(mean_lat))
        dy_m = abs(dlat) * m_per_deg_lat
        dx_m = abs(dlon) * max(m_per_deg_lon, 1e-9)
        score = abs(dy_m - 10.0) + abs(dx_m - 10.0)
        candidates.append((score, rev, ny, nx, dy_m, dx_m))

    if not candidates:
        raise ValueError("このXMLは10mメッシュとして解析できません。")

    score, rev, ny, nx, dy_m, dx_m = sorted(candidates, key=lambda x: x[0])[0]

    if not ((10.0 - tol_m) <= dy_m <= (10.0 + tol_m) and (10.0 - tol_m) <= dx_m <= (10.0 + tol_m)):
        raise ValueError(f"このXMLは10mメッシュではありません（推定解像度: dy≈{dy_m:.2f} m, dx≈{dx_m:.2f} m）。")

    dlat = (late - lats) / max(ny - 1, 1)
    dlon = (lone - lons) / max(nx - 1, 1)
    lat_grid = np.array([lats + dlat * i for i in range(ny)])
    lon_grid = np.array([lons + dlon * j for j in range(nx)])
    elev = body.reshape((ny, nx))[::-1, :]
    elev[elev < -990] = np.nan
    lalodomain = [lats, late, lons, lone]
    return elev, lat_grid, lon_grid, lalodomain, dy_m, dx_m


def to_2d_grid(arr, name):
    arr = np.array(arr)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3:
        return arr[0, :, :]
    else:
        st.warning(f"{name} の次元が想定外（ndim={arr.ndim}）")
        return None


def safe_scalar(val, name):
    try:
        return float(np.array(val).flatten()[0])
    except Exception:
        return float(np.nanmean(val))

# ============================================================
# 実行部分
# ============================================================
if st.button("🌏 マップ作成"):
    if not xml_file:
        st.info("XMLファイルを指定してください。")
        st.stop()

    try:
        nli10m, lat10m, lon10m, lalodomain, dy_m, dx_m = parse_gml_tuplelist_xml_10m(xml_file.getvalue(), tol_m=3.0)
        st.caption(f"推定メッシュ解像度: dy≈{dy_m:.2f} m, dx≈{dx_m:.2f} m（10m判定OK）")

        timedomain = [str(date_sel), str(date_sel)]
        Msh, tim, _, _, nam, uni = amd.GetMetData(element, timedomain, lalodomain, namuni=True)
        Msha, _, _, nama, unia = amd.GetGeoData("altitude", lalodomain, namuni=True)

        Msh2D = to_2d_grid(Msh, "気象データ(1km)")
        Msha2D = to_2d_grid(Msha, "標高データ(1km)")

        val_msh = safe_scalar(Msh, "気象データ")
        val_msha = safe_scalar(Msha, "標高データ(1km)")
        nola, nolo = len(lat10m), len(lon10m)
        lapse = 0.006  # 0.6 ℃ / 100 m
        corrected = np.full((nola, nolo), val_msh) + (np.full((nola, nolo), val_msha) - nli10m) * lapse

        # --- 1km格子軸作成 ---
        lon_km = lat_km = None
        if Msh2D is not None:
            ny, nx = Msh2D.shape
            lat_km = np.linspace(lat10m.min(), lat10m.max(), ny)
            lon_km = np.linspace(lon10m.min(), lon10m.max(), nx)

        # =======================================================
        # 図の描画（自動スケール）
        # =======================================================
        st.subheader("🗺️ マップ表示（10m補正 と 1kmメッシュ 別表示）")
        tabs = st.tabs(["🗺️ 10m DEM補正マップ", "🧭 1kmメッシュ（元データ）"])

        base_cmap = copy.copy(plt.cm.get_cmap("Spectral_r"))
        base_cmap.set_over('w', 1.0)
        base_cmap.set_under('k', 1.0)

        tate = 6
        lat_span = float(np.max(lat10m) - np.min(lat10m))
        lon_span = float(np.max(lon10m) - np.min(lon10m))
        yoko = tate * (lon_span / max(1e-9, lat_span)) + 2

        # --- タブ1: 10m DEM補正 ---
        with tabs[0]:
            figtitle = f"{nam} [{uni}] on {tim[0].strftime('%Y-%m-%d')} (10m補正)"
            fig = plt.figure(figsize=(yoko, tate))
            ax = plt.gca()
            ax.set_facecolor('0.85')

            vmin = np.nanmin(corrected)
            vmax = np.nanmax(corrected)
            levels = np.linspace(vmin, vmax, 20)

            cf = ax.contourf(lon10m, lat10m, corrected, levels, cmap=base_cmap, extend='both')
            cbar1 = plt.colorbar(cf, ax=ax, fraction=0.025, pad=0.02)
            cbar1.set_label(f"DEM補正後 {nam} [{uni}]")

            ax.set_title(figtitle)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            st.pyplot(fig)

        # --- タブ2: 1kmメッシュ ---
        with tabs[1]:
            if (Msh2D is not None) and (lat_km is not None) and (lon_km is not None):
                figtitle_km = f"1kmメッシュ {nam} [{uni}] on {tim[0].strftime('%Y-%m-%d')}"
                fig_km = plt.figure(figsize=(yoko, tate))
                ax_km = plt.gca()
                ax_km.set_facecolor('0.85')

                vmin2 = np.nanmin(Msh2D)
                vmax2 = np.nanmax(Msh2D)

                pcm = ax_km.pcolormesh(lon_km, lat_km, Msh2D, shading='auto',
                                       cmap=base_cmap, vmin=vmin2, vmax=vmax2)
                cbar2 = plt.colorbar(pcm, ax=ax_km, fraction=0.025, pad=0.02)
                cbar2.set_label(f"1kmメッシュ {nam} [{uni}]")

                ax_km.set_title(figtitle_km)
                ax_km.set_xlabel("Longitude")
                ax_km.set_ylabel("Latitude")
                st.pyplot(fig_km)
            else:
                st.info("この領域では1kmメッシュデータが取得できませんでした。")

        # =======================================================
        # CSVダウンロード
        # =======================================================
        st.subheader("📥 CSVダウンロード")

        flat_10m = [
            [float(la), float(lo), round(float(corrected[i, j]), 3)]
            for i, la in enumerate(lat10m)
            for j, lo in enumerate(lon10m)
            if not np.isnan(corrected[i, j])
        ]
        df_10m = pd.DataFrame(flat_10m, columns=["lat", "lon", f"corrected_{nam} [{uni}]"])
        st.download_button(
            "DEM補正（10m）CSVをダウンロード",
            df_10m.to_csv(index=False).encode("utf-8-sig"),
            file_name="corrected_map_10m.csv",
            mime="text/csv"
        )

        if Msh2D is not None and lat_km is not None and lon_km is not None:
            rows_km = [
                [float(la), float(lo), round(float(Msh2D[ii, jj]), 3)]
                for ii, la in enumerate(lat_km)
                for jj, lo in enumerate(lon_km)
                if not np.isnan(Msh2D[ii, jj])
            ]
            df_km = pd.DataFrame(rows_km, columns=["lat", "lon", f"met1km_{nam} [{uni}]"])
            st.download_button(
                "1kmメッシュCSVをダウンロード",
                df_km.to_csv(index=False).encode("utf-8-sig"),
                file_name="met1km_map.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ 処理中にエラーが発生しました: {e}")

else:
    st.info("10m標高XMLと日付を指定してから「🌏 マップ作成」を押してください。")
