# app.py
# 標高補正付き気象マップ（10mメッシュ + 1kmメッシュを別表示）
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

# タイトル（10m専用に明示）
st.markdown(
    "<h2 style='text-align: center; font-size:22px;'>標高補正付き気象マップ（10mメッシュ + 1kmメッシュ）</h2>",
    unsafe_allow_html=True
)

# ============================================================
# 気象要素の選択肢
# ============================================================
ELEMENT_OPTIONS = {
    "日平均気温 (TMP_mea)": "TMP_mea",
    "日最高気温 (TMP_max)": "TMP_max",
    "日最低気温 (TMP_min)": "TMP_min",
    "降水量 (APCP)": "APCP",
    "降水量高精度 (APCPRA)": "APCPRA",
    "降水の有無 (OPR)": "OPR",
    "日照時間 (SSD)": "SSD",
    "全天日射量 (GSR)": "GSR",
    "下向き長波放射量 (DLR)": "DLR",
    "相対湿度 (RH)": "RH",
    "風速 (WIND)": "WIND",
    "積雪深 (SD)": "SD",
    "積雪水量 (SWE)": "SWE",
    "降雪水量 (SFW)": "SFW",
    "予報気温の確からしさ (PTMP)": "PTMP"
}

# ============================================================
# 入力 UI
# ============================================================
xml_file = st.file_uploader("📂 10m標高メッシュXMLファイル（※10mのみ対応）", type="xml")
element_label = st.selectbox("気象要素を選択", list(ELEMENT_OPTIONS.keys()))
element = ELEMENT_OPTIONS[element_label]
date_sel = st.date_input("対象日を選択", value=_date.today())

# ============================================================
# ユーティリティ関数
# ============================================================
def parse_gml_tuplelist_xml_10m(xml_bytes: bytes, tol_m: float = 2.0):
    """
    基盤地図情報 標高（GML/XML）の <gml:tupleList> をパース。
    格子解像度が約10m（±tol_m）かを緯度・経度方向でチェックし、
    10mメッシュでない場合は ValueError を投げる。
    戻り値:
      nli10m (ny, nx), lat_grid (ny,), lon_grid (nx,), lalodomain [lat_min, lat_max, lon_min, lon_max]
    """
    xml_str = xml_bytes.decode("utf-8")
    lines = xml_str.splitlines()

    try:
        idx = lines.index('<gml:tupleList>')
    except ValueError:
        idx = [i for i, l in enumerate(lines) if "<gml:tupleList" in l]
        if not idx:
            raise ValueError("gml:tupleList タグが見つかりません。")
        idx = idx[0]

    headers = lines[:idx]
    try:
        idx_end = lines.index('</gml:tupleList>')
    except ValueError:
        idx_end = [i for i, l in enumerate(lines) if "</gml:tupleList>" in l]
        if not idx_end:
            raise ValueError("</gml:tupleList> タグが見つかりません。")
        idx_end = idx_end[0]

    datalist = lines[idx + 1 : idx_end]
    # tuple は "(x,y)" 形式が多い。2番目（標高値）を抽出。
    try:
        body = np.array([float(l.split(',')[1].rstrip(') \r\n')) for l in datalist])
    except Exception:
        raise ValueError("標高データの読み取りに失敗しました（tupleListの構造が想定と異なります）。")

    def header(tag):
        # <gml:lowerCorner> / <gml:upperCorner> / <gml:high> 等から中身を抜く
        hit = next((l for l in headers if f"<gml:{tag}>" in l or f"{tag}" in l), None)
        if hit is None:
            raise ValueError(f"ヘッダ {tag} が見つかりません。")
        txt = hit.split(">")[1].split("<")[0].strip()
        return txt.split(" ")

    lats, lons = map(float, header("lowerCorner"))
    late, lone = map(float, header("upperCorner"))
    high_vals = list(map(int, header("high")))
    # gml:high は(最大インデックス)なので +1 が格子数
    nola, nolo = [x + 1 for x in high_vals[::-1]]  # [lat_count, lon_count] に合わせる

    # 格子間隔（度）
    dlat = (late - lats) / max(nola - 1, 1)
    dlon = (lone - lons) / max(nolo - 1, 1)

    # 近似メートル換算
    mean_lat = (lats + late) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(mean_lat))

    dy_m = abs(dlat) * m_per_deg_lat
    dx_m = abs(dlon) * max(m_per_deg_lon, 1e-9)

    # 10mチェック（±tol_m で許容）
    def is_10m(v): 
        return (10.0 - tol_m) <= v <= (10.0 + tol_m)

    if not (is_10m(dy_m) and is_10m(dx_m)):
        raise ValueError(
            f"このXMLは10mメッシュではありません（推定解像度: dy≈{dy_m:.2f} m, dx≈{dx_m:.2f} m）。"
        )

    lat_grid = np.array([lats + dlat * i for i in range(nola)])
    lon_grid = np.array([lons + dlon * j for j in range(nolo)])

    nli10m = body.reshape((nola, nolo))[::-1, :]  # 北が上になるよう上下反転
    nli10m[nli10m < -990] = np.nan  # 欠損値処理

    lalodomain = [lats, late, lons, lone]
    return nli10m, lat_grid, lon_grid, lalodomain, dy_m, dx_m


def to_2d_grid(arr, name):
    arr = np.array(arr)
    if arr.ndim == 0:
        st.info(f"{name} がスカラー値のため、領域平均として扱います。")
        return None
    elif arr.ndim == 1:
        n = arr.size
        ny = int(np.sqrt(n))
        nx = n // ny if ny > 0 else 0
        if ny * nx == n and ny > 1 and nx > 1:
            return arr.reshape(ny, nx)
        else:
            st.warning(f"{name} が1Dで整形できず、平均表示になります。shape={arr.shape}")
            return None
    elif arr.ndim == 2:
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
        st.warning(f"{name} がスカラーでなかったため平均値で補間します。shape={np.shape(val)}")
        return float(np.nanmean(val))

# ============================================================
# 実行部分
# ============================================================
if st.button("🌏 マップ作成（10m標高のみ）"):
    if not xml_file:
        st.info("XMLファイルを指定してください。")
        st.stop()
    if not date_sel:
        st.info("日付を指定してください。")
        st.stop()

    try:
        # --- XML パース（10mチェック付） ---
        nli10m, lat10m, lon10m, lalodomain, dy_m, dx_m = parse_gml_tuplelist_xml_10m(xml_file.getvalue())
        st.caption(f"推定メッシュ解像度: dy≈{dy_m:.2f} m, dx≈{dx_m:.2f} m（10m判定OK）")

        # --- AMDデータ取得 ---
        timedomain = [str(date_sel), str(date_sel)]
        Msh, tim, _, _, nam, uni = amd.GetMetData(element, timedomain, lalodomain, namuni=True)
        Msha, _, _, nama, unia = amd.GetGeoData("altitude", lalodomain, namuni=True)

        st.write(f"気象データ shape: {np.shape(Msh)}")
        st.write(f"標高データ(1km) shape: {np.shape(Msha)}")

        # --- 2D化 ---
        Msh2D = to_2d_grid(Msh, "気象データ(1km)")
        Msha2D = to_2d_grid(Msha, "標高データ(1km)")

        # --- DEM補正: 1kmスカラーを10m格子へ展開して lapse rate で補正 ---
        val_msh = safe_scalar(Msh, "気象データ")
        val_msha = safe_scalar(Msha, "標高データ(1km)")
        nola, nolo = len(lat10m), len(lon10m)
        Msh10m = np.full((nola, nolo), val_msh)
        Msha10m = np.full((nola, nolo), val_msha)
        lapse = 0.006  # 0.6 ℃ / 100 m
        corrected = Msh10m + (Msha10m - nli10m) * lapse

        # --- 1km格子軸作成 ---
        lon_km = lat_km = None
        if Msh2D is not None:
            ny, nx = Msh2D.shape
            lat_km = np.linspace(lat10m.min(), lat10m.max(), ny)
            lon_km = np.linspace(lon10m.min(), lon10m.max(), nx)

        # =======================================================
        # 図の描画（別表示タブ）
        # =======================================================
        st.subheader("🗺️ マップ表示（10m補正 と 1kmメッシュ 別表示）")
        tabs = st.tabs(["🗺️ 10m DEM補正マップ", "🧭 1kmメッシュ（元データ）"])

        base_cmap = copy.copy(plt.cm.get_cmap("Spectral_r"))
        base_cmap.set_over('w', 1.0)
        base_cmap.set_under('k', 1.0)

        # 共通アスペクト計算
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

                pcm = ax_km.pcolormesh(lon_km, lat_km, Msh2D, shading='auto', cmap=base_cmap)
                cbar2 = plt.colorbar(pcm, ax=ax_km, fraction=0.025, pad=0.02)
                cbar2.set_label(f"1kmメッシュ {nam} [{uni}]")

                ax_km.set_title(figtitle_km)
                ax_km.set_xlabel("Longitude")
                ax_km.set_ylabel("Latitude")
                st.pyplot(fig_km)
            else:
                st.info("この領域では1kmメッシュデータが取得できませんでした。")

        # =======================================================
        # CSV出力
        # =======================================================
        st.subheader("📥 CSVダウンロード")

        # --- 10m補正 ---
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

        # --- 1kmメッシュ ---
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
        else:
            st.caption("※1kmメッシュがスカラー等で得られない場合はCSVは表示されません。")

    except Exception as e:
        st.error(f"❌ 処理中にエラーが発生しました: {e}")

else:
    st.info("10m標高XMLと日付を指定してから「🌏 マップ作成（10m標高のみ）」を押してください。")
