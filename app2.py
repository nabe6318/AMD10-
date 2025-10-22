# app.py
# 標高補正付き気象マップ（5mメッシュ + 1kmメッシュを別表示）
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

# ============================================================
# 画面設定
# ============================================================
st.set_page_config(page_title="標高補正付き気象マップ（5m + 1km別表示）", layout="wide")

# タイトル部分（少し小さめのフォントに変更）
st.markdown(
    "<h2 style='text-align: center; font-size:22px;'>標高補正付き気象マップ（5mメッシュ + 1kmメッシュ）</h2>",
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
xml_file = st.file_uploader("📂 5m標高メッシュXMLファイル", type="xml")
element_label = st.selectbox("気象要素を選択", list(ELEMENT_OPTIONS.keys()))
element = ELEMENT_OPTIONS[element_label]
date_sel = st.date_input("対象日を選択", value=_date.today())

# ============================================================
# ユーティリティ関数
# ============================================================
def parse_gml_tuplelist_xml(xml_bytes: bytes):
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
    body = np.array([float(l.split(',')[1].rstrip(') \r\n')) for l in datalist])

    def header(tag):
        hit = next(l for l in headers if f"<gml:{tag}>" in l or f"{tag}" in l)
        txt = hit.split(">")[1].split("<")[0].strip()
        return txt.split(" ")

    lats, lons = map(float, header("lowerCorner"))
    late, lone = map(float, header("upperCorner"))
    high_vals = list(map(int, header("high")))
    nola, nolo = [x + 1 for x in high_vals[::-1]]

    dlat = (late - lats) / (nola - 1)
    dlon = (lone - lons) / (nolo - 1)
    lat_grid = np.array([lats + dlat * i for i in range(nola)])
    lon_grid = np.array([lons + dlon * j for j in range(nolo)])

    nli50m = body.reshape((nola, nolo))[::-1, :]
    nli50m[nli50m < -990] = np.nan
    lalodomain = [lats, late, lons, lone]
    return nli50m, lat_grid, lon_grid, lalodomain


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
if st.button("🌏 マップ作成"):
    if not xml_file:
        st.info("XMLファイルを指定してください。")
        st.stop()
    if not date_sel:
        st.info("日付を指定してください。")
        st.stop()

    try:
        # --- XML パース ---
        nli50m, lat_grid, lon_grid, lalodomain = parse_gml_tuplelist_xml(xml_file.getvalue())

        # --- AMDデータ取得 ---
        timedomain = [str(date_sel), str(date_sel)]
        Msh, tim, _, _, nam, uni = amd.GetMetData(element, timedomain, lalodomain, namuni=True)
        Msha, _, _, nama, unia = amd.GetGeoData("altitude", lalodomain, namuni=True)

        st.write(f"気象データ shape: {np.shape(Msh)}")
        st.write(f"標高データ shape: {np.shape(Msha)}")

        # --- 2D化 ---
        Msh2D = to_2d_grid(Msh, "気象データ(1km)")
        Msha2D = to_2d_grid(Msha, "標高データ(1km)")

        # --- DEM補正 ---
        val_msh = safe_scalar(Msh, "気象データ")
        val_msha = safe_scalar(Msha, "標高データ")
        nola, nolo = len(lat_grid), len(lon_grid)
        Msh5m = np.full((nola, nolo), val_msh)
        Msha5m = np.full((nola, nolo), val_msha)
        corrected = Msh5m + (Msha5m - nli50m) * 0.006  # lapse rate

        # --- 1km格子軸作成 ---
        lon_km = lat_km = None
        if Msh2D is not None:
            ny, nx = Msh2D.shape
            lat_km = np.linspace(lat_grid.min(), lat_grid.max(), ny)
            lon_km = np.linspace(lon_grid.min(), lon_grid.max(), nx)

        # =======================================================
        # 図の描画（別表示タブ）
        # =======================================================
        st.subheader("🗺️ マップ表示（5m補正 と 1kmメッシュ 別表示）")
        tabs = st.tabs(["🗺️ 5m DEM補正マップ", "🧭 1kmメッシュ（元データ）"])

        base_cmap = copy.copy(plt.cm.get_cmap("Spectral_r"))
        base_cmap.set_over('w', 1.0)
        base_cmap.set_under('k', 1.0)

        # --- タブ1: 5m DEM補正 ---
        with tabs[0]:
            figtitle = f"{nam} [{uni}] on {tim[0].strftime('%Y-%m-%d')}"
            tate = 6
            lat_span = float(np.max(lat_grid) - np.min(lat_grid))
            lon_span = float(np.max(lon_grid) - np.min(lon_grid))
            yoko = tate * (lon_span / max(1e-9, lat_span)) + 2

            fig = plt.figure(figsize=(yoko, tate))
            ax = plt.gca()
            ax.set_facecolor('0.85')

            levels = np.linspace(np.nanmin(corrected), np.nanmax(corrected), 20)
            cf = ax.contourf(lon_grid, lat_grid, corrected, levels, cmap=base_cmap, extend='both')
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
                tate_km = 6
                yoko_km = tate_km * (lon_span / max(1e-9, lat_span)) + 2

                fig_km = plt.figure(figsize=(yoko_km, tate_km))
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

        # --- 5m補正 ---
        flat_5m = [
            [float(la), float(lo), round(float(corrected[i, j]), 3)]
            for i, la in enumerate(lat_grid)
            for j, lo in enumerate(lon_grid)
            if not np.isnan(corrected[i, j])
        ]
        df_5m = pd.DataFrame(flat_5m, columns=["lat", "lon", f"corrected_{nam} [{uni}]"])
        st.download_button(
            "DEM補正（5m）CSVをダウンロード",
            df_5m.to_csv(index=False).encode("utf-8-sig"),
            file_name="corrected_map_5m.csv",
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
    st.info("XMLファイルと日付を指定してから「🌏 マップ作成」を押してください。")
