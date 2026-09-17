"""无浏览器依赖的结果地图渲染器。

输出一个包含数据摘要和 GeoJSON 的 HTML，前端可以直接下载或在自己的地图组件
中读取。它不是 GUI 自动化，也不依赖第三方地图服务。
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from app.core.models import Dataset
from app.gis.errors import GISFailure


class MapRenderer:
    def render(self, dataset: Dataset, output_path: str | Path, *, title: str | None = None) -> Path:
        try:
            frame = gpd.read_file(dataset.path)
            geojson = frame.to_json()
        except Exception as exc:
            raise GISFailure("MAP_RENDER_FAILED", f"地图结果渲染失败：{exc}") from exc
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        label = title or dataset.name
        html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{label}</title>
<style>body{{font:14px system-ui;margin:2rem;color:#16324f}}pre{{background:#f3f7fa;padding:1rem;overflow:auto}}.badge{{background:#d9f2e6;padding:.3rem .6rem;border-radius:1rem}}</style>
</head><body><h1>{label}</h1><p><span class="badge">GeoAgent map artifact</span> 数据集：{dataset.name} · 要素：{len(frame)}</p>
<p>该产物包含可供 MapLibre/Leaflet 载入的 GeoJSON 数据。</p><script>window.GEOJSON = {geojson};</script>
<pre id="preview"></pre><script>document.querySelector('#preview').textContent=JSON.stringify(window.GEOJSON,null,2).slice(0,12000);</script>
</body></html>"""
        target.write_text(html, encoding="utf-8")
        return target

