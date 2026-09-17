"""第一版内置 GIS 知识卡片。"""

KNOWLEDGE = {
    "crs": "距离、面积和缓冲区应使用具有线性单位的投影 CRS；EPSG:4326 的单位是度，不能直接解释为米。",
    "buffer": "缓冲区距离的数值单位来自输入 CRS。GeoAgent 在发现 geographic CRS 时会先选择覆盖范围所在 UTM 分区。",
    "nodata": "栅格统计必须区分 NoData 和有效像元，不能把 NoData 当成真实高程或分类值。",
    "lineage": "派生空间数据应记录输入数据集、操作、参数、运行和工具调用，便于复现与审计。",
}

