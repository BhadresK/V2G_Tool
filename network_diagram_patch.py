from __future__ import annotations
EDGES = [
    ("TF", "B32"),
    ("B32", "B15"), ("B32", "B5"), ("B32", "B36"),
    ("B15", "B31"), ("B5", "B26"), ("B36", "B37"),
    ("B31", "B20"), ("B26", "B18"), ("B37", "B38"),
    ("B20", "B9"),  ("B18", "B35"), ("B38", "B39"),
    ("B9", "B2"),
    ("B35", "B14"), ("B35", "B21"),
    ("B39", "B40"), ("B39", "B42"),
    ("B14", "B12"), ("B21", "B16"),
    ("B40", "B41"), ("B42", "B43"),
    ("B12", "B1"),  ("B16", "B24"),
    ("B43", "B44"),
    ("B1", "B29"),  ("B24", "B30"),
    ("B29", "B7"),  ("B30", "B23"),
    ("B7", "B33"),  ("B23", "B22"),
    ("B33", "B28"),
    ("B28", "B8"), ("B28", "B17"),
    ("B8", "B11"), ("B17", "B4"),
    ("B11", "B19"), ("B4", "B6"),
    ("B19", "B34"), ("B6", "B13"),
    ("B34", "B3"), ("B13", "B25"),
    ("B3", "B10"),
]
NODES = {
    "MV":  ("MV Grid", "20 kV", "source"),
    "TF":  ("Transformer", "1.00 MVA", "source"),
    "B32": ("Bus 32", "Transformer LV Bus", "source"),
    "B1":  ("Bus 1", "H0-B", "res"), "B2": ("Bus 2", "G1-A", "com"),
    "B3":  ("Bus 3", "H0-A", "res"), "B4": ("Bus 4", "H0-B", "res"),
    "B5":  ("Bus 5", "H0-L/B", "res"), "B6": ("Bus 6", "H0-B", "res"),
    "B7":  ("Bus 7", "H0-G", "res"), "B8": ("Bus 8", "H0-A", "res"),
    "B9":  ("Bus 9", "H0-B", "res"), "B10": ("Bus 10", "H0-B", "res"),
    "B11": ("Bus 11", "H0-B", "res"), "B12": ("Bus 12", "H0-L", "res"),
    "B13": ("Bus 13", "H0-A", "res"), "B14": ("Bus 14", "H0-A", "res"),
    "B15": ("Bus 15", "H0-L/G", "res"), "B16": ("Bus 16", "H0-A", "res"),
    "B17": ("Bus 17", "H0-A", "res"), "B18": ("Bus 18", "H0-B", "res"),
    "B19": ("Bus 19", "H0-L", "res"), "B20": ("Bus 20", "H0-G", "res"),
    "B21": ("Bus 21", "H0-B", "res"), "B22": ("Bus 22", "H0-B", "res"),
    "B23": ("Bus 23", "H0-L", "res"), "B24": ("Bus 24", "H0-L", "res"),
    "B25": ("Bus 25", "H0-G", "res"), "B26": ("Bus 26", "H0-G", "res"),
    "B28": ("Bus 28", "junction", "junc"),
    "B29": ("Bus 29", "H0-G", "res"), "B30": ("Bus 30", "H0-G", "res"),
    "B31": ("Bus 31", "H0-L", "res"),
    "B33": ("Bus 33", "H0-L", "res"), "B34": ("Bus 34", "H0-G", "res"),
    "B35": ("Bus 35", "junction", "junc"),
    "B36": ("Bus 36", "G4-A", "com"), "B37": ("Bus 37", "G2-A", "com"),
    "B38": ("Bus 38", "G4-A", "com"),
    "B39": ("Bus 39", "junction", "junc"),
    "B40": ("Bus 40", "G6-A", "com"), "B41": ("Bus 41", "G2-A", "com"),
    "B42": ("Bus 42", "G4-B", "com"), "B43": ("Bus 43", "G1-A", "com"),
    "B44": ("Bus 44", "Trailer", "trailer"),
}

COLORS = {
    "source":  ("#f0f0ee", "#888888"),
    "res":     ("#dcf5ed", "#0F6E56"),
    "com":     ("#ddeeff", "#185FA5"),
    "junc":    ("#f0f0ee", "#888888"),
    "trailer": ("#fff3d6", "#BA7517"),
}

BOX_W, BOX_H = 116, 38
COL_GAP, ROW_GAP = 26, 22


def _compute_layout():
    children = {}
    for p, c in EDGES:
        children.setdefault(p, []).append(c)

    depth = {"MV": 0, "TF": 1}
    order = ["TF"]
    i = 0
    while i < len(order):
        node = order[i]
        i += 1
        for ch in children.get(node, []):
            depth[ch] = depth[node] + 1
            order.append(ch)

    leaf_x = {}
    counter = [0]

    def assign_x(node):
        kids = children.get(node, [])
        if not kids:
            leaf_x[node] = counter[0]
            counter[0] += 1
            return leaf_x[node]
        xs = [assign_x(k) for k in kids]
        leaf_x[node] = sum(xs) / len(xs)
        return leaf_x[node]

    assign_x("TF")
    leaf_x["MV"] = leaf_x["TF"]

    pos = {}
    for node, d in depth.items():
        x = leaf_x[node] * (BOX_W + COL_GAP) + BOX_W / 2 + 20
        y = d * (BOX_H + ROW_GAP) + 20
        pos[node] = (x, y)
    return pos


def _node_svg(node_id, x, y, label1, label2):
    cat = NODES[node_id][2]
    fill, stroke = COLORS[cat]
    sw = "2" if cat == "trailer" else "0.8"
    dash = ' stroke-dasharray="2 2"' if cat == "junc" else ""
    return f"""
<rect x="{x - BOX_W/2:.1f}" y="{y:.1f}" width="{BOX_W}" height="{BOX_H}" rx="5"
      fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>
<text x="{x:.1f}" y="{y + BOX_H*0.38:.1f}" text-anchor="middle"
      font-size="11" font-weight="600" fill="#222" font-family="system-ui,sans-serif">{label1}</text>
<text x="{x:.1f}" y="{y + BOX_H*0.74:.1f}" text-anchor="middle"
      font-size="9.5" fill="#555" font-family="system-ui,sans-serif">{label2}</text>"""


def _edge_svg(x1, y1, x2, y2):
    midy = (y1 + y2) / 2
    return (f'<path d="M{x1:.1f},{y1:.1f} L{x1:.1f},{midy:.1f} '
            f'L{x2:.1f},{midy:.1f} L{x2:.1f},{y2:.1f}" '
            f'fill="none" stroke="#888" stroke-width="1.3"/>')


def _build_full_network_svg(trafo_mva: float = 1.0):
    NODES["TF"] = ("Transformer", f"{trafo_mva:.2f} MVA", "source")
    pos = _compute_layout()

    max_x = max(x for x, y in pos.values()) + BOX_W / 2 + 20
    max_y = max(y for x, y in pos.values()) + BOX_H + 20

    parts = []
    for p, c in EDGES:
        x1, y1 = pos[p]
        x2, y2 = pos[c]
        parts.append(_edge_svg(x1, y1 + BOX_H, x2, y2))

    for node_id, (x, y) in pos.items():
        l1, l2, _cat = NODES[node_id]
        parts.append(_node_svg(node_id, x, y, l1, l2))

    return "\n".join(parts), max_x, max_y


def render_static_network_diagram(trafo_mva: float) -> None:
    import streamlit.components.v1 as components

    body, max_x, max_y = _build_full_network_svg(trafo_mva)

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; padding:0; background:transparent; font-family:system-ui,sans-serif; }}
  #legend {{
      display:flex; flex-wrap:wrap; gap:14px; align-items:center;
      font-size:11px; color:#444; padding:4px 8px 8px 8px;
  }}
  .swatch {{ width:12px; height:10px; border-radius:2px; display:inline-block; margin-right:4px; vertical-align:-1px; }}
  #hint {{ font-size:10px; color:#999; font-style:italic; padding:0 8px 6px 8px; }}
  #canvas {{
      width:100%; height:560px; border:1px solid #e2e2e2; border-radius:6px;
      overflow:hidden; position:relative; background:#fff; cursor:grab;
  }}
  #zoomWrap {{ transform-origin: 0 0; }}
  #controls {{
      position:absolute; top:8px; right:8px; display:flex; flex-direction:column; gap:4px; z-index:10;
  }}
  #controls button {{
      width:26px; height:26px; border:1px solid #ccc; border-radius:4px; background:#fff;
      cursor:pointer; font-size:14px; color:#444; line-height:1;
  }}
  #controls button:hover {{ background:#f0f2f6; }}
</style>
</head>
<body>

<div id="legend">
  <span><span class="swatch" style="background:#dcf5ed;border:1px solid #0F6E56;"></span>Residential (H0)</span>
  <span><span class="swatch" style="background:#ddeeff;border:1px solid #185FA5;"></span>Commercial (G-profile)</span>
  <span><span class="swatch" style="background:#f0f0ee;border:1px dashed #888;"></span>Junction (no load)</span>
  <span><span class="swatch" style="background:#fff3d6;border:2px solid #BA7517;"></span>Trailer (worst-case)</span>
</div>
<div id="hint">Scroll to zoom &middot; drag to pan &middot; double-click to reset</div>

<div id="canvas">
  <div id="controls">
    <button id="zoomInBtn" title="Zoom in">+</button>
    <button id="zoomOutBtn" title="Zoom out">&minus;</button>
    <button id="resetBtn" title="Reset view">&#8634;</button>
  </div>
  <div id="zoomWrap">
    <svg id="network-svg" viewBox="0 0 {max_x:.0f} {max_y:.0f}" width="{max_x:.0f}" height="{max_y:.0f}"
         xmlns="http://www.w3.org/2000/svg">
      {body}
    </svg>
  </div>
</div>

<script>
(function() {{
  var canvas = document.getElementById('canvas');
  var wrap = document.getElementById('zoomWrap');
  var svgW = {max_x:.0f}, svgH = {max_y:.0f};
  var scale = 1, tx = 0, ty = 0;
  var isPanning = false, startX, startY;

  function applyTransform() {{
    wrap.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
  }}

  function fitToContainer() {{
    var cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (!cw || !ch) return;
    var s = Math.min(cw / svgW, ch / svgH) * 0.95;
    scale = s;
    tx = (cw - svgW * s) / 2;
    ty = (ch - svgH * s) / 2;
    applyTransform();
  }}

  function zoomBy(factor, cx, cy) {{
    var newScale = Math.min(Math.max(scale * factor, 0.2), 8);
    tx = cx - (cx - tx) * (newScale / scale);
    ty = cy - (cy - ty) * (newScale / scale);
    scale = newScale;
    applyTransform();
  }}

  canvas.addEventListener('wheel', function(e) {{
    e.preventDefault();
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left, my = e.clientY - rect.top;
    var factor = e.deltaY < 0 ? 1.12 : 0.89;
    zoomBy(factor, mx, my);
  }}, {{ passive: false }});

  canvas.addEventListener('mousedown', function(e) {{
    isPanning = true;
    startX = e.clientX - tx;
    startY = e.clientY - ty;
    canvas.style.cursor = 'grabbing';
  }});
  window.addEventListener('mousemove', function(e) {{
    if (!isPanning) return;
    tx = e.clientX - startX;
    ty = e.clientY - startY;
    applyTransform();
  }});
  window.addEventListener('mouseup', function() {{
    isPanning = false;
    canvas.style.cursor = 'grab';
  }});
  canvas.addEventListener('dblclick', function() {{
    fitToContainer();
  }});

  document.getElementById('zoomInBtn').addEventListener('click', function() {{
    zoomBy(1.25, canvas.clientWidth / 2, canvas.clientHeight / 2);
  }});
  document.getElementById('zoomOutBtn').addEventListener('click', function() {{
    zoomBy(0.8, canvas.clientWidth / 2, canvas.clientHeight / 2);
  }});
  document.getElementById('resetBtn').addEventListener('click', fitToContainer);

  window.addEventListener('load', fitToContainer);
  setTimeout(fitToContainer, 60);
  setTimeout(fitToContainer, 250);
}})();
</script>

</body>
</html>"""

    components.html(html, height=620, scrolling=False)