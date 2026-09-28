"""Draw a compact pitch diagram with editable SVG text and vector primitives."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 14, 4.2
INK = "#24333D"
MUTED = "#75818B"
LINE = "#DAE1E4"
BASE = "#A47A73"
ACCENT = "#286F73"
TINT = "#F0F7F6"
VISUAL = "#9D681E"
VISUAL_TINT = "#FCF3DF"


def render(output, dpi=280):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "svg.fonttype": "none",
        "svg.hashsalt": "compact-geometry-pitch", "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0), aspect="equal")
    ax.axis("off")
    texts, counters = [], {}

    def identify(artist, key):
        counters[key] = counters.get(key, 0) + 1
        artist.set_gid(f"{key}-{counters[key]}")
        return artist

    def text(x, y, label, size=11, color=INK, weight="normal", ha="left", **kw):
        artist = ax.text(x, y, label, fontsize=size, color=color,
                         fontweight=weight, ha=ha, va="center", **kw)
        identify(artist, "text")
        texts.append(artist)
        return artist

    def line(points, color=LINE, lw=1.2, dashed=False, key="line", zorder=3):
        artist, = ax.plot(*zip(*points), color=color, lw=lw,
                          linestyle=(0, (3, 3)) if dashed else "-",
                          solid_capstyle="round", zorder=zorder)
        return identify(artist, key)

    def arrow(a, b, color=INK, lw=1.25, dashed=False):
        artist = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 linestyle=(0, (3, 3)) if dashed else "-", zorder=2)
        ax.add_patch(artist)
        return identify(artist, "arrow")

    def box(x, y, w, h, color=LINE, fill="white", key="node"):
        artist = FancyBboxPatch((x,y), w,h,
                               boxstyle="round,pad=0,rounding_size=0.07",
                               edgecolor=color, facecolor=fill, lw=1.0, zorder=2)
        ax.add_patch(artist)
        return identify(artist,key)

    def circle(x,y,r,color=INK,fill="white",lw=1.2):
        artist = Circle((x,y),r,edgecolor=color,facecolor=fill,lw=lw,zorder=4)
        ax.add_patch(artist)
        return identify(artist,"point")

    def camera(x,y,color):
        box(x,y+.065,.34,.235,color,key="camera")
        line([(x+.055,y+.065),(x+.095,y),(x+.22,y),(x+.265,y+.065)],color,1.2)
        circle(x+.175,y+.18,.064,color,lw=1.1)

    def robot(x,y,color):
        line([(x,y+.40),(x+.35,y+.40)],color,1.4)
        line([(x+.105,y+.39),(x+.105,y+.24),(x+.235,y+.03),(x+.405,y+.165)],color,1.8)
        for px,py in [(x+.105,y+.24),(x+.235,y+.03),(x+.405,y+.165)]:
            circle(px,py,.032,color,lw=1.1)
        line([(x+.395,y+.195),(x+.35,y+.275),(x+.40,y+.32)],color,1.1)
        line([(x+.43,y+.19),(x+.49,y+.255),(x+.47,y+.315)],color,1.1)

    def cube(x,y,s=.80):
        points = [(x,y+.21*s),(x+.45*s,y),(x+.90*s,y+.21*s),
                  (x+.45*s,y+.43*s),(x,y+.78*s),(x+.45*s,y+s),
                  (x+.90*s,y+.78*s)]
        for indices,fill in [((0,1,2,3),"#DCEBE9"),((0,3,5,4),"#FAFCFC"),
                             ((3,2,6,5),"#E9F2F1")]:
            artist = Polygon([points[i] for i in indices],edgecolor=ACCENT,
                             facecolor=fill,lw=1.0,zorder=3)
            ax.add_patch(artist)
            identify(artist,"geometry-face")
        line([points[5],points[6]],ACCENT,2.5,zorder=4,key="highlighted-edge")
        circle(x+.45*s,y+.55*s,.028,BASE,BASE,lw=.8)

    # Aligned lanes isolate the change from repeated images to geometric text.
    text(.45,.38,"Repeated visual reasoning",size=13.2,color=BASE,weight="bold")
    text(.45,2.20,"With geometric abstraction",size=13.2,color=ACCENT,weight="bold")
    for y,h,color in [(.78,.70,BASE),(2.69,.74,ACCENT)]:
        box(.55,y,1.48,h)
        camera(.74,y+(h-.30)/2,color)
        text(1.24,y+h/2,"Visual",size=11.0,weight="bold")
        box(5.97,y,1.68,h)
        text(6.81,y+h/2,"Multimodal LLM" if y<1 else "LLM reasoning",
             size=10.4 if y<1 else 10.8,weight="bold",ha="center")
        box(8.10,y,1.53,h)
        robot(8.26,y+(h-.40)/2,INK)
        text(8.91,y+h/2,"Act",size=11.2,weight="bold")
        arrow((7.70,y+h/2),(8.05,y+h/2),color)

    arrow((2.08,1.13),(5.92,1.13),BASE)
    text(4.00,.94,"Images",size=10.0,color=BASE,ha="center")
    line([(8.86,1.48),(8.86,1.79),(1.29,1.79)],BASE,1.2,key="visual-refresh")
    arrow((1.29,1.79),(1.29,1.53),BASE)
    text(5.13,1.79,"new observation",size=9.2,color=MUTED,ha="center",
         bbox=dict(facecolor="white",edgecolor="none",pad=3.2))

    box(3.08,2.69,2.10,.74,color=ACCENT,fill=TINT,key="geometric-abstraction")
    text(4.13,2.94,"Geometry",size=11.4,color=ACCENT,weight="bold",ha="center")
    text(4.13,3.23,"text + symbols",size=9.6,color=ACCENT,ha="center")
    arrow((2.08,3.06),(3.03,3.06),ACCENT)
    arrow((5.23,3.06),(5.92,3.06),ACCENT)
    line([(8.86,3.43),(8.86,3.90),(1.29,3.90)],ACCENT,1.2,key="geometry-refresh")
    arrow((1.29,3.90),(1.29,3.48),ACCENT)
    text(5.13,3.90,"refresh geometry",size=9.2,color=ACCENT,ha="center",
         bbox=dict(facecolor="white",edgecolor="none",pad=3.2))

    # Images can bypass abstraction when the agent needs visual evidence.
    line([(1.29,2.69),(1.29,2.50),(6.81,2.50)],VISUAL,1.85,True,key="image-bypass")
    arrow((6.81,2.50),(6.81,2.65),VISUAL,1.85,True)
    text(6.65,2.26,"Visual on demand",size=10.8,color=VISUAL,weight="bold",ha="center",
         bbox=dict(boxstyle="round,pad=0.30",facecolor=VISUAL_TINT,
                   edgecolor=VISUAL,linewidth=.65))

    # One compact example replaces the former full-height benefits column.
    box(10.04,.36,3.56,3.60,color=LINE,fill=TINT,key="state-example")
    text(10.31,.72,"Text + symbols",size=13.0,color=ACCENT,weight="bold")
    text(10.31,1.30,"red.bottom_face",size=10.3,family="DejaVu Sans Mono")
    text(10.31,1.61,"blue.top_face",size=10.3,family="DejaVu Sans Mono")
    text(10.31,2.05,"center = (x, y, z)",size=10.1,family="DejaVu Sans Mono")
    cube(12.48,1.16,s=.80)
    line([(10.31,2.40),(13.34,2.40)],"#D4E2E0",.8)
    text(10.31,2.75,"Evidence",size=10.4,weight="bold")
    text(10.31,3.06,"observed / inferred / unavailable",size=9.1,color=MUTED)
    text(10.31,3.62,"Current values + timestamps",size=10.1,color=ACCENT)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [artist.get_window_extent(renderer) for artist in texts]
    for artist,extent in zip(texts,bounds):
        if not fig.bbox.contains(extent.x0,extent.y0) or not fig.bbox.contains(extent.x1,extent.y1):
            raise ValueError("Text outside canvas: "+artist.get_text())
    for i,extent in enumerate(bounds):
        for j in range(i):
            if extent.overlaps(bounds[j]):
                raise ValueError("Overlapping text: "+texts[i].get_text()+" / "+texts[j].get_text())
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in (".svg",".pdf",".png"):
        fig.savefig(output.with_suffix(suffix),dpi=dpi,facecolor="white")
        if suffix == ".svg":
            path = output.with_suffix(suffix)
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    return {"files":[str(output.with_suffix(s)) for s in (".svg",".pdf",".png")],
            "canvas_inches":[WIDTH,HEIGHT],"png_pixels":[round(WIDTH*dpi),round(HEIGHT*dpi)],
            "editable_svg_text":True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/media/geometric_abstraction_pitch")
    parser.add_argument("--dpi",type=int,default=280)
    args = parser.parse_args()
    print(json.dumps(render(args.output,args.dpi),indent=2))
