"""Render a schematic README animation with Pillow (no model execution).

Run: python docs/render_method_intro.py
Requires: Python >= 3.10 and Pillow >= 10. Fonts fall back across macOS/Linux.
The animation illustrates control flow, not measured speed or quality.
"""
from pathlib import Path
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

W, H, SCALE = 1200, 720, 2
BG = '#FAFBFD'
INK = '#17243B'
MUTED = '#69758A'
LINE = '#DCE3ED'
WHITE = '#FFFFFF'
PALE = '#F0F4FA'
ORANGE = '#C97513'
FONTS = {}


def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in FONTS:
        names = (['/System/Library/Fonts/Menlo.ttc', 'DejaVuSansMono.ttf'] if mono else
                 ['/System/Library/Fonts/Supplemental/Arial Bold.ttf', 'DejaVuSans-Bold.ttf'] if bold else
                 ['/System/Library/Fonts/Supplemental/Arial.ttf', 'DejaVuSans.ttf'])
        for name in names:
            try:
                FONTS[key] = ImageFont.truetype(name, size * SCALE)
                break
            except OSError:
                continue
        else:
            raise RuntimeError('Install DejaVu Sans or edit the font paths in this script.')
    return FONTS[key]


def mix(a, b, t):
    def rgb(c): return tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
    return tuple(round(x + (y-x)*t) for x, y in zip(rgb(a), rgb(b)))


class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W*SCALE, H*SCALE), BG)
        self.d = ImageDraw.Draw(self.im)

    def rect(self, box, fill=WHITE, outline=None, width=1, radius=16):
        self.d.rounded_rectangle(tuple(round(v*SCALE) for v in box),
                                 radius=radius*SCALE, fill=fill, outline=outline,
                                 width=width*SCALE)

    def line(self, points, fill=LINE, width=2):
        self.d.line([(round(x*SCALE), round(y*SCALE)) for x, y in points],
                    fill=fill, width=width*SCALE, joint='curve')

    def circle(self, x, y, r, fill, outline=None):
        self.d.ellipse(((x-r)*SCALE, (y-r)*SCALE, (x+r)*SCALE, (y+r)*SCALE),
                       fill=fill, outline=outline, width=2*SCALE)

    def text(self, x, y, value, size=20, color=INK, bold=False, mono=False, center=False):
        f = font(size, bold, mono)
        box = self.d.textbbox((0, 0), value, font=f)
        width = box[2] - box[0]
        px = x*SCALE - (width/2 if center else 0)
        assert px >= -1 and px+width <= W*SCALE+1, (value, px, width)
        self.d.text((px, y*SCALE-box[1]), value, fill=color, font=f)

    def arrow(self, pts, color=LINE, width=3, travel=None):
        self.line(pts, color, width)
        (x0,y0),(x,y)=pts[-2:]
        a=math.atan2(y-y0,x-x0)
        ends=[(x-9*math.cos(a+d), y-9*math.sin(a+d)) for d in [-0.5,0.5]]
        self.line([ends[0],(x,y),ends[1]],color,width)
        if travel is not None:
            lengths=[math.hypot(q[0]-p[0],q[1]-p[1]) for p,q in zip(pts,pts[1:])]
            distance=(travel%1)*sum(lengths)
            for p,q,n in zip(pts,pts[1:],lengths):
                if distance <= n:
                    k=distance/n if n else 0
                    self.circle(p[0]+k*(q[0]-p[0]),p[1]+k*(q[1]-p[1]),6,color)
                    break
                distance-=n

    def chip(self,x,y,w,label,color=ACCENT if 'ACCENT' in globals() else '#2563EB',active=True):
        self.rect((x,y,x+w,y+32),mix(color,WHITE,0.92) if active else PALE,None,radius=8)
        self.text(x+w/2,y+9,label,14,color if active else MUTED,bold=True,center=True)

    def card(self,x,y,w,h,label,active=True):
        self.rect((x,y+3,x+w,y+h+3),'#EBEFF5',None,radius=18)
        self.rect((x,y,x+w,y+h),WHITE,ACCENT if active else LINE,2 if active else 1,18)
        self.text(x+20,y+22,label,19,bold=True)

    def finish(self):
        return self.im.resize((W,H),Image.Resampling.LANCZOS)


def frame(stage,u,elapsed,poster=False):
    c=Canvas()
    c.text(55,30,PROJECT+'  /  METHOD IN MOTION',15,ACCENT,bold=True)
    c.chip(945,23,200,'FROZEN MODEL WEIGHTS',ACCENT)
    c.text(55,73,TITLE,36,bold=True)
    c.text(55,125,SUBTITLE,21,MUTED)
    for i,label in enumerate(STEPS):
        x=55+i*280
        active=i==stage
        c.rect((x,177,x+250,225),mix(ACCENT,WHITE,0.92) if active else WHITE,
               ACCENT if active else LINE,2 if active else 1,12)
        c.circle(x+25,201,12,ACCENT if active else '#E7ECF4')
        c.text(x+25,194,str(i+1),14,WHITE if active else MUTED,bold=True,center=True)
        c.text(x+49,193,label,18,ACCENT if active else MUTED,bold=True)
    draw_method(c,stage,u,poster)
    c.rect((55,597,1145,678),mix(ACCENT,WHITE,0.94),None,radius=14)
    c.text(75,614,CAPTIONS[stage],21,bold=True)
    c.text(75,649,FOOTNOTE,16,MUTED)
    c.rect((55,700,1145,704),'#E7ECF4',None,radius=2)
    c.rect((55,700,55+max(4,1090*elapsed),704),ACCENT,None,radius=2)
    return c.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'assets')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frames=[]
    for n in range(160):
        stage=n//40
        u=(n%40)/39
        frames.append(frame(stage,u,n/159))
    samples=Image.new('RGB',(W,H*4))
    for s in range(4): samples.paste(frames[s*40+25],(0,s*H))
    palette=samples.quantize(colors=192,method=Image.Quantize.MEDIANCUT)
    indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    path=args.output_dir/'method-intro.gif'
    indexed[0].save(path,save_all=True,append_images=indexed[1:],duration=100,loop=0,
                    optimize=True,disposal=1)
    frame(3,1,1,poster=True).save(args.output_dir/'method-overview.png',optimize=True)
    print(f'{PROJECT}: {path} ({path.stat().st_size/1024:.0f} KiB), 1200x720, 16 seconds')


PROJECT='AutoCache'
ACCENT='#0B8C7C'
TITLE='Learn when to compute, and when to reuse.'
SUBTITLE='Agent-designed caching policies for frozen image and video generators.'
STEPS=['Discover a policy','Read signals','Compute or reuse','Freeze & deploy']
CAPTIONS=[
 'An agent writes signals.py and updater.py; development feedback guides search.',
 'At each denoising step, the policy reads observations and extracted signals.',
 'Recompute to refresh the residual, or reuse it to skip model computation.',
 'Deploy a frozen policy; it makes cache decisions during generation.',
]
FOOTNOTE='Illustrative decisions, not a measured schedule. Sampling and cache execution stay with the framework.'


def draw_method(c,s,u,poster):
    c.card(55,270,275,244,'Frozen policy' if s==3 else 'Policy search',s in [0,3])
    c.rect((75,322,310,366),PALE,None,radius=10)
    c.text(91,336,'signals.py',18,ACCENT,mono=True)
    c.rect((75,381,310,425),PALE,None,radius=10)
    c.text(91,395,'updater.py',18,ACCENT,mono=True)
    c.text(75,453,'Evaluate → revise → select',16,MUTED)
    c.text(75,480,'Freeze the selected files',15,MUTED)
    c.arrow([(330,347),(388,347),(388,285),(531,285)],ACCENT if s in [0,1,3] else LINE,
            travel=u*2 if s in [0,1,3] else None)

    c.rect((531,259,791,312),mix(ACCENT,WHITE,0.92),ACCENT,2,12)
    c.text(661,278,'Policy: compute or reuse?',18,ACCENT,bold=True,center=True)
    c.arrow([(661,312),(661,338),(438,338),(438,391)],ACCENT,2,
            travel=u*2 if s in [1,2,3] else None)
    c.text(535,318,'action',13,ACCENT)
    c.text(840,276,'observations + signals',15,MUTED)
    c.arrow([(1095,307),(1095,325),(890,325),(890,306),(791,306)],
            ACCENT if s==1 else LINE,travel=u*2 if s==1 else None)

    schedule=[False,True,True,False,True,False,True]
    active=min(6,int(u*7))
    reuse=(s in [2,3]) and schedule[active]
    model_col=ORANGE if not reuse else '#CCD4E0'
    cache_col=ACCENT if reuse or poster else '#B8C3D3'
    c.circle(390,391,20,PALE,LINE)
    c.text(390,382,'x',21,bold=True,center=True)
    c.text(390,425,'state',15,MUTED,center=True)
    c.rect((484,356,683,424),mix(ORANGE,WHITE,0.93) if not reuse else PALE,model_col,2,12)
    c.text(584,380,'Model compute',19,model_col,bold=True,center=True)
    c.arrow([(411,391),(484,391)],model_col,travel=u*3 if not reuse else None)
    c.arrow([(683,391),(1070,391)],model_col,travel=u*3 if not reuse else None)
    c.circle(1093,391,22,mix(ACCENT,WHITE,0.91),ACCENT)
    c.text(1093,383,'x′',20,ACCENT,bold=True,center=True)
    c.text(1093,431,'next state',15,MUTED,center=True)

    c.rect((730,449,956,500),mix(ACCENT,WHITE,0.92),cache_col,2,12)
    c.text(843,467,'Cached residual',18,cache_col,bold=True,center=True)
    c.arrow([(584,425),(584,474),(730,474)],ORANGE if not reuse else LINE,
            travel=u*2 if not reuse else None)
    c.text(575,447,'refresh',14,ORANGE if not reuse else MUTED,center=True)
    c.arrow([(410,401),(436,401),(436,493),(730,493)],cache_col,
            travel=u*2 if reuse else None)
    c.arrow([(956,474),(1035,474),(1035,407),(1072,407)],cache_col,
            travel=u*2 if reuse else None)
    c.text(843,429,'REUSE' if reuse else 'REFRESH',13,ACCENT if reuse else ORANGE,bold=True,center=True)

    c.text(55,549,'Example denoising steps',16,MUTED)
    for i,r in enumerate(schedule):
        x=365+i*111
        col=ACCENT if r else ORANGE
        c.rect((x,535,x+96,572),mix(col,WHITE,0.91) if i<=active or poster else WHITE,
               col if i==active and s in [2,3] else LINE,2,9)
        c.text(x+48,547,('REUSE' if r else 'COMPUTE'),12,col,bold=True,center=True)


if __name__=='__main__': main()
