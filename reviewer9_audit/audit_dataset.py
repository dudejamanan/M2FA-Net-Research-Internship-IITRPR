from pathlib import Path
import csv, json, re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Dataset' / 'WeedyRice-RGBMS-DB'
OUT = Path(__file__).resolve().parent
splits = {s: (DATA / f'{s}_list.txt').read_text().splitlines() for s in ['train','val','test']}
membership = {n.strip():s for s, names in splits.items() for n in names}
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)

def measure(path):
    rgb = np.asarray(Image.open(path).convert('RGB'))
    mask = np.asarray(Image.open(DATA/'Masks'/f'{path.stem}.png').convert('L')) > 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = np.asarray(Image.fromarray(mask).resize((512,512), Image.Resampling.NEAREST))
    mean = gray.mean()
    # Screening proxies only: dark pixels are not confirmed shadows.
    return dict(filename=path.name,split=membership.get(path.name,'unknown'),width=rgb.shape[1],height=rgb.shape[0],weed_pct=100*mask.mean(),weed_pct_512_nearest=100*small.mean(),components=len(areas),min_component_px=int(areas.min()) if len(areas) else 0,components_1_100px=int(((areas>0)&(areas<=100)).sum()),mean_gray=float(mean),p05_gray=float(np.percentile(gray,5)),p95_gray=float(np.percentile(gray,95)),dark_lt50_pct=float(100*(gray<50).mean()),brightness_std=float(gray.std()))

paths = sorted(p for p in (DATA/'RGB').glob('*') if p.suffix.lower() in ['.jpg','.png'] and not p.name.startswith('._'))
with ThreadPoolExecutor(max_workers=8) as pool:
    rows = list(pool.map(measure,paths))
with (OUT/'image_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
test = [r for r in rows if r['split']=='test']
def short(name):
    return name.split('_lat_')[0].replace('DJI_DateTime_','')
def sheet(items, name, cols=4):
    cw,ch=400,355
    canvas=Image.new('RGB',(cols*cw, ((len(items)+cols-1)//cols)*ch),'white')
    draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(items):
        x,y=(i%cols)*cw,(i//cols)*ch
        im=Image.open(DATA/'RGB'/r['filename']).convert('RGB'); im.thumbnail((396,297))
        canvas.paste(im,(x,y))
        # Separate mask thumbnail preserves RGB appearance for visual inspection.
        m=Image.open(DATA/'Masks'/f"{Path(r['filename']).stem}.png").convert('L'); m.thumbnail((100,75))
        canvas.paste(m,(x+296,y+222))
        draw.text((x+3,y+300),short(r['filename']),font=font,fill='black')
        draw.text((x+3,y+323),f"weed {r['weed_pct']:.2f}% | gray {r['mean_gray']:.0f} | dark {r['dark_lt50_pct']:.0f}%",font=font,fill='black')
    canvas.save(OUT/name)
for start in range(0,len(test),24):
    sheet(test[start:start+24],f'test_contact_{start//24+1:02}.jpg')
rankings = {
    'small_weed':sorted([r for r in test if r['weed_pct']>0],key=lambda r:r['weed_pct'])[:16],
    'dark_candidates':sorted(test,key=lambda r:r['dark_lt50_pct'],reverse=True)[:16],
    'bright_candidates':sorted(test,key=lambda r:r['mean_gray'],reverse=True)[:16],
    'dense_candidates':sorted(test,key=lambda r:r['weed_pct'],reverse=True)[:16],
}
for key,items in rankings.items(): sheet(items,f'{key}.jpg')
summary={'total':len(rows),'split_counts':{s:sum(r['split']==s for r in rows) for s in splits},'test_weed_below_1pct':sum(0<r['weed_pct']<1 for r in test),'test_weed_below_5pct':sum(0<r['weed_pct']<5 for r in test),'test_gray_range':[min(r['mean_gray'] for r in test),max(r['mean_gray'] for r in test)],'rankings':rankings}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
print(json.dumps({k:v for k,v in summary.items() if k!='rankings'},indent=2,default=lambda x:x.item()))
