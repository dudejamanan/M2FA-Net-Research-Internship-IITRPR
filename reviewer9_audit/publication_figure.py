"""Publication layout for the six preselected test cases; no inference or data alteration."""
from pathlib import Path
import csv,io,json
import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor,Color
from reportlab.lib.units import mm
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'Dataset'/'WeedyRice-RGBMS-DB'
RUN=ROOT/'reviewer9_audit'/'checkpoint_analysis'
OUT=ROOT/'output'/'pdf'
OUT.mkdir(parents=True,exist_ok=True)
meta=json.loads((RUN/'inference_metadata.json').read_text())
assert meta['normalization']['provisional'] is False
selected=list(csv.DictReader((ROOT/'reviewer9_audit'/'selected_test_examples.csv').open()))
metrics={r['filename']:r for r in csv.DictReader((RUN/'per_image_metrics.csv').open())}
pdfmetrics.registerFont(TTFont('Paper','C:/Windows/Fonts/times.ttf'))
pdfmetrics.registerFont(TTFont('PaperBold','C:/Windows/Fonts/timesbd.ttf'))

W=170*mm
margin=2.5
gap=5
pw=(W-2*margin-3*gap)/4
ph=pw*3/4
header=19
rowhead=15
rowgap=3
footer=20
H=header+6*(rowhead+ph)+5*rowgap+footer
path=OUT/'figure_qualitative_six_cases.pdf'
c=canvas.Canvas(str(path),pagesize=(W,H),pageCompression=1)
c.setTitle('Qualitative analysis of M2FA-Net on six difficult test cases')
c.setAuthor('M2FA-Net research figure')
c.setSubject('RGB, binary ground truth, model predictions and segmentation errors; original checkpoint and normalization')

def text(x,y,s,size=10,bold=False):
    c.setFillColor(HexColor('#111111'))
    c.setFont('PaperBold' if bold else 'Paper',size)
    c.drawString(x,y,s)

def center(x,y,s,size=10,bold=False):
    c.setFillColor(HexColor('#111111'));c.setFont('PaperBold' if bold else 'Paper',size)
    c.drawCentredString(x,y,s)

def nettitle(cx,y):
    chunks=[('M',10.8,0),('2',7.3,4),('FA-Net',10.8,0)]
    width=sum(pdfmetrics.stringWidth(s,'PaperBold',sz) for s,sz,_ in chunks)
    x=cx-width/2
    for s,sz,dy in chunks:
        text(x,y+dy,s,sz,True);x+=pdfmetrics.stringWidth(s,'PaperBold',sz)

for j,label in enumerate(['RGB image','Ground truth','M2FA-Net','Error overlay']):
    cx=margin+j*(pw+gap)+pw/2
    if j==2: nettitle(cx,H-13)
    else: center(cx,H-13,label,10.8,True)

conditions=['Sparse weed region','Small isolated weed regions','Uneven brightness','Dense vegetation','Darker canopy','Dense irregular canopy']
mapping=[]
for i,(s,condition) in enumerate(zip(selected,conditions)):
    r=metrics[s['filename']];base=Path(s['filename']).stem
    rgb=cv2.resize(cv2.cvtColor(cv2.imread(str(DATA/'RGB'/s['filename'])),cv2.COLOR_BGR2RGB),(512,512))
    gt=cv2.resize(cv2.imread(str(DATA/'Masks'/(base+'.png')),0),(512,512))>0
    pred=cv2.imread(str(RUN/'predictions'/(base+'.png')),0)>0
    overlay=rgb.copy();colors=np.zeros_like(rgb)
    colors[pred&gt]=[0,200,65];colors[pred&~gt]=[255,145,0];colors[~pred&gt]=[50,105,255]
    active=pred|gt
    overlay[active]=(.22*rgb[active]+.78*colors[active]).astype('uint8')
    top=H-header-i*(rowhead+ph+rowgap)
    letter=chr(97+i)
    label=f'({letter})  {condition}'
    metric=f"IoU {100*float(r['iou']):.2f}%   |   Dice {100*float(r['dice']):.2f}%"
    text(margin,top-12,label,10.2,True)
    metricwidth=pdfmetrics.stringWidth(metric,'Paper',9.8)
    assert pdfmetrics.stringWidth(label,'PaperBold',10.2)+metricwidth+12<W-2*margin
    text(W-margin-metricwidth,top-12,metric,9.8)
    panels=[rgb,(gt*255).astype('uint8'),(pred*255).astype('uint8'),overlay]
    bottom=top-rowhead-ph
    for j,a in enumerate(panels):
        x=margin+j*(pw+gap)
        img=Image.fromarray(a)
        # Display the same image extent in its native 4:3 aspect ratio in every column.
        # Binary masks remain exact raster samples; no new annotations are introduced.
        stream=io.BytesIO();img.save(stream,format='PNG');stream.seek(0)
        c.drawImage(ImageReader(stream),x,bottom,width=pw,height=ph,mask='auto')
        c.setStrokeColor(Color(.72,.72,.72));c.setLineWidth(.25);c.rect(x,bottom,pw,ph,stroke=1,fill=0)
    mapping.append(dict(panel=letter,condition=condition,filename=s['filename'],iou=float(r['iou']),dice=float(r['dice'])))

legend=[('#00C841','True positive'),('#FF9100','False positive'),('#3269FF','False negative')]
size=10.2;box=7
widths=[box+4+pdfmetrics.stringWidth(label,'Paper',size) for _,label in legend]
spacing=22
x=(W-sum(widths)-spacing*2)/2
y=7.5
for (color,label),width in zip(legend,widths):
    c.setFillColor(HexColor(color));c.rect(x,y,box,box,stroke=0,fill=1)
    text(x+box+4,y-0.2,label,size)
    x+=width+spacing
c.showPage();c.save()

caption='''Qualitative analysis of M²FA-Net on six challenging test images: (a) a sparse weed region, (b) small isolated weed regions, (c) spatially uneven brightness, (d) dense vegetation, (e) a darker canopy, and (f) a dense irregular canopy. Columns show the RGB image, binary ground truth, M²FA-Net prediction, and error overlay. Green, orange, and blue indicate true-positive, false-positive, and false-negative pixels, respectively. Foreground IoU and Dice are reported for each image. Panels (a) and (b) illustrate missed weed areas and false-positive detections, while panel (d) shows over-segmentation. The darker canopy in (e) is not a verified severe cast-shadow example.'''
notes='''# Six-case figure for the manuscript

Use the PDF for LaTeX to retain embedded, selectable text. The figure is 170 mm wide and approximately 234 mm high, intended for full-page width. Font sizes are 9.8-10.8 pt at that width. Avoid shrinking it to a single column. PNG and LZW-compressed TIFF are exported at 600 dpi.

## Suggested caption

'''+caption+'''

## LaTeX insertion

```latex
\\begin{figure*}[p]
    \\centering
    \\includegraphics[width=170mm]{Images/figure_qualitative_six_cases.pdf}
    \\caption{Qualitative analysis of $M^2$FA-Net on six challenging test images: (a) a sparse weed region, (b) small isolated weed regions, (c) spatially uneven brightness, (d) dense vegetation, (e) a darker canopy, and (f) a dense irregular canopy. Columns show the RGB image, binary ground truth, $M^2$FA-Net prediction, and error overlay. Green, orange, and blue indicate true-positive, false-positive, and false-negative pixels, respectively. Foreground IoU and Dice are reported for each image. The darker canopy in (e) is not a verified severe cast-shadow example.}
    \\label{fig:difficult_cases}
\\end{figure*}
```

Place this six-row figure on a dedicated figure page. If the journal requires a substantially narrower figure width, split the layout into two groups of three rows instead of shrinking the font.

## Exact sample mapping

| Panel | Sample | Weed IoU | Dice |
|---|---|---|---|
'''
for row in mapping:
    notes+=f"| ({row['panel']}) | `{row['filename']}` | {row['iou']*100:.2f}% | {row['dice']*100:.2f}% |\n"
notes+='\nThe six cases are the condition-based shortlist selected before inference, not the six lowest-IoU samples. Predictions use the supplied checkpoint and original normalization. No baseline predictions were added. Display extent is restored to the native 4:3 aspect ratio for all columns; evaluation remains at 512 x 512 using the original code.\n'
(OUT/'figure_qualitative_six_cases_caption.md').write_text(notes,encoding='utf-8')
(OUT/'figure_qualitative_six_cases_samples.json').write_text(json.dumps(mapping,indent=2))
reader=PdfReader(path)
assert len(reader.pages)==1
extracted=reader.pages[0].extract_text()
for i in range(6): assert f'({chr(97+i)})' in extracted
assert '23.57%' in extracted and '89.25%' in extracted
print(f'Created {path}; page {W/mm:.1f} x {H/mm:.1f} mm; fonts 9.8-10.8 pt; verified six panel labels and scores.')
