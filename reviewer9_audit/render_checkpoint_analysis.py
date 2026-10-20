"""Assemble evidence panels from original inputs, annotations, and saved model outputs."""
from pathlib import Path
import csv,json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'Dataset'/'WeedyRice-RGBMS-DB'
OUT=ROOT/'reviewer9_audit'/'checkpoint_analysis'
FONT='C:/Windows/Fonts/arial.ttf'
f=ImageFont.truetype(FONT,21)
small=ImageFont.truetype(FONT,18)
titlefont=ImageFont.truetype(FONT,27)
rows=list(csv.DictReader((OUT/'per_image_metrics.csv').open()))
lookup={r['filename']:r for r in rows}
metadata=json.loads((OUT/'inference_metadata.json').read_text())
provisional=metadata['normalization']['provisional']

def arrays(name):
    base=Path(name).stem
    rgb=cv2.resize(cv2.cvtColor(cv2.imread(str(DATA/'RGB'/name)),cv2.COLOR_BGR2RGB),(512,512))
    gt=cv2.resize(cv2.imread(str(DATA/'Masks'/(base+'.png')),0),(512,512))>0
    pred=cv2.imread(str(OUT/'predictions'/(base+'.png')),0)>0
    return rgb,gt,pred

def errors(rgb,gt,pred):
    colors=np.zeros_like(rgb)
    colors[pred&gt]=[0,220,60]
    colors[pred&~gt]=[255,145,0]
    colors[~pred&gt]=[50,110,255]
    active=pred|gt
    output=rgb.copy()
    output[active]=(.28*rgb[active]+.72*colors[active]).astype(np.uint8)
    return output

def render(items,path,title):
    size=350; pad=14; rowheight=size+66; width=4*(size+pad)+pad
    canvas=Image.new('RGB',(width,145+len(items)*rowheight),'white'); d=ImageDraw.Draw(canvas)
    d.text((pad,8),title,font=titlefont,fill='black')
    status='PROVISIONAL: reconstructed training normalization; verify original ms_mean_std.npy.' if provisional else 'Original supplied normalization; single checkpoint; supplied test split.'
    d.text((pad,45),status,font=small,fill='#9b1c1c' if provisional else 'black')
    d.text((pad,72),'Errors: green = true positive | orange = false positive | blue = false negative',font=small,fill='black')
    for j,label in enumerate(['RGB input','Ground truth','M2FA-Net prediction','Error overlay']):
        d.text((pad+j*(size+pad),108),label,font=f,fill='black')
    for i,(name,label) in enumerate(items):
        r=lookup[name]; rgb,gt,pred=arrays(name)
        y=145+i*rowheight
        short=Path(name).stem.replace('DJI_DateTime_','').split('_lat_')[0]
        d.text((pad,y),f'{short} | {label}',font=f,fill='black')
        d.text((pad,y+27),f"Weed IoU {float(r['iou'])*100:.2f}% | Dice {float(r['dice'])*100:.2f}% | precision {float(r['precision'])*100:.1f}% | recall {float(r['recall'])*100:.1f}%",font=small,fill='black')
        panels=[rgb,np.repeat((gt*255).astype('uint8')[...,None],3,axis=2),np.repeat((pred*255).astype('uint8')[...,None],3,axis=2),errors(rgb,gt,pred)]
        for j,a in enumerate(panels):
            method=Image.Resampling.NEAREST if j in [1,2] else Image.Resampling.LANCZOS
            canvas.paste(Image.fromarray(a).resize((size,size),method),(pad+j*(size+pad),y+60))
    canvas.save(OUT/path,dpi=(300,300))

selected=list(csv.DictReader((ROOT/'reviewer9_audit'/'selected_test_examples.csv').open()))
items=[(r['filename'],r['condition']) for r in selected if r['filename'] in lookup]
if items: render(items,'difficult_test_cases.png','Difficult test conditions: measured predictions')
worst=sorted(rows,key=lambda r:float(r['iou']))[:6]
render([(r['filename'],'lowest foreground IoU subset') for r in worst],'lowest_iou_cases.png',f'Lowest foreground IoU cases among {len(rows)} evaluated test images')

# Separate zooms use the largest connected false-negative component; locations are marked.
zoomrows=[]
for r in worst[:3]:
    rgb,gt,pred=arrays(r['filename']); fn=(gt&~pred).astype('uint8')
    n,_,stats,_=cv2.connectedComponentsWithStats(fn,8)
    if n<=1: continue
    s=stats[1:][np.argmax(stats[1:,cv2.CC_STAT_AREA])]
    x,y,w,h,_=map(int,s); x0=max(x-16,0); y0=max(y-16,0); x1=min(x+w+16,512); y1=min(y+h+16,512)
    zoomrows.append((r,rgb,gt,pred,(x0,y0,x1,y1)))
if zoomrows:
    canvas=Image.new('RGB',(1400,100+len(zoomrows)*395),'white'); d=ImageDraw.Draw(canvas)
    d.text((12,8),'Failure details: largest missed annotated region',font=titlefont,fill='black')
    d.text((12,45),'PROVISIONAL normalization' if provisional else 'Original supplied normalization',font=f,fill='#9b1c1c' if provisional else 'black')
    d.text((12,73),'Errors: green = true positive | orange = false positive | blue = false negative',font=small,fill='black')
    for i,(r,rgb,gt,pred,box) in enumerate(zoomrows):
        y=100+i*395
        short=Path(r['filename']).stem.replace('DJI_DateTime_','').split('_lat_')[0]
        d.text((12,y),f"{short} | full-image IoU {float(r['iou'])*100:.2f}% | crop xyxy={box}",font=f,fill='black')
        rgbim=Image.fromarray(rgb); ImageDraw.Draw(rgbim).rectangle(box,outline='red',width=4)
        panels=[rgbim,Image.fromarray(rgb).crop(box),Image.fromarray((gt*255).astype('uint8')).crop(box).convert('RGB'),Image.fromarray(errors(rgb,gt,pred)).crop(box)]
        for j,(im,label) in enumerate(zip(panels,['Full RGB: crop in red','RGB crop','Ground-truth crop','Error crop'])):
            d.text((12+j*348,y+30),label,font=small,fill='black')
            scale=min(330/im.width,330/im.height)
            im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST if j==2 else Image.Resampling.LANCZOS)
            panel=Image.new('RGB',(330,330),(242,242,242))
            panel.paste(im,((330-im.width)//2,(330-im.height)//2))
            canvas.paste(panel,(12+j*348,y+55))
    canvas.save(OUT/'failure_zoom_details.png',dpi=(300,300))

with (OUT/'ranked_test_cases.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(sorted(rows,key=lambda r:float(r['iou'])))
print('Saved condition panels, lowest-IoU panels, failure zooms, and ranked CSV.')
print('Worst six:',[(r['filename'].split('_lat_')[0],round(float(r['iou']),4)) for r in worst])
