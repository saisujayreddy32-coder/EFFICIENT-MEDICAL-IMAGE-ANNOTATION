from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from PIL import Image
import numpy as np, os, json, uuid, random, shutil
from .ml import discover_pairs, train_model, predict, load_gray, IMAGE_SIZE

BASE=Path(__file__).resolve().parents[1]; DATA=BASE/'data'; INBOX=DATA/'inbox'; MASKS=DATA/'annotations'; MODELS=BASE/'models'; OUTPUTS=BASE/'outputs'
for p in (DATA,INBOX,MASKS,MODELS,OUTPUTS): p.mkdir(parents=True,exist_ok=True)
MODEL_PATH=MODELS/'lung_unet.pt'; STATE={'dataset_root':str(DATA),'pairs':[],'unlabelled':[],'queue':[],'history':[],'last_train':None,'strategy':'uncertainty'}
app=Flask(__name__,static_folder='../frontend',static_url_path='')
CORS(app)

@app.get('/')
def home(): return send_from_directory(app.static_folder,'index.html')
@app.get('/<path:path>')
def static_files(path): return send_from_directory(app.static_folder,path)

def refresh_dataset(root=None):
    root=Path(root or STATE['dataset_root']); pairs, unlabelled=discover_pairs(root)
    # Include user-saved annotations as labelled pairs.
    for img in list(INBOX.glob('*')):
        ann=MASKS/(img.stem+'.png')
        if ann.exists(): pairs.append((str(img),str(ann)))
        elif img.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}: unlabelled.append(str(img))
    STATE['dataset_root']=str(root); STATE['pairs']=pairs; STATE['unlabelled']=sorted(set(unlabelled)); return pairs, STATE['unlabelled']

@app.get('/api/status')
def status():
    pairs, pool=refresh_dataset()
    return jsonify({'dataset_root':STATE['dataset_root'],'labelled':len(pairs),'unlabelled':len(pool),'model_ready':MODEL_PATH.exists(),'device':'CUDA' if __import__('torch').cuda.is_available() else 'CPU','strategy':STATE['strategy'],'history':STATE['history'],'last_train':STATE['last_train'],'queue_count':len(STATE['queue'])})

@app.post('/api/dataset/scan')
def scan():
    data=request.get_json(silent=True) or {}; root=data.get('path') or str(DATA)
    if not Path(root).exists(): return jsonify({'error':'Dataset path does not exist.'}),400
    pairs,pool=refresh_dataset(root)
    return jsonify({'labelled':len(pairs),'unlabelled':len(pool),'dataset_root':str(root),'message':'Dataset scan completed. Images without matching masks are kept in the annotation pool.'})

@app.post('/api/upload')
def upload():
    files=request.files.getlist('files')
    if not files: return jsonify({'error':'Choose one or more image files.'}),400
    saved=[]
    for f in files:
        if not f.filename: continue
        ext=Path(f.filename).suffix.lower()
        if ext not in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}: continue
        name=f'{uuid.uuid4().hex[:10]}_{Path(f.filename).name}'; f.save(INBOX/name); saved.append(name)
    refresh_dataset(); return jsonify({'saved':len(saved),'files':saved})

@app.get('/api/queue')
def queue():
    refresh_dataset()
    if not STATE['queue']: build_queue('uncertainty',min(24,len(STATE['unlabelled'])))
    items=[]
    for item in STATE['queue']:
        p=Path(item['path'])
        if p.exists(): items.append({'id':item['id'],'name':p.name,'score':item['score'],'method':item['method'],'url':'/api/image/'+item['id']})
    return jsonify({'items':items,'strategy':STATE['strategy']})

def build_queue(strategy='uncertainty',count=12):
    refresh_dataset(); STATE['strategy']=strategy; scored=[]
    for path in STATE['unlabelled']:
        try:
            arr=load_gray(path)
            if strategy=='uncertainty' and MODEL_PATH.exists():
                _,score=predict(MODEL_PATH,path)
            else:
                # Transparent cold-start proxy: normalized grayscale entropy + edge density.
                hist=np.histogram(arr,bins=64,range=(0,1),density=True)[0]; hist=hist/(hist.sum()+1e-9); ent=float(-(hist*np.log(hist+1e-9)).sum()/np.log(64))
                gy,gx=np.gradient(arr); edge=float(np.mean(np.sqrt(gx*gx+gy*gy))); score=min(1.0,0.82*ent+0.18*min(edge*5,1))
            scored.append({'id':uuid.uuid5(uuid.NAMESPACE_URL,path).hex,'path':path,'score':round(float(score),5),'method':strategy})
        except Exception: continue
    if strategy=='random': random.Random(42).shuffle(scored)
    else: scored.sort(key=lambda x:x['score'],reverse=True)
    STATE['queue']=scored[:count]

@app.post('/api/queue/build')
def queue_build():
    data=request.get_json(silent=True) or {}; strategy=data.get('strategy','uncertainty'); count=max(1,min(int(data.get('count',12)),100))
    if strategy not in ('uncertainty','random'): return jsonify({'error':'Strategy must be uncertainty or random.'}),400
    build_queue(strategy,count); return jsonify({'count':len(STATE['queue']),'strategy':strategy,'items':[{'id':x['id'],'name':Path(x['path']).name,'score':x['score']} for x in STATE['queue']]})

@app.get('/api/image/<image_id>')
def image(image_id):
    for item in STATE['queue']:
        if item['id']==image_id: return send_from_directory(Path(item['path']).parent,Path(item['path']).name)
    return jsonify({'error':'Image not found in current queue.'}),404

@app.post('/api/annotate')
def annotate():
    data=request.get_json(force=True); image_id=data.get('id'); mask_data=data.get('mask')
    item=next((x for x in STATE['queue'] if x['id']==image_id),None)
    if not item or not mask_data: return jsonify({'error':'Missing queue image or mask.'}),400
    import base64
    try:
        raw=base64.b64decode(mask_data.split(',',1)[1]); from io import BytesIO
        im=Image.open(BytesIO(raw)).convert('L').resize((IMAGE_SIZE,IMAGE_SIZE)); out=MASKS/(Path(item['path']).stem+'.png'); im.save(out)
    except Exception as e: return jsonify({'error':f'Could not save annotation: {e}'}),400
    refresh_dataset(); STATE['queue']=[x for x in STATE['queue'] if x['id']!=image_id]
    return jsonify({'saved':True,'labelled':len(STATE['pairs']),'unlabelled':len(STATE['unlabelled']),'message':'Annotation saved and added to the labelled training pool.'})

@app.post('/api/train')
def train():
    pairs,_=refresh_dataset(); data=request.get_json(silent=True) or {}; epochs=max(1,min(int(data.get('epochs',3)),20))
    try: result=train_model(pairs,MODEL_PATH,epochs=epochs); STATE['history']=result['history']; STATE['last_train']={'device':result['device'],'train_count':result['train_count'],'validation_count':result['validation_count'],'epochs':epochs}; return jsonify({'success':True,**result})
    except Exception as e: return jsonify({'error':str(e)}),400

@app.post('/api/predict')
def infer():
    f=request.files.get('image')
    if not f: return jsonify({'error':'Upload an X-ray image.'}),400
    path=INBOX/('inference_'+uuid.uuid4().hex+'.png'); f.save(path)
    if not MODEL_PATH.exists(): return jsonify({'error':'Train the segmentation model before requesting a prediction.'}),400
    probs,unc=predict(MODEL_PATH,str(path)); from PIL import Image as PILImage
    out=OUTPUTS/(path.stem+'_mask.png'); PILImage.fromarray((probs*255).astype('uint8')).save(out)
    return jsonify({'mask_url':'/api/output/'+out.name,'uncertainty':round(unc,4),'note':'Research visualization only; not a clinical diagnosis.'})

@app.get('/api/output/<name>')
def output(name): return send_from_directory(OUTPUTS,name)

@app.get('/api/metrics')
def metrics():
    pairs,pool=refresh_dataset(); total=len(pairs)+len(pool); saved=(1-len(pairs)/total)*100 if total else 0
    return jsonify({'labelled':len(pairs),'unlabelled':len(pool),'total':total,'annotation_savings_vs_full_labeling':round(saved,1),'history':STATE['history'],'comparison':{'random':None,'uncertainty':None},'note':'Savings are a pool-coverage estimate, not a measured clinical performance claim. Compare strategies at equal annotation budgets.'})

if __name__=='__main__': app.run(host='127.0.0.1',port=8000,debug=True)
