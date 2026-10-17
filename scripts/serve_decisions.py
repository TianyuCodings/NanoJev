#!/usr/bin/env python3
"""Serve a local checkpoint and reader demo; weights load once, no provider calls."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
from pathlib import Path
import time
from urllib.parse import unquote, urlparse


def server_class(engine, web_root):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, content, mime='application/json; charset=utf-8'):
            self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(content)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)

        def send_json(self, code, data):
            self.send(code,json.dumps(data,ensure_ascii=False,allow_nan=False).encode())

        def do_GET(self):
            route=urlparse(self.path).path
            if route=='/api/health':
                self.send_json(200,{'ready':True,'model_loaded_once':True,'provider_calls':0});return
            relative=unquote(route).lstrip('/') or 'index.html'
            target=(web_root/relative).resolve()
            if not target.is_relative_to(web_root) or not target.is_file():
                self.send_json(404,{'error':'File not found'});return
            self.send(200,target.read_bytes(),mimetypes.guess_type(str(target))[0] or 'application/octet-stream')

        def do_POST(self):
            if urlparse(self.path).path!='/api/evaluate':
                self.send_json(404,{'error':'Unknown endpoint'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 2_000_000:raise ValueError('Request must contain 1..2000000 bytes')
                origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):raise ValueError('Cross-origin requests are disabled')
                from predict_toy_decisions import unique_object,reject_nonfinite,validate_request
                payload=json.loads(self.rfile.read(length),object_pairs_hook=unique_object,parse_constant=reject_nonfinite)
                states=validate_request(payload)
                questions=[q for s in states for q in s['questions'].values()]
                paths=sum(1 if q['type']=='boolean' else len(q['criteria']) for q in questions)
                if len(states)>32 or len(questions)>96 or paths>256:raise ValueError('Local demo limit:32 states,96 questions,256 candidate paths per request')
                before=time.perf_counter();result=engine.predict(payload)
                result['execution']['server_evaluation_seconds']=time.perf_counter()-before
                self.send_json(200,result)
            except (ValueError,TypeError,KeyError) as exc:
                self.send_json(400,{'error':str(exc)})
            except Exception:
                self.send_json(500,{'error':'Local model inference failed; inspect the server process. No teacher fallback was used.'})
                raise

        def log_message(self, fmt, *args):
            # HTTP method/path/status only; request states and credentials are not logged.
            print(fmt % args,flush=True)
    return Handler


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--web-root',default='web');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');p.add_argument('--device',default='auto',help='auto/mps/cpu/cuda:0');p.add_argument('--disable-native-triton',action='store_true')
    a=p.parse_args()
    from predict_toy_decisions import DecisionPredictor
    engine=DecisionPredictor(a.checkpoint_dir,device_name=a.device,precision=a.precision,disable_native_triton=a.disable_native_triton)
    root=Path(a.web_root).resolve()
    if not (root/'index.html').is_file():raise ValueError('web-root must contain index.html')
    server=HTTPServer((a.host,a.port),server_class(engine,root))
    print(json.dumps({'url':f'http://{a.host}:{a.port}','ready':True,'provider_calls':0}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
