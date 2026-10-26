import json
import queue
import tempfile
import time
from eeg_pybci import Session

def wait_for(session, predicate, seconds=12):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        try: message=session.messages.get(timeout=.5)
        except queue.Empty: continue
        print(message,flush=True)
        if message['kind']=='error': raise RuntimeError(message['text'])
        if predicate(message): return message
    raise TimeoutError('session event not received')

def main():
    with tempfile.TemporaryDirectory() as directory:
        s=Session(demo=True,window=1,minimum=2,prepare=.2,directory=directory)
        s.start()
        try:
            wait_for(s,lambda m:m['kind']=='connected')
            time.sleep(.5)
            for label in ['rest','math','rest','math']:
                s.request(label)
                wait_for(s,lambda m:m['kind']=='trial_done')
                time.sleep(.4)
            msg=wait_for(s,lambda m:m['kind']=='counts' and len(m['counts'])==2 and min(v[1] for v in m['counts'].values())>=2)
            s.request('predict')
            wait_for(s,lambda m:m['kind']=='mode')
            wait_for(s,lambda m:m['kind']=='prediction' and m['text'] in ['睁眼静坐','心算'])
            assert (s.directory/'model.joblib').exists()
            assert (s.directory/'training_features.npz').exists()
        finally:
            s.stop.set()
            s.thread.join(timeout=10)
            assert not s.thread.is_alive()
        loaded=Session(demo=True,window=1,minimum=2,directory=directory,model=s.directory/'model.joblib')
        loaded.start()
        try:
            wait_for(loaded,lambda m:m['kind']=='connected' and m['mode']=='predict')
            wait_for(loaded,lambda m:m['kind']=='prediction' and m['text']=='睁眼静坐')
        finally:
            loaded.stop.set()
            loaded.thread.join(timeout=10)
            assert not loaded.thread.is_alive()
        print('PASS: real PyBCI + LSL demo train/save/predict/shutdown',flush=True)

if __name__=='__main__': main()
