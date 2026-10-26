"""PyBCI 静坐/心算学习：先启动 eeg_viewer.py 并连接，再运行此程序。"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import joblib
import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, local_clock, resolve_byprop
from sklearn.linear_model import LogisticRegression
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings
from pybci.Utils.Logger import Logger
from bci_features import MetricsOutlet, MetricFeatures, LABELS, stream_names

TASK_NAMES = {'rest':'睁眼静坐', 'math':'心算'}


def query(bci, event_name, queue_name, timeout=3):
    event, responses = getattr(bci,event_name), getattr(bci,queue_name)
    while True:
        try: responses.get_nowait()
        except queue.Empty: break
    event.set()
    try:
        return responses.get(timeout=timeout)
    finally:
        event.clear()


class Session:
    def __init__(self, demo=False, window=8, minimum=6, prepare=2, directory=None, model=None):
        self.demo, self.window, self.minimum, self.prepare = demo, window, minimum, prepare
        self.model_path = model
        self.commands, self.messages = queue.Queue(), queue.Queue()
        self.stop = threading.Event()
        self.bci = None
        self.features = MetricFeatures(window)
        self.demo_task = 'rest'
        self.directory = Path(directory or Path(__file__).resolve().parent/'bci_sessions') / (
            ('demo_' if demo else 'session_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.thread = threading.Thread(target=self.run,daemon=True)

    def start(self): self.thread.start()

    def send(self, kind, **items): self.messages.put({'kind':kind, **items})

    def request(self, command): self.commands.put(command)

    def event(self, name, **details):
        self.event_file.write(json.dumps({'event':name,'utc':datetime.now(timezone.utc).isoformat(),
                              'lsl_time':local_clock(),**details},ensure_ascii=False)+'\n')
        self.event_file.flush()

    def demo_source(self):
        outlet=MetricsOutlet(True)
        count=0
        rng=np.random.default_rng(7)
        while not self.stop.wait(.02):
            v=[0.]*21
            v[0],v[2]=count%256,1
            v[15:18]=np.array([20,50,5] if self.demo_task=='rest' else [35,25,10])+rng.normal(0,.4,3)
            outlet.push(v)
            count+=1

    def run(self):
        monitor=None
        demo_thread=None
        try:
            self.directory.mkdir(parents=True)
            with (self.directory/'events.jsonl').open('x',encoding='utf-8') as self.event_file, \
                 (self.directory/'metrics.csv').open('x',encoding='utf-8-sig',newline='') as metrics_file:
                writer=csv.writer(metrics_file)
                writer.writerow(['lsl_timestamp',*LABELS])
                self.event('session_start', demo=self.demo,window_seconds=self.window,minimum_trials_per_class=self.minimum)
                data_name, marker_name=stream_names(self.demo)
                marker_info=StreamInfo(marker_name,'Markers',1,0,'string',marker_name)
                marker_outlet=StreamOutlet(marker_info)
                if self.demo:
                    demo_thread=threading.Thread(target=self.demo_source,daemon=True)
                    demo_thread.start()
                settings=GlobalEpochSettings()
                settings.tmin=0
                settings.tmax=self.window
                settings.windowLength=self.window
                settings.windowOverlap=0
                settings.splitCheck=False
                self.bci=PyBCI(dataStreams=[data_name],markerStream=marker_name,
                               streamTypes=['DeviceMetrics'],globalEpochSettings=settings,
                               minimumEpochsRequired=self.minimum,
                               streamCustomFeatureExtract={data_name:self.features},
                               clf=LogisticRegression(max_iter=1000,random_state=0),loggingLevel=Logger.WARNING)
                self.send('status',text='等待设备指标流；请在监视器中连接设备…')
                while not self.stop.is_set() and not self.bci.Connect():
                    self.stop.wait(1)
                if self.stop.is_set(): return
                resolved=resolve_byprop('name',data_name,timeout=3)
                if len(resolved)!=1:
                    raise ValueError('指标流缺失或存在重名流，请只保留一个设备源')
                monitor=StreamInlet(resolved[0])
                monitor.open_stream(timeout=3)
                self.bci.TrainMode()
                mode='train'
                mapping={}
                if self.model_path:
                    saved=joblib.load(self.model_path)
                    if saved['demo'] != self.demo or saved['window_seconds'] != self.window:
                        raise ValueError('模型的模拟/实机模式或窗口长度与当前配置不一致')
                    classifier=self.bci.classifierThread.classifier
                    classifier.clf, classifier.scaler=saved['classifier'],saved['scaler']
                    mapping=saved['labels']
                    self.bci.TestMode()
                    mode='predict'
                self.send('connected',text='已连接。交替录制静坐与心算，每类至少 '+str(self.minimum)+' 次。',mode=mode)
                phase=None
                counts={}
                latest=None
                last_poll=last_flush=0
                pending_predict=False
                last_saved_counts=None
                while not self.stop.wait(.05):
                    chunk, stamps=monitor.pull_chunk(timeout=0,max_samples=4096)
                    for row, stamp in zip(chunk,stamps):
                        writer.writerow([stamp,*row])
                        latest=row
                    now=time.monotonic()
                    live=latest is not None and local_clock()-latest[5]<.75 and latest[3]==1
                    if now-last_flush>1:
                        metrics_file.flush()
                        last_flush=now
                    try: command=self.commands.get_nowait()
                    except queue.Empty: command=None
                    if command in TASK_NAMES:
                        if mode!='train' or phase:
                            self.send('notice',text='请等待当前试次结束；预测模式不再采集训练标签。')
                        elif not live:
                            self.send('notice',text='没有新鲜且佩戴正常的数据，暂不开始试次。')
                        else:
                            self.demo_task=command
                            phase={'task':command,'stage':'prepare','due':now+self.prepare}
                            self.event('prepare',label=command)
                            self.send('trial',text=f'准备 {TASK_NAMES[command]}，{self.prepare:g} 秒后开始；保持头部静止。')
                    if command=='predict' and not phase and mode=='train':
                        if len(counts)==2 and min(v[1] for v in counts.values())>=self.minimum:
                            pending_predict=True
                        else:
                            self.send('notice',text='有效试次数不足，请继续交替采集两类。')
                    if phase and now>=phase['due']:
                        if phase['stage']=='prepare':
                            marker_outlet.push_sample([phase['task']])
                            self.event('marker',label=phase['task'],window_seconds=self.window)
                            phase.update(stage='collect',due=now+self.window+.5)
                            self.send('trial',text=f'正在 {TASK_NAMES[phase["task"]]}：保持 {self.window:g} 秒…')
                        else:
                            self.event('trial_finished',label=phase['task'])
                            phase=None
                            self.send('trial_done',text='本次采集结束。查看有效次数是否增加；建议下一次换另一类。')
                    if now-last_poll<1: continue
                    last_poll=now
                    if mode=='train':
                        counts=query(self.bci,'markerCountRetrieveEvent','markerCountQueue')
                        self.send('counts',counts=counts,quality=self.features.snapshot()[1])
                        # Archive features as they arrive, even if the user exits before prediction.
                        count_tuple=tuple((k,v[1]) for k,v in sorted(counts.items()))
                        if count_tuple!=last_saved_counts:
                            feature_data=query(self.bci,'queryFeaturesEvent','queryFeaturesQueue')
                            np.savez(self.directory/'training_features.npz',**feature_data)
                            (self.directory/'labels.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding='utf-8')
                            last_saved_counts=count_tuple
                        if pending_predict:
                            classifier=self.bci.classifierThread.classifier
                            total=sum(v[1] for v in counts.values())
                            if getattr(classifier,'_trained_count',0)<total: continue
                            self.bci.TestMode()
                            mapping={v[0]:k for k,v in counts.items()}
                            model={'classifier':classifier.clf,'scaler':classifier.scaler,'labels':mapping,
                                   'demo':self.demo,'window_seconds':self.window,'feature_version':1,
                                   'validation':'not independently validated'}
                            joblib.dump(model,self.directory/'model.joblib')
                            self.event('prediction_start',labels=mapping)
                            self.features.last_valid=0
                            mode='predict'
                            self.send('mode',text='模型已保存，开始实时预测；尚未做独立验证。',mode=mode)
                    else:
                        guess=query(self.bci,'classifierGuessMarkerEvent','classifierGuessMarkerQueue',timeout=2)
                        valid_at,quality,_,_=self.features.snapshot()
                        valid=live and valid_at>0 and now-valid_at<=self.window+1
                        label=None
                        if valid and isinstance(guess,(np.ndarray,list,tuple,int,float,np.number)) and np.asarray(guess).size==1:
                            try: label=mapping.get(int(np.asarray(guess).item()))
                            except (ValueError,TypeError): pass
                        text=TASK_NAMES.get(label,'等待有效完整片段')
                        self.send('prediction',text=text,quality=quality)
                        self.event('prediction',label=label,quality=quality)
                self.event('session_end')
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.send('error',text=f'{type(exc).__name__}: {exc}')
        finally:
            self.stop.set()
            if self.bci and self.bci.connected:
                self.bci.StopThreads()
            if monitor: monitor.close_stream()
            if demo_thread: demo_thread.join(timeout=1)
            self.send('closed',text=f'会话已保存：{self.directory}')


class App:
    def __init__(self, root, session):
        self.root,self.session=root,session
        root.title('PyBCI · 静坐 / 心算学习'+(' · 模拟数据' if session.demo else ''))
        root.geometry('820x470')
        ttk.Label(root,text='PyBCI 任务训练',font=('Microsoft YaHei UI',20,'bold')).pack(pady=18)
        self.status=ttk.Label(root,text='正在连接…',wraplength=750)
        self.status.pack(pady=8)
        ttk.Label(root,text='静坐：睁眼、放松下颌。心算：默算连续减 7，保持姿势一致。\n每次试次前有准备时间；动作和接触异常的片段会跳过。',justify='center').pack(pady=10)
        frame=ttk.Frame(root);frame.pack(pady=12)
        self.buttons=[]
        for key,label in [('rest','录制静坐试次'),('math','录制心算试次'),('predict','训练完成 → 保存并预测')]:
            button=ttk.Button(frame,text=label,command=lambda k=key:session.request(k),state='disabled')
            button.pack(side='left',padx=8);self.buttons.append(button)
        self.counts=ttk.Label(root,text='有效试次：静坐 0 / 心算 0')
        self.counts.pack(pady=10)
        self.prediction=ttk.Label(root,text='尚未开始预测',font=('Microsoft YaHei UI',22))
        self.prediction.pack(pady=16)
        ttk.Label(root,text='输出是实验任务类别，不是专注百分比或情绪诊断。\n模型尚未通过独立实验验证。',justify='center').pack()
        ttk.Label(root,text=str(session.directory),wraplength=750).pack(pady=12)
        self.closing=False
        self.predicting=bool(session.model_path)
        root.protocol('WM_DELETE_WINDOW',self.close)
        session.start()
        root.after(100,self.update)

    def close(self):
        self.closing=True
        self.session.stop.set()
        self.status.configure(text='正在停止线程并保存…')
        for b in self.buttons:b.configure(state='disabled')

    def update(self):
        while True:
            try: item=self.session.messages.get_nowait()
            except queue.Empty: break
            kind=item['kind']
            if kind=='counts':
                counts=item['counts']
                self.counts.configure(text=f'有效试次：静坐 {counts.get("rest",[0,0])[1]} / 心算 {counts.get("math",[0,0])[1]} · {item["quality"]}')
            elif kind=='prediction': self.prediction.configure(text=item['text'])
            else: self.status.configure(text=item.get('text',''))
            if kind in ('connected','trial_done') and not self.predicting and not self.closing:
                for b in self.buttons:b.configure(state='normal')
            if kind in ('trial','error','closed'):
                for b in self.buttons:b.configure(state='disabled')
            if kind=='mode' or (kind=='connected' and item.get('mode')=='predict'):
                self.predicting=True
                for b in self.buttons:b.configure(state='disabled')
        if self.closing and not self.session.thread.is_alive():
            self.root.destroy();return
        self.root.after(100,self.update)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true',help='内置模拟指标源；不连接真实设备')
    parser.add_argument('--model',type=Path,help='加载本地可信的 model.joblib 直接预测')
    args=parser.parse_args()
    root=tk.Tk()
    App(root,Session(demo=args.demo,model=args.model))
    root.mainloop()


if __name__=='__main__': main()
