import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('eeg_fft',Path(__file__).with_name('eeg-fft.py'))
fft=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fft)

class FFTTests(unittest.TestCase):
    def row(self, seq=0, wear=1):
        v=[0.0]*21
        v[0],v[2]=seq,wear
        v[6:8]=[2.0,3.0]
        return v

    def test_decoder_batch_and_partial(self):
        d=fft.LineDecoder()
        line=(','.join(map(str,self.row()))+'\r\n').encode()
        self.assertEqual(d.feed(line[:10]),[])
        self.assertEqual(len(d.feed(line[10:]+line+line[:5])),2)
        self.assertEqual(len(d.feed(line[5:])),1)
        d.feed(b'bad\n'+line.replace(b'2.0',b'nan'))
        self.assertEqual(d.invalid,2)

    def test_wear_gap_wrap_and_zero_rate(self):
        a=fft.Acquisition(SimpleNamespace(rows=32,wear_ok=1))
        now=time.monotonic()
        a.accept(self.row(255),now)
        a.accept(self.row(0),now)
        self.assertEqual(a.discontinuities,0)
        self.assertEqual(len(a.samples),2)
        a.accept(self.row(1,0),now)
        self.assertEqual(len(a.samples),0)
        a.accept(self.row(2),now)
        a.accept(self.row(4),now)
        self.assertEqual(len(a.samples),1)
        self.assertEqual(a.discontinuities,1)
        a.accept(self.row(5),now+2)
        self.assertEqual(len(a.samples),1)
        self.assertTrue(np.isfinite(a.snapshot()[2]))

    def test_psd_known_frequencies(self):
        t=np.arange(200)/50
        values=np.column_stack([np.sin(2*np.pi*10*t),np.sin(2*np.pi*6*t)])
        f,p=fft.estimate_psd(values,50)
        self.assertAlmostEqual(f[np.argmax(p[:,0])],10,delta=.4)
        self.assertAlmostEqual(f[np.argmax(p[:,1])],6,delta=.4)
        f,p=fft.estimate_psd(values)
        self.assertAlmostEqual(f[np.argmax(p[:,0])],.2,delta=.01)

if __name__=='__main__': unittest.main()
