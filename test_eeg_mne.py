import unittest
import numpy as np
from eeg_mne import quality, compute_spectrum, make_raw

class MNETests(unittest.TestCase):
    def data(self,n=512):
        t=np.arange(n)/50
        x=np.sin(2*np.pi*10*t)
        return {'host_monotonic_seconds':t,'packet_counter':np.arange(n)%256,
                'wear_detection':np.ones(n),'ADC1_filtered':x,'ADC2_filtered':x*2,
                **{k:np.ones(n) for k in ['theta','alpha','beta','attention','relaxation']}}

    def test_psd_axes(self):
        d=self.data()
        f,p,count=compute_spectrum(d,[(0,512)],50)
        self.assertAlmostEqual(f[p[0].argmax()],10,delta=.4)
        self.assertEqual(count,7)
        f,p,_=compute_spectrum(d,[(0,512)])
        self.assertAlmostEqual(f[p[0].argmax()],.2,delta=.01)
        self.assertIsNone(compute_spectrum(d,[(0,50)]))

    def test_segmentation_and_annotations(self):
        d=self.data()
        d['wear_detection'][100:150]=0
        d['packet_counter'][300:]=(d['packet_counter'][300:]+1)%256
        bad,boundaries,segments=quality(d)
        self.assertEqual(segments,[(0,100),(150,300),(300,512)])
        raw=make_raw(d,50)
        self.assertTrue(all(t=='misc' for t in raw.get_channel_types()))
        self.assertEqual(set(raw.annotations.description),{'BAD_wear','BAD_boundary'})
        self.assertIn('ASSUMED',raw.info['description'])

if __name__=='__main__': unittest.main()
