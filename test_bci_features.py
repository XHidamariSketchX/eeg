import unittest
import numpy as np
from bci_features import MetricFeatures

class FeatureTests(unittest.TestCase):
    def epoch(self):
        n=400
        x=np.zeros((n,6))
        x[:,:3]=[20,40,5]
        x[:,3]=1
        x[:,4]=np.arange(n)%256
        x[:,5]=100+np.arange(n)/50
        return x

    def test_independent_of_sr(self):
        f=MetricFeatures(8)
        x=self.epoch()
        a=f.ProcessFeatures(x,100)
        b=f.ProcessFeatures(x,0)
        np.testing.assert_allclose(a,b)
        self.assertEqual(a.shape,(12,))
        np.testing.assert_allclose(a[:3],[20,40,5])

    def test_invalid_windows(self):
        for kind in ['wear','gap','counter','nonfinite','short']:
            x=self.epoch()
            if kind=='wear':x[10,3]=0
            if kind=='gap':x[100:,5]+=1
            if kind=='counter':x[100:,4]=(x[100:,4]+1)%256
            if kind=='nonfinite':x[0,0]=np.nan
            if kind=='short':x=x[:10]
            with self.subTest(kind=kind):
                f=MetricFeatures(8)
                self.assertIsNone(f.ProcessFeatures(x,100))
                self.assertEqual(f.snapshot()[0],0)

if __name__=='__main__': unittest.main()
