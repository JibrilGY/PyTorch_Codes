"""Offline behavioral checks; do not download data or run the 50-epoch cell.
Run: .venv/bin/python -m unittest discover -s tests -v
"""
import ast
import json
import unittest
from pathlib import Path
import tempfile
import math
import random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from tqdm.auto import tqdm


def definitions():
    notebook = json.loads((Path(__file__).resolve().parents[1] / '07_YOLOv1.ipynb').read_text())
    ns = dict(globals())
    ns.update(S=7, B=2, IMAGE_SIZE=448, SEED=42)
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.lstrip().startswith('%'):
            continue
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook>', 'exec'), ns)
    return ns


class YoloChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ns = definitions()

    def test_training_components_present(self):
        for name in ['encode_target', 'YoloV1Loss', 'RoadSignDataset', 'decode_predictions', 'mean_ap50', 'run_epoch']:
            self.assertIn(name, self.ns)

    def test_grid_encoding_and_collision(self):
        encode = self.ns['encode_target']
        boxes = torch.tensor([[.2,.3,.4,.5], [.25,.35,.35,.45]])
        target, dropped = encode(boxes, torch.tensor([2,1]), 4)
        self.assertEqual(tuple(target.shape), (7,7,9))
        self.assertEqual(dropped, 1)
        self.assertEqual(target[...,4].sum().item(), 1)
        # center=(.3,.4), cell=(row 2,col 2), offsets=(.1,.8)
        torch.testing.assert_close(target[2,2,:4], torch.tensor([.1,.8,.2,.2]))
        self.assertEqual(target[2,2,7].item(), 1)
        edge,_ = encode(torch.tensor([[.98,.98,1.,1.]]), torch.tensor([0]),4)
        self.assertEqual(edge[6,6,4].item(),1)

    def test_loss_perfect_prediction_and_negative_width_gradient(self):
        loss_fn = self.ns['YoloV1Loss'](num_classes=4)
        t = torch.zeros(1,7,7,9)
        t[0,2,3,:] = torch.tensor([.5,.5,.2,.3,1,0,1,0,0])
        p = torch.zeros(1,7,7,14)
        p[0,2,3,:5] = t[0,2,3,:5]
        p[0,2,3,10:] = t[0,2,3,5:]
        value,_ = loss_fn(p,t)
        self.assertLess(value.item(),1e-6)
        # Unassigned box confidence is penalized even in an occupied cell.
        p[0,2,3,9] = .7
        value,_ = loss_fn(p,t)
        self.assertAlmostEqual(value.item(),.5*.7**2,places=5)
        p = torch.full((1,7,7,14),-.2,requires_grad=True)
        value,_ = loss_fn(p,t);value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(p.grad).all())
        empty,_ = loss_fn(torch.zeros_like(p),torch.zeros_like(t))
        self.assertEqual(empty.item(),0)

    def test_decoder_nms_and_ap(self):
        p=torch.zeros(1,7,7,14)
        p[0,2,3,:5]=torch.tensor([.5,.5,.2,.2,1.])
        p[0,2,3,5:10]=p[0,2,3,:5]
        p[0,2,3,10]=1
        detections=self.ns['decode_predictions'](p,4,score_threshold=.01)
        self.assertEqual(len(detections[0]['boxes']),1)
        truth={'boxes':torch.tensor([[.4,2.5/7-.1,.6,2.5/7+.1]]),'labels':torch.tensor([0])}
        score,_=self.ns['mean_ap50'](detections,[truth],4)
        self.assertAlmostEqual(score,1.0,places=5)
        detections[0]['labels']=torch.tensor([1])
        score,_=self.ns['mean_ap50'](detections,[truth],4)
        self.assertEqual(score,0)

    def test_dataset_xml_and_resize(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'images').mkdir();(root/'annotations').mkdir()
            Image.new('RGB',(100,80)).save(root/'images'/'road0.png')
            (root/'annotations'/'road0.xml').write_text('''<annotation><filename>road0.png</filename><size><width>100</width><height>80</height></size><object><name>stop</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object></annotation>''')
            records,names=self.ns['read_records'](root)
            self.assertEqual(names,['stop'])
            ds=self.ns['RoadSignDataset'](records,names,training=False)
            image,target,truth=ds[0]
            self.assertEqual(tuple(image.shape),(3,448,448))
            self.assertEqual(tuple(target.shape),(7,7,6))
            torch.testing.assert_close(truth['boxes'],torch.tensor([[.1,.25,.5,.75]]))

    def test_actual_model_shape_without_allocating_weights(self):
        with torch.device('meta'):
            model=self.ns['YOLOv1'](num_classes=4)
            result=model(torch.empty(2,3,448,448))
        self.assertEqual(tuple(result.shape),(2,7,7,14))

    def test_training_step_and_eval_no_update(self):
        # Test the actual loop using a small model, not a mocked optimizer/loss.
        class SmallDetector(nn.Module):
            def __init__(self):
                super().__init__();self.head=nn.Conv2d(3,14,1)
            def forward(self,x):
                return self.head(F.adaptive_avg_pool2d(x,(7,7))).permute(0,2,3,1)
        model=SmallDetector();opt=torch.optim.SGD(model.parameters(),lr=.01)
        loss_fn=self.ns['YoloV1Loss'](num_classes=4)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        truth={'boxes':torch.tensor([[.1,.1,.3,.3]]),'labels':torch.tensor([0])}
        target,_=self.ns['encode_target'](truth['boxes'],truth['labels'],4)
        batch=[(torch.rand(2,3,16,16),torch.stack([target,target]),[truth,truth])]
        before=model.head.weight.detach().clone()
        result=self.ns['run_epoch'](model,batch,loss_fn,torch.device('cpu'),opt,scaler,4)
        self.assertTrue(math.isfinite(result['loss']))
        self.assertFalse(torch.equal(before,model.head.weight))
        before=model.head.weight.detach().clone()
        result=self.ns['run_epoch'](model,batch,loss_fn,torch.device('cpu'),None,scaler,4)
        self.assertTrue(torch.equal(before,model.head.weight))
        self.assertTrue(0<=result['map50']<=1)

if __name__=='__main__': unittest.main()
