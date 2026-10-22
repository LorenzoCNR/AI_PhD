"""Independent mathematical, numerical and data-boundary qualification."""
import copy
import unittest
import numpy as np
import torch
from user_soft.loss import evaluate_soft, fit_distance_scales
from user_soft.recovered_encoders import CEBRAOffset10Encoder, TemporalTransformerEncoder

SPECS = [dict(key="behavior", geometry="euclidean", weight=.5), dict(key="time_id", geometry="temporal", weight=.5)]

def metadata(n=8):
    return dict(behavior=torch.stack((torch.linspace(0,2,n),torch.arange(n).remainder(2).float()),1),time_id=torch.arange(n).float(),trial_id=torch.arange(n).remainder(2),session_id=torch.zeros(n,dtype=torch.long))

def field(result,key):
    return result[key] if isinstance(result,dict) else getattr(result,key)

class SoftQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(42)
        self.batch=metadata()
        self.scales=fit_distance_scales(self.batch,SPECS,split="train",source_ids=torch.arange(8))
        self.z=torch.randn(8,3,dtype=torch.float64,requires_grad=True)
    def call(self,z=None,batch=None,**kw):
        return evaluate_soft(self.z if z is None else z,self.batch if batch is None else batch,SPECS,scales=self.scales,**kw)
    def test_probability_normalization_and_diagonal(self):
        result=self.call()
        for key in ("Q","P"):
            matrix=field(result,key)
            torch.testing.assert_close(matrix.sum(1),torch.ones(8,dtype=matrix.dtype))
            self.assertEqual(torch.count_nonzero(matrix.diag()).item(),0)
            self.assertTrue(torch.isfinite(matrix).all())
    def test_independent_formula_frozen(self):
        result=self.call(version="soft_v1_frozen")
        db=torch.cdist(self.batch["behavior"].double(),self.batch["behavior"].double())
        dt=abs(self.batch["time_id"][:,None].double()-self.batch["time_id"][None,:].double())
        w=torch.exp(-(.5*db/db.max()+.5*dt/dt.max())/.5)
        w.fill_diagonal_(0)
        q=w/w.sum(1,keepdim=True)
        unit=self.z/self.z.norm(dim=1,keepdim=True)
        logits=unit@unit.T/.1
        logits.fill_diagonal_(-torch.inf)
        logp=logits-torch.logsumexp(logits,1,keepdim=True)
        mask=~torch.eye(8,dtype=torch.bool)
        reference=-(q[mask]*logp[mask]).sum()/8
        torch.testing.assert_close(field(result,"Q").double(),q,rtol=2e-6,atol=2e-7)
        torch.testing.assert_close(field(result,"loss"),reference,rtol=2e-6,atol=2e-7)
    def test_finite_nonzero_gradients_both_versions(self):
        for version in ("soft_v1_frozen","soft_v1_stable"):
            z=self.z.detach().clone().requires_grad_(True)
            loss=field(self.call(z=z,version=version),"loss")
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertGreater(z.grad.norm().item(),0)
    def test_double_gradcheck_stable(self):
        self.assertTrue(torch.autograd.gradcheck(lambda z:field(self.call(z=z),"loss"),(self.z,),eps=1e-6,atol=3e-5,rtol=3e-4))
    def test_uniform_targets_and_predictions(self):
        batch={key:torch.zeros_like(value) for key,value in self.batch.items()}
        scales=fit_distance_scales(batch,SPECS,split="train",source_ids=torch.arange(8))
        result=evaluate_soft(torch.ones(8,3),batch,SPECS,scales=scales)
        torch.testing.assert_close(field(result,"loss"),torch.tensor(np.log(7),dtype=torch.float32))
        mask=~torch.eye(8,dtype=torch.bool)
        torch.testing.assert_close(field(result,"Q")[mask],torch.full((56,),1/7))
    def test_concentrated_similarity_and_small_temperature(self):
        result=self.call(metadata_temperature=1e-8,temperature=1e-5)
        self.assertTrue(torch.isfinite(field(result,"loss")))
        torch.testing.assert_close(field(result,"Q").sum(1),torch.ones(8,dtype=field(result,"Q").dtype))
        field(result,"loss").backward()
        self.assertTrue(torch.isfinite(self.z.grad).all())
    def test_coherent_batch_permutation(self):
        permutation=torch.tensor([6,2,0,7,4,1,5,3])
        a=self.call()
        b=self.call(z=self.z[permutation],batch={key:value[permutation] for key,value in self.batch.items()})
        torch.testing.assert_close(field(a,"loss"),field(b,"loss"))
        for key in ("Q","P","W"):
            torch.testing.assert_close(field(a,key)[permutation][:,permutation],field(b,key))
    def test_official_cnn_and_recovered_encoders(self):
        import cebra.models
        official=cebra.models.init("offset10-model",num_neurons=4,num_units=32,num_output=32)
        cnn=CEBRAOffset10Encoder(n_features=4,embedding_dim=3)
        transformer=TemporalTransformerEncoder(n_features=4,embedding_dim=3,model_dim=8,n_heads=2,n_layers=1,dropout=0,max_length=10)
        neural=torch.randn(8,10,4)
        for model,x in ((official,neural.transpose(1,2)),(cnn,neural),(transformer,neural)):
            z=model(x)
            self.assertEqual(z.shape[0],8)
            field(self.call(z=z),"loss").backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0)
    def test_cross_trial_same_time_intentional(self):
        batch=metadata(4)
        batch["behavior"]=torch.tensor([[0.,1.],[0.,1.],[1.,0.],[1.,0.]])
        batch["time_id"]=torch.tensor([0.,0.,10.,10.])
        batch["trial_id"]=torch.tensor([10,11,10,11])
        scales=fit_distance_scales(batch,SPECS,split="train",source_ids=torch.arange(4))
        result=evaluate_soft(torch.randn(4,3),batch,SPECS,scales=scales)
        self.assertGreater(field(result,"Q")[0,1],field(result,"Q")[0,2])
        self.assertGreater(field(result,"Q")[0,1],0)
    def test_nontrain_scale_estimation_forbidden(self):
        for split in ("valid","validation","test"):
            with self.assertRaises((ValueError,RuntimeError)):
                fit_distance_scales(self.batch,SPECS,split=split)
        saved=copy.deepcopy(self.scales)
        extreme={key:value.clone() for key,value in self.batch.items()}
        extreme["behavior"]*=1e6
        self.call(batch=extreme)
        self.assertEqual(self.scales,saved)
    def test_numpy_training_ids_provenance_equivalent(self):
        scales=fit_distance_scales(self.batch,SPECS,split="train",source_ids=np.arange(8,dtype=np.int64))
        self.assertEqual(scales["train_id_sha256"],self.scales["train_id_sha256"])
    def test_seed_reproducibility(self):
        records=[]
        for _ in range(2):
            torch.manual_seed(123)
            encoder=torch.nn.Linear(4,3)
            neural=torch.randn(8,4)
            loss=field(self.call(z=encoder(neural)),"loss")
            optimizer=torch.optim.Adam(encoder.parameters(),lr=1e-3)
            loss.backward()
            optimizer.step()
            records.append((loss.detach(),torch.cat([p.detach().flatten() for p in encoder.parameters()])))
        for a,b in zip(*records):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_empty_candidate_rows_explicit(self):
        mask=torch.ones(8,8,dtype=torch.bool)
        mask[0]=False
        with self.assertRaises((ValueError,RuntimeError)):
            self.call(candidate_mask=mask)
        result=self.call(candidate_mask=mask,empty_rows="skip")
        self.assertFalse(field(result,"valid_rows")[0])
        self.assertEqual(field(result,"Q")[0].sum().item(),0)
        self.assertTrue(torch.isfinite(field(result,"loss")))
        with self.assertRaises((ValueError,RuntimeError)):
            evaluate_soft(torch.randn(1,3),metadata(1),SPECS,scales=self.scales)
    def test_invalid_inputs_fail(self):
        for value in (0.,-1.,float("nan"),float("inf")):
            with self.assertRaises((ValueError,RuntimeError)):
                self.call(temperature=value)
        bad=self.z.detach().clone()
        bad[0,0]=torch.nan
        with self.assertRaises((ValueError,RuntimeError)):
            self.call(z=bad)
    def test_pair_scale_invariance_batch_composition(self):
        bigger={key:torch.cat((value,value[-1:]),0) for key,value in self.batch.items()}
        bigger["behavior"][-1]=1000
        bigger["time_id"][-1]=1000
        z=torch.cat((self.z,torch.randn(1,3,dtype=torch.float64)))
        a,b=self.call(),self.call(z=z,batch=bigger)
        torch.testing.assert_close(field(a,"log_W")[0,1],field(b,"log_W")[0,1])
        a,b=self.call(version="soft_v1_frozen"),self.call(z=z,batch=bigger,version="soft_v1_frozen")
        self.assertGreater(abs((field(a,"log_W")[0,1]-field(b,"log_W")[0,1]).item()),.01)

    def test_stable_formula_independent_of_production_q(self):
        result=self.call()
        # Independently estimate train extents: Euclidean box diagonal and temporal range.
        behavior=self.batch["behavior"].double()
        times=self.batch["time_id"].double()
        extent=(behavior.max(0).values-behavior.min(0).values).norm()
        db=torch.cdist(behavior,behavior)/extent
        dt=abs(times[:,None]-times[None,:])/(times.max()-times.min())
        target_logits=-(.5*db+.5*dt)/.5
        target_logits.fill_diagonal_(-torch.inf)
        q=torch.softmax(target_logits,1)
        unit=self.z/self.z.norm(dim=1,keepdim=True)
        logits=unit@unit.T/.1
        logits.fill_diagonal_(-torch.inf)
        logp=torch.log_softmax(logits,1)
        mask=~torch.eye(8,dtype=torch.bool)
        reference=-(q[mask]*logp[mask]).sum()/8
        torch.testing.assert_close(field(result,"Q").double(),q,rtol=2e-6,atol=2e-7)
        torch.testing.assert_close(field(result,"loss"),reference,rtol=2e-6,atol=2e-7)


class TrialBoundaryQualification(unittest.TestCase):
    def rows(self,split="train",offset=0):
        from user_soft.data import TrialRows
        return TrialRows(neural=np.arange(120,dtype=np.float32).reshape(30,4),
                         behavior=np.ones((30,3),dtype=np.float32),
                         source_id=np.arange(offset,offset+30),
                         trial_id=np.repeat([offset+1,offset+2],15),time_id=np.tile(np.arange(15),2),
                         session_id=np.zeros(30,dtype=np.int64),split=split)
    def test_windows_never_cross_trial_boundary(self):
        from user_soft.data import eligible_centers,windows
        rows=self.rows()
        centers=eligible_centers(rows,left=5,right=5)
        x=windows(rows,centers,left=5,right=5)
        self.assertEqual(tuple(x.shape),(len(centers),4,10))
        for center in np.asarray(centers):
            self.assertTrue(np.all(rows.trial_id[center-5:center+5]==rows.trial_id[center]))
            self.assertEqual(len(rows.time_id[center-5:center+5]),10)
    def test_train_sampler_rejects_test_rows_and_is_seeded(self):
        from user_soft.data import train_batches
        with self.assertRaises((ValueError,RuntimeError)):
            next(iter(train_batches(self.rows("test"),batch_size=4,seed=42)))
        a=next(iter(train_batches(self.rows(),batch_size=4,seed=42)))
        b=next(iter(train_batches(self.rows(),batch_size=4,seed=42)))
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    def test_split_overlap_is_rejected(self):
        from user_soft.data import assert_disjoint
        assert_disjoint({"train":self.rows(),"valid":self.rows("valid",100),"test":self.rows("test",200)})
        with self.assertRaises((ValueError,RuntimeError,AssertionError)):
            assert_disjoint({"train":self.rows(),"test":self.rows("test",10)})
    def test_context_internal_metadata_discontinuity_rejected(self):
        from user_soft.data import eligible_centers,windows
        for key,new_value in (("trial_id",999),("session_id",999),("time_id",999)):
            rows=self.rows()
            getattr(rows,key)[8]=new_value
            self.assertNotIn(10,np.asarray(eligible_centers(rows,left=5,right=5)))
            with self.assertRaises((ValueError,RuntimeError)):
                windows(rows,np.array([10]),left=5,right=5)


class CpuEndToEndQualification(unittest.TestCase):
    def test_short_cpu_training_reproducible(self):
        from user_soft.smoke import run_toy
        first=run_toy(steps=30,seed=42)
        repeat=run_toy(steps=30,seed=42)
        self.assertLess(first["final_fixed_support_loss"],first["initial_fixed_support_loss"])
        self.assertEqual(first["actual_optimizer_updates"],30)
        self.assertEqual(first["parameter_sha256"],repeat["parameter_sha256"])
        self.assertEqual(first["history"],repeat["history"])


class DownstreamFirewallQualification(unittest.TestCase):
    def arrays(self):
        arrays={}
        rng=np.random.default_rng(42)
        for number,split in enumerate(("train","valid","test")):
            z=rng.normal(size=(48,3)).astype(np.float32)
            direction=(z[:,1]>0).astype(np.float32)
            arrays[split]=dict(embedding=z,behavior=np.column_stack((z[:,0],direction,1-direction)),
                               source_id=np.arange(number*100,number*100+48),
                               trial_id=np.repeat(np.arange(number*10,number*10+4),12),
                               time_id=np.tile(np.arange(12),4),valid_mask=np.ones(48,dtype=bool))
        return arrays
    def test_test_outcomes_do_not_select_ridge_alpha(self):
        from user_soft.downstream import score_arrays
        arrays=self.arrays()
        a=score_arrays(arrays)
        modified=copy.deepcopy(arrays)
        modified["test"]["behavior"][:,0]*=1e6
        b=score_arrays(modified)
        def choices(results):
            rows=results if isinstance(results,list) else results.get("rows",results.get("metrics",[]))
            return sorted((str(row.get("representation")),row.get("selected_alpha"))
                          for row in rows if row.get("method",row.get("decoder"))=="ridge" and row.get("split")=="valid")
        self.assertTrue(choices(a))
        self.assertEqual(choices(a),choices(b))
    def test_downstream_source_overlap_rejected(self):
        from user_soft.downstream import score_arrays
        arrays=self.arrays()
        arrays["test"]["source_id"]=arrays["train"]["source_id"].copy()
        with self.assertRaises((ValueError,RuntimeError,AssertionError)):
            score_arrays(arrays)

if __name__=="__main__":
    unittest.main()
