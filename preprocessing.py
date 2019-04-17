#!/usr/bin/env python
# coding: utf-8 In[1]:
import numpy as np 
from rootpy.vector import LorentzVector 
import h5py
# ### Preprocessing
# 
# * center * rotate In[64]:
import copy 

def preprocessing(jet): # every entry would be a sequence of 4-vecs (E, px, py, pz) of jet constituents
    jet = copy.deepcopy(jet)
    jet=jet.reshape(-1,4)
    n_consti=len(jet)
    
    # find the jet (eta, phi)
    center=jet.sum(axis=0)
    
    v_jet=LorentzVector(center[1], center[2], center[3], center[0])
        
    # centering
    phi=v_jet.phi()
    bv = v_jet.boost_vector()
    bv.set_perp(0)
     
        
    # rotating
    weighted_phi=0
    weighted_eta=0
    for i in range(n_consti):
        if jet[i,0]<1e-10:
            continue
        v = LorentzVector(jet[i,1], jet[i,2], jet[i,3], jet[i,0])
        r=np.sqrt(v.phi()**2 + v.eta()**2)
        weighted_phi += v.phi() * v.E()/r
        weighted_eta += v.eta() * v.E()/r
    alpha = -np.arctan2(weighted_phi, weighted_eta)
        
    for i in range(n_consti):
        v = LorentzVector(jet[i,1], jet[i,2], jet[i,3], jet[i,0])
        v.rotate_z(-phi)
        v.boost(-bv)
        v.rotate_x(alpha)
        
        jet[i, 0]=v[3]
        jet[i, 1]=v[0]
        jet[i, 2]=v[1]
        jet[i, 3]=v[2]
    
    jet=jet.reshape(1,-1)
    
    return jet
